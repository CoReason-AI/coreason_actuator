# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import asyncio
import concurrent.futures
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from coreason_manifest.spec.ontology import (
    BrowserDOMState,
    MCPServerManifest,
    ObservationEvent,
    ToolInvocationEvent,
    ToolManifest,
)
from coreason_manifest.utils.algebra import verify_ast_safety
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from coreason_actuator.interfaces import IPCBrokerProtocol, KinematicBrowserProtocol
from coreason_actuator.semantic_extractor import TensorStorageProtocol
from coreason_actuator.utils.logger import logger


class ExecutionStrategyProtocol(Protocol):
    """Protocol defining the interface for execution strategies."""

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        """Executes the specific protocol logic."""
        ...


class DistributedLockProtocol(Protocol):
    """Protocol for acquiring exclusive execution locks."""

    @asynccontextmanager
    async def acquire(self, lock_key: str, ttl: int | None = None) -> AsyncIterator[Any]:
        """Acquires a lock for the given key."""
        _ = lock_key  # pragma: no cover
        _ = ttl  # pragma: no cover
        yield  # pragma: no cover


class PostgresDistributedLock:
    """Postgres advisory lock implementation for multi-process distributed synchronization."""

    def __init__(self, pool: Any) -> None:
        # pool is expected to be an asyncpg.Pool
        self.pool = pool

    def _hash_lock_key(self, lock_key: str) -> int:
        """Convert a string lock key into a 64-bit integer for pg_advisory_lock."""
        # Using a stable hash to get an int64 for Postgres
        hash_bytes = hashlib.sha256(lock_key.encode("utf-8")).digest()
        # Take first 8 bytes and convert to signed 64-bit int
        return int.from_bytes(hash_bytes[:8], byteorder="big", signed=True)

    @asynccontextmanager
    async def acquire(self, lock_key: str, ttl: int | None = None) -> AsyncIterator[Any]:
        lock_id = self._hash_lock_key(lock_key)

        async with self.pool.acquire() as conn:
            if ttl is not None:
                # To simulate acquisition timeout / TTL, we would normally use try_advisory_lock
                # with a timeout loop, or simply set lock_timeout locally.
                timeout_ms = int(ttl)
                await conn.execute(f"SET LOCAL lock_timeout = '{timeout_ms}ms'")

            try:
                await conn.execute("SELECT pg_advisory_lock($1)", lock_id)
            except Exception as err:
                if "lock_not_available" in str(err).lower() or "timeout" in str(err).lower():
                    raise TimeoutError(f"Failed to acquire lock for {lock_key} within TTL {ttl}ms") from err
                raise err

            try:
                yield
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", lock_id)


class NativeRegistryProtocol(Protocol):
    """Protocol for the internal asynchronous Python registry."""

    async def get_callable(self, tool_name: str) -> Callable[..., Awaitable[Any]] | None:
        """Retrieves the registered asynchronous callable for the tool."""
        ...


class MCPServerRegistryProtocol(Protocol):
    """Protocol for discovering MCPServerManifest definitions."""

    async def get_server_manifest(self, tool_name: str) -> MCPServerManifest | None:
        """Retrieves the server manifest for dynamically discovered tools."""
        ...


class MCPTransportProtocol(Protocol):
    """Protocol abstracting the MCP transport layer (stdio, sse, http)."""

    async def dispatch(self, server_manifest: MCPServerManifest, packet: dict[str, Any]) -> Any:
        """Dispatches a JSON-RPC packet over the appropriate transport layer."""
        ...


class NativeExecutionStrategy:
    """Strategy for executing native asynchronous Python callables."""

    def __init__(self, registry: NativeRegistryProtocol, lock_manager: DistributedLockProtocol):
        self.registry = registry
        self.lock_manager = lock_manager

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        """
        Executes the native python function.

        Evaluates AST safety for dynamic python code.
        Handles idempotency with Tenacity retries.
        Handles state mutations by acquiring distributed locks.
        """
        _ = sandbox_pid  # Unused for native executions, but required by protocol
        tool_callable = await self.registry.get_callable(intent.tool_name)
        if not tool_callable:
            raise ValueError(f"Tool {intent.tool_name} not found in native registry.")

        # Crucial Security Gate: AST Safety Verification
        # If the intent parameters contain any dynamic python payload string (e.g. for eval/exec)
        # we must mathematically guarantee no OS-level imports or fork commands can bypass container bounds.
        # Here we assume the tool intent parameters themselves might contain such payloads.
        # In a generic way, we stringify the parameters and check them.
        # Based on FRD: "before executing any dynamically provided native Python strings,
        # the Actuator MUST pass the payload through verify_ast_safety()"

        # We check all string values in the parameters for AST safety.
        params = intent.parameters or {}
        for key, value in params.items():
            # We attempt to parse and verify it as an AST
            if isinstance(value, str):
                try:
                    if not verify_ast_safety(value):
                        import traceback

                        return ObservationEvent(
                            event_id=f"obs-{intent.event_id}",
                            timestamp=1704067200.0,
                            payload={
                                "execution_status": "fatal_crash",
                                "traceback": f"AST safety verification failed for parameter '{key}'.",
                            },
                        )
                except ValueError as e:
                    if str(e) == "Payload is not valid syntax.":
                        continue
                    # Any other parsing or value errors from verify_ast_safety should also be trapped
                    import traceback

                    return ObservationEvent(
                        event_id=f"obs-{intent.event_id}",
                        timestamp=1704067200.0,
                        payload={
                            "execution_status": "fatal_crash",
                            "traceback": traceback.format_exc(),
                        },
                    )
                except Exception:
                    # Broad catch-all for AST parsing failures to prevent bubbling unhandled exceptions
                    import traceback

                    return ObservationEvent(
                        event_id=f"obs-{intent.event_id}",
                        timestamp=1704067200.0,
                        payload={
                            "execution_status": "fatal_crash",
                            "traceback": traceback.format_exc(),
                        },
                    )

        # Idempotency Rules
        side_effects = manifest.side_effects

        # Determine if we need to lock
        mutates_state = getattr(side_effects, "mutates_state", False)
        is_idempotent = getattr(side_effects, "is_idempotent", False)

        async def _do_execute() -> Any:
            # The actual execution of the callable
            # We pass the unpacked params. The tool_callable is expected to accept these as kwargs.
            return await tool_callable(**params)

        # If mutates_state == True:
        # Bypass local HTTP/DNS caches (handled at lower network layers or via env vars ideally)
        # Disable retry loops
        # Acquire exclusive distributed lock
        if mutates_state:
            payload_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
            # Lock key MUST be formulated using specific cryptographic nonce of execution
            lock_key = f"lock:{intent.tool_name}:{payload_hash}:{intent.event_id}"
            # Lock TTL MUST be mathematically bound to ExecutionSLA.max_execution_time_ms
            ttl = manifest.sla.max_execution_time_ms if manifest.sla else None

            logger.info(f"Acquiring exclusive execution lock for {lock_key} with ttl {ttl}ms")
            async with self.lock_manager.acquire(lock_key, ttl=ttl):
                return await _do_execute()

        # If is_idempotent == True (and not mutates_state, mutates_state takes precedence)
        # Wrap in Tenacity retry loop for network faults.
        if is_idempotent:
            # We define a retry-wrapped function
            # Retrying on specific transient faults
            @retry(
                wait=wait_exponential(multiplier=1, min=1, max=10),
                stop=stop_after_attempt(3),
                retry=retry_if_exception_type((ConnectionError, TimeoutError)),
                reraise=True,
            )
            async def _retry_execute() -> Any:
                return await _do_execute()

            return await _retry_execute()

        # Default execution path
        return await _do_execute()


class MCPClientStrategy:
    """Strategy for dispatching intents to remote MCP servers."""

    def __init__(self, registry: MCPServerRegistryProtocol, transport: MCPTransportProtocol):
        self.registry = registry
        self.transport = transport

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        """
        Executes the MCP protocol dispatch.

        Instantiates an ephemeral client connection, negotiates transport, and dispatches a JSON-RPC 2.0 packet.
        """
        _ = manifest  # Used for permissions/SLA in the orchestrator or broader context
        _ = sandbox_pid  # MCP connections might originate from the actuator daemon directly

        server_manifest = await self.registry.get_server_manifest(intent.tool_name)
        if not server_manifest:
            raise ValueError(f"MCPServerManifest not found for tool: {intent.tool_name}")

        # Construct strictly compliant JSON-RPC 2.0 packet
        packet = {
            "jsonrpc": "2.0",
            "id": intent.event_id,
            "method": intent.tool_name,
            "params": intent.parameters or {},
        }

        # Dispatch the packet via the ephemeral transport connection
        return await self.transport.dispatch(server_manifest, packet)


class KinematicExecutionStrategy:
    """Strategy for translating spatial intent into headless browser automation."""

    def __init__(self, browser: KinematicBrowserProtocol, tensor_storage: TensorStorageProtocol | None = None):
        self.browser = browser
        self.tensor_storage = tensor_storage

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        """
        Executes the kinematic interaction using purely atomic locators.

        Binds the semantic verification of the expected_visual_concept to the physical action
        in a single, inseparable operation to eliminate Time-Of-Check to Time-Of-Use (TOCTOU) race conditions.
        """
        _ = manifest
        _ = sandbox_pid

        params = intent.parameters or {}
        x = params.get("x")
        y = params.get("y")
        expected_visual_concept = params.get("expected_visual_concept")
        action = params.get("action")
        timeout = params.get("timeout", 100)

        if x is None or y is None or expected_visual_concept is None or action is None:
            raise ValueError(
                "Kinematic interaction requires 'x', 'y', 'expected_visual_concept', and 'action' parameters."
            )

        try:
            x_float = float(x)
            y_float = float(y)
            timeout_int = int(timeout)
        except (ValueError, TypeError) as err:
            raise ValueError("Coordinates 'x', 'y' and 'timeout' must be valid numbers.") from err

        expected_visual_concept_str = str(expected_visual_concept)

        # Execute Atomic Action
        if action == "click":
            logger.info(
                f"Executing atomic physical click at ({x_float}, {y_float}) "
                f"for concept '{expected_visual_concept_str}' with timeout {timeout_int}ms"
            )
            await self.browser.click(x_float, y_float, expected_visual_concept_str, timeout_int)
        elif action == "type_text":
            text = params.get("text")
            if text is None:
                raise ValueError("Kinematic interaction 'type_text' requires a 'text' parameter.")
            logger.info(
                f"Executing atomic physical type_text at ({x_float}, {y_float}) "
                f"for concept '{expected_visual_concept_str}' with timeout {timeout_int}ms"
            )
            await self.browser.type_text(x_float, y_float, str(text), expected_visual_concept_str, timeout_int)
        else:
            raise ValueError(f"Unsupported kinematic action: {action}")

        # Capture Post-Action State (Screenshot/DOM hash)
        current_url = await self.browser.get_current_url()
        viewport_size = await self.browser.get_viewport_size()
        dom_hash = await self.browser.get_dom_hash()

        # Get screenshot bytes
        img_bytes = await self.browser.capture_viewport_screenshot()

        screenshot_cid = None
        if self.tensor_storage is not None:
            from collections.abc import AsyncGenerator

            async def _stream_screenshot() -> AsyncGenerator[bytes]:
                yield img_bytes

            # Stream directly to cold storage via storage protocol
            screenshot_cid = await self.tensor_storage.stream_to_storage(_stream_screenshot())

        # The accessibility_tree_hash is populated with a static value indicating it's deprecated/bypassed
        # in favor of pure Atomic Locators per architectural constraints.
        return BrowserDOMState(
            current_url=current_url,
            viewport_size=viewport_size,
            dom_hash=dom_hash,
            accessibility_tree_hash="[DEPRECATED_BY_ATOMIC_LOCATORS]",
            screenshot_cid=screenshot_cid,
        )


class BackgroundPollingStrategy:
    """
    Wrapper strategy that executes the underlying execution strategy based on the Hardware SLA Evaluation.
    If the max execution time < 30,000ms, it executes synchronously.
    Otherwise, it executes as a background task, immediately yielding a pending ObservationEvent.
    """

    def __init__(
        self,
        execution_strategy: ExecutionStrategyProtocol,
        broker: IPCBrokerProtocol,
        executor: concurrent.futures.Executor | None = None,
    ):
        self.execution_strategy = execution_strategy
        self.broker = broker
        self.executor = executor or concurrent.futures.ProcessPoolExecutor()

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        sla = manifest.sla
        max_time_ms = sla.max_execution_time_ms if sla else None

        if max_time_ms is None or max_time_ms < 30000:
            logger.info(f"Executing tool {intent.tool_name} synchronously (SLA: {max_time_ms}ms)")
            return await self.execution_strategy.execute(intent, manifest, sandbox_pid)

        logger.info(f"Executing tool {intent.tool_name} asynchronously (SLA: {max_time_ms}ms)")
        job_id = str(uuid.uuid4())

        # Spawn the background task
        # In a real environment, this could use a ProcessPoolExecutor.
        # Here we use asyncio.create_task for the async underlying strategy.
        task = asyncio.create_task(self._background_execute(intent, manifest, sandbox_pid, job_id))

        # Keep a reference to prevent garbage collection
        if not hasattr(self, "_background_tasks"):
            self._background_tasks = set()
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Return a strictly typed ObservationEvent indicating pending execution
        return ObservationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            type="observation",
            payload={"status": "pending_async_execution", "job_id": job_id},
            triggering_invocation_id=intent.event_id,
        )

    @staticmethod
    def _run_sync_in_executor(
        execution_strategy: ExecutionStrategyProtocol,
        intent: ToolInvocationEvent,
        manifest: ToolManifest,
        sandbox_pid: Any,
    ) -> Any:
        """Synchronously runs the async execution strategy. Designed to be run in an executor."""
        return asyncio.run(execution_strategy.execute(intent, manifest, sandbox_pid))

    async def _background_execute(
        self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any, job_id: str
    ) -> None:
        """Executes the task in the background and pushes the final observation to the broker."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor, self._run_sync_in_executor, self.execution_strategy, intent, manifest, sandbox_pid
            )
            # Wrap the successful result in an ObservationEvent
            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                payload={"status": "completed", "job_id": job_id, "result": result},
                triggering_invocation_id=intent.event_id,
            )
            logger.info(f"Background execution completed for job {job_id}")
        except Exception as e:
            # Handle the error and wrap in an ObservationEvent
            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                payload={
                    "status": "fatal_crash",
                    "job_id": job_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                triggering_invocation_id=intent.event_id,
            )
            logger.error(f"Background execution failed for job {job_id}: {e}")

        # Push the final observation back to the IPC broker
        await self.broker.push(observation.model_dump())
