# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from coreason_manifest.spec.ontology import ToolInvocationEvent, ToolManifest
from coreason_manifest.utils.algebra import verify_ast_safety
from tenacity import retry, stop_after_attempt, wait_exponential

from coreason_actuator.utils.logger import logger


class ExecutionStrategyProtocol(Protocol):
    """Protocol defining the interface for execution strategies."""

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        """Executes the specific protocol logic."""
        ...


class DistributedLockProtocol(Protocol):
    """Protocol for acquiring exclusive execution locks."""

    @asynccontextmanager
    async def acquire(self, lock_key: str) -> AsyncIterator[Any]:
        """Acquires a lock for the given key."""
        _ = lock_key  # pragma: no cover
        yield  # pragma: no cover


class NativeRegistryProtocol(Protocol):
    """Protocol for the internal asynchronous Python registry."""

    async def get_callable(self, tool_name: str) -> Callable[..., Awaitable[Any]] | None:
        """Retrieves the registered asynchronous callable for the tool."""
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
            # If verify_ast_safety returns False, we raise an exception
            if isinstance(value, str):
                try:
                    if not verify_ast_safety(value):
                        raise ValueError(f"AST safety verification failed for parameter '{key}'.")
                except ValueError as e:
                    if str(e) == "Payload is not valid syntax.":
                        continue
                    raise e

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
            lock_key = f"{intent.tool_name}:{payload_hash}"
            logger.info(f"Acquiring exclusive execution lock for {lock_key}")
            async with self.lock_manager.acquire(lock_key):
                return await _do_execute()

        # If is_idempotent == True (and not mutates_state, mutates_state takes precedence)
        # Wrap in Tenacity retry loop for network faults.
        if is_idempotent:
            # We define a retry-wrapped function
            # Retrying on general Exceptions for demonstration. In a real scenario, this might be NetworkError etc.
            @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)  # type: ignore[misc]
            async def _retry_execute() -> Any:
                return await _do_execute()

            return await _retry_execute()

        # Default execution path
        return await _do_execute()
