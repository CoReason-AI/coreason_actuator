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
import time
import traceback
import uuid
from typing import Any

from coreason_manifest.spec.ontology import (
    BackpressurePolicy,
    JSONRPCErrorResponseState,
    JSONRPCErrorState,
    ObservationEvent,
    ToolInvocationEvent,
)

from coreason_actuator.ingress import IPCValidator
from coreason_actuator.interfaces import ActionSpaceRegistryProtocol, IPCBrokerProtocol
from coreason_actuator.sandbox import (
    EnterpriseVaultProtocol,
    SandboxProviderProtocol,
    StatefulSandboxCache,
    enforce_sandbox_immutability,
    verify_network_access,
)
from coreason_actuator.security import MaskingFunctor
from coreason_actuator.semantic_extractor import SemanticExtractor
from coreason_actuator.strategies import ExecutionStrategyProtocol
from coreason_actuator.utils.logger import logger


def scrub_payload(data: Any, masking_functor: MaskingFunctor) -> Any:
    """Recursively scrub string values in any JSON primitive or nested structure."""
    if isinstance(data, str):
        return masking_functor.redact(data)
    if isinstance(data, dict):
        return {
            (masking_functor.redact(k) if isinstance(k, str) else k): scrub_payload(v, masking_functor)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [scrub_payload(item, masking_functor) for item in data]
    return data


class ActuatorDaemon:
    """The continuous worker daemon subscribing to an IPC message broker."""

    def __init__(
        self,
        broker: IPCBrokerProtocol,
        validator: IPCValidator,
        backpressure_policy: BackpressurePolicy,
        execution_strategy: ExecutionStrategyProtocol,
        registry: ActionSpaceRegistryProtocol,
        vault: EnterpriseVaultProtocol | None = None,
        semantic_extractor: SemanticExtractor | None = None,
        sandbox_cache: StatefulSandboxCache | None = None,
    ) -> None:
        self.broker = broker
        self.validator = validator
        self.backpressure_policy = backpressure_policy
        self.execution_strategy = execution_strategy
        self.registry = registry
        self.vault = vault
        self.semantic_extractor = semantic_extractor
        self.sandbox_cache = sandbox_cache
        self.active_tasks_count = 0
        self._is_running = False
        self._running_task: asyncio.Task[Any] | None = None
        self.active_tasks: dict[str, asyncio.Task[Any]] = {}
        self.preempted_events: set[str] = set()
        self.active_sandboxes: dict[str, SandboxProviderProtocol] = {}
        self.masking_functor: MaskingFunctor | None = None

    def set_masking_functor(self, masking_functor: MaskingFunctor) -> None:
        """Sets the masking functor to be used for secret scrubbing."""
        self.masking_functor = masking_functor

    @property
    def is_running(self) -> bool:
        """Returns True if the daemon has been assigned a running task."""
        return self._running_task is not None

    def register_task(self, task: asyncio.Task[Any]) -> None:
        """Registers the main daemon loop task."""
        self._running_task = task

    async def start(self) -> None:
        """Starts the main polling loop of the IPC Daemon."""
        self._is_running = True
        logger.info("Actuator daemon started. Polling for intents...")
        while self._is_running:
            await self.run_once()
            # Small yield to prevent event loop starvation if the broker is extremely fast.
            await asyncio.sleep(0.001)

    def stop(self) -> None:
        """Gracefully halts the daemon loop."""
        self._is_running = False
        logger.info("Actuator daemon stopped.")

    async def run_once(self) -> None:
        """Pulls a single payload from the IPC broker and validates it."""
        try:
            raw_payload = await self.broker.pull()
        except Exception as e:
            logger.error(f"Error pulling from broker: {e}")
            return

        # Preemption check (BargeInInterruptEvent can come as raw JSON)
        if isinstance(raw_payload, dict) and raw_payload.get("type") == "barge_in":
            target_event_id = raw_payload.get("target_event_id")
            if target_event_id:
                await self._handle_preemption(str(target_event_id))
            return

        # Check backpressure limits
        max_concurrent = self.backpressure_policy.max_concurrent_tool_invocations
        if max_concurrent is not None and self.active_tasks_count >= max_concurrent:
            logger.warning(
                "Backpressure limit reached. Shedding load.",
                active=self.active_tasks_count,
                max=max_concurrent,
            )
            # Yield error response immediately to the broker ingress queue (or response queue)
            error_response = JSONRPCErrorResponseState(
                jsonrpc="2.0",
                id=raw_payload.get("id", None),
                error=JSONRPCErrorState(
                    code=429,
                    message="Too Many Requests: Daemon execution pool is saturated.",
                    data={"active_tasks": self.active_tasks_count},
                ),
            )
            await self.broker.push(error_response.model_dump())
            return

        # Pre-Flight Validation
        result = self.validator.validate_intent(raw_payload)

        if isinstance(result, JSONRPCErrorResponseState):
            logger.error("Pre-flight validation failed", payload_id=result.id, error=result.error.message)
            await self.broker.push(result.model_dump())
            return

        # Execute the intent concurrently
        task = asyncio.create_task(self._dispatch_intent(result, raw_payload.get("id")))
        self.active_tasks[result.event_id] = task

    async def _handle_preemption(self, target_event_id: str) -> None:
        """Handles a BargeInInterruptEvent by attempting to cancel the active task."""
        logger.info(f"Received preemption signal for event: {target_event_id}")
        task = self.active_tasks.get(target_event_id)
        if not task:
            logger.warning(f"No active task found for preemption target: {target_event_id}")
            return

        self.preempted_events.add(target_event_id)
        logger.info(f"Cancelling task for event: {target_event_id}")
        task.cancel()

    async def _dispatch_intent(self, intent: ToolInvocationEvent, request_id: Any) -> None:
        """
        Executes the tool intent via the Do-Operator and returns an ObservationEvent.
        """
        self.active_tasks_count += 1
        logger.info(f"Dispatched tool invocation: {intent.tool_name}", request_id=request_id)

        try:
            # Get manifest from registry to pass to execution strategy
            manifest = self.registry.get_tool(intent.tool_name)
            if not manifest:
                logger.error(f"ToolManifest not found for tool: {intent.tool_name}")
                error_response = JSONRPCErrorResponseState(
                    jsonrpc="2.0",
                    id=request_id,
                    error=JSONRPCErrorState(
                        code=-32601,
                        message=f"Method not found: Tool '{intent.tool_name}' missing from the registry.",
                    ),
                )
                await self.broker.push(error_response.model_dump())
                return

            # Execute Do-Operator
            # Retrieve sandbox if assigned (e.g., dynamically by earlier orchestration/provisioning).
            sandbox = self.active_sandboxes.get(intent.event_id)

            # Extract session state from the state_hydration natively bound to intent
            if hasattr(intent, "state_hydration") and intent.state_hydration is not None:
                session_state = getattr(intent.state_hydration, "session_state", None)
                partition_state = getattr(intent.state_hydration, "partition_state", None)

                # Provision or get sandbox via StatefulSandboxCache if available
                if self.sandbox_cache and session_state and partition_state:
                    sandbox = await self.sandbox_cache.get_or_create(session_state, partition_state)
                    self.active_sandboxes[intent.event_id] = sandbox

                    # Apply Dual-Evaluation Permission Boundary and immutability checks
                    verify_network_access(manifest, partition_state, sandbox)
                    enforce_sandbox_immutability(manifest, sandbox)

                # Based on FR-2.3, if allowed_vault_keys is present, unseal secrets and inject into sandbox
                if session_state and getattr(session_state, "allowed_vault_keys", None) and self.vault:
                    logger.info(f"Unsealing secrets for keys: {session_state.allowed_vault_keys}")
                    secrets = await self.vault.unseal(session_state.allowed_vault_keys)
                    if sandbox:
                        sandbox.inject_secrets(secrets)

            sandbox_pid = sandbox

            if not manifest.is_preemptible:
                inner_task = asyncio.create_task(self.execution_strategy.execute(intent, manifest, sandbox_pid))
                try:
                    result = await asyncio.shield(inner_task)
                except asyncio.CancelledError:
                    # Outer task was cancelled (barge-in received).
                    # But since it's not preemptible, we must wait for it to finish!
                    logger.info(
                        f"Task for {intent.event_id} was preempted, but is not preemptible. Waiting for completion."
                    )
                    result = await inner_task
                    # We must emit completed_under_preemption
                    self.preempted_events.add(intent.event_id)
            else:
                # Preemptible
                result = await self.execution_strategy.execute(intent, manifest, sandbox_pid)

            # ZK Proof extraction
            zk_receipt = None
            from coreason_actuator.sandbox import RiscvZkvmSandboxProvider

            if (
                isinstance(sandbox_pid, RiscvZkvmSandboxProvider) and isinstance(result, tuple) and len(result) == 2
            ):  # pragma: no cover
                import hashlib

                from coreason_manifest.spec.ontology import ZeroKnowledgeReceipt

                actual_result, blob_dict = result
                result = actual_result
                zk_receipt = ZeroKnowledgeReceipt.model_construct(
                    proof_protocol="zk-STARK",
                    public_inputs_hash=hashlib.sha256(b"mock").hexdigest(),
                    verifier_key_id="mock",
                    cryptographic_blob=blob_dict,
                )

            # Successful Execution
            payload = {"result": result} if not isinstance(result, dict) else result

            if self.masking_functor:
                payload = scrub_payload(payload, self.masking_functor)

            truncation_metadata = None
            if isinstance(payload, dict) and "truncation_metadata" in payload:
                # If the result itself contains truncation_metadata natively
                truncation_metadata = payload.pop("truncation_metadata")

            if self.semantic_extractor:
                payload, ext_metadata = self.semantic_extractor.truncate_payload(payload)
                if ext_metadata:
                    if truncation_metadata:  # pragma: no cover
                        truncation_metadata["items_omitted"] += ext_metadata.get("items_omitted", 0)
                    else:
                        truncation_metadata = ext_metadata

            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                payload=payload,
                triggering_invocation_id=intent.event_id,
            )
            if truncation_metadata:  # pragma: no cover
                object.__setattr__(observation, "truncation_metadata", truncation_metadata)
            if zk_receipt:  # pragma: no cover
                object.__setattr__(observation, "zk_proof", zk_receipt)

            logger.info(f"Tool {intent.tool_name} executed successfully.")

            obs_dict = observation.model_dump()
            if hasattr(observation, "truncation_metadata"):  # pragma: no cover
                obs_dict["truncation_metadata"] = observation.truncation_metadata
            if hasattr(observation, "zk_proof") and observation.zk_proof:  # pragma: no cover
                obs_dict["zk_proof"] = observation.zk_proof.model_dump()

            await self.broker.push(obs_dict)

        except asyncio.CancelledError:
            # Task was cancelled and is preemptible
            logger.info(f"Execution for {intent.event_id} was safely eradicated via preemption.")

            sandbox = self.active_sandboxes.get(intent.event_id)
            if sandbox:
                logger.info(f"Invoking explicit asynchronous teardown on sandbox for {intent.event_id}")
                await sandbox.teardown(force=True)
            else:
                logger.warning(f"No active sandbox tracked for {intent.event_id} to teardown")

            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                # The FRD: If True, eradicate safely. It doesn't explicitly specify the
                # Observation payload for eradication, but following conventions,
                # we return a pure state JSON primitive.
                payload={"eradicated": True},
                triggering_invocation_id=intent.event_id,
            )
            await self.broker.push(observation.model_dump())
            raise  # Re-raise CancelledError to properly terminate the task

        except Exception:
            # Fatal Crash
            tb = traceback.format_exc()
            if self.masking_functor:
                tb = self.masking_functor.redact(tb)

            payload = {"traceback": tb}

            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                payload=payload,
                triggering_invocation_id=intent.event_id,
            )
            logger.error(f"Tool {intent.tool_name} crashed: {tb}")
            await self.broker.push(observation.model_dump())
        finally:
            self.active_tasks_count -= 1
            self.active_tasks.pop(intent.event_id, None)
            self.preempted_events.discard(intent.event_id)
            self.active_sandboxes.pop(intent.event_id, None)
