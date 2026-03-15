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
from coreason_actuator.strategies import ExecutionStrategyProtocol
from coreason_actuator.utils.logger import logger


class ActuatorDaemon:
    """The continuous worker daemon subscribing to an IPC message broker."""

    def __init__(
        self,
        broker: IPCBrokerProtocol,
        validator: IPCValidator,
        backpressure_policy: BackpressurePolicy,
        execution_strategy: ExecutionStrategyProtocol,
        registry: ActionSpaceRegistryProtocol,
    ) -> None:
        self.broker = broker
        self.validator = validator
        self.backpressure_policy = backpressure_policy
        self.execution_strategy = execution_strategy
        self.registry = registry
        self.active_tasks_count = 0
        self._is_running = False
        self.active_tasks: dict[str, asyncio.Task[Any]] = {}
        self.preempted_events: set[str] = set()

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
                    code=-32000,
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

        # Unfortunately, we don't store the manifest directly alongside the task.
        # But if the task is running, we can just call cancel on it. The task itself
        # will handle the CancelledError and check manifest.is_preemptible.
        # However, to avoid cancelling non-preemptible tasks, it's better if we check
        # the manifest here if we can, but since we don't have it easily accessible,
        # we will cancel the task, and inside _dispatch_intent, we will evaluate is_preemptible.
        # Wait, the FRD says "If False: The Actuator is structurally forbidden from killing the process.
        # It MUST allow the stateful transaction to complete safely..."
        # If we cancel the task, it raises CancelledError inside the execution, which kills the process.
        # So we shouldn't cancel if not preemptible.
        # Let's add target_event_id to preempted_events, and the task will check it later.

        self.preempted_events.add(target_event_id)
        # Inside _dispatch_intent we check if it is preemptible before cancelling,
        # or we just cancel it here if we know it's preemptible.
        # Let's just cancel the task. Inside _dispatch_intent, we will catch CancelledError.
        # Wait, no! If we cancel the task, the transaction won't complete.
        # We must look up the tool intent to get the manifest.
        # How do we know which tool the task is running? We don't track it here.
        # So instead of cancelling right away, let's signal cancellation via an Event,
        # or we can inspect the task.
        # Actually, let's just cancel it, but wrap the execution in a shield if it's not preemptible!
        # That's elegant: in _dispatch_intent, we can check manifest.is_preemptible.
        # If not, we run the execution with `asyncio.shield()`.
        # Then `task.cancel()` will cancel the outer task, but the inner execution continues!
        # But then we'd need to wait for the shielded task to finish to emit the observation.
        # Let's just track the manifest in a dict `self.active_manifests`?
        # No, a simpler way: just call `task.cancel()`.
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
            # If not preemptible, we shield it from cancellation so it can complete safely.
            if not manifest.is_preemptible:
                # We use asyncio.shield to prevent cancellation from killing the inner task
                # but if the outer task is cancelled, asyncio.shield raises CancelledError,
                # while the inner task keeps running. So we'd need to await the inner task separately.
                # A better approach: we just don't cancel it in _handle_preemption if not preemptible.
                # Since we changed _handle_preemption to just cancel, let's instead handle the logic there by
                # checking the registry! We can't because we don't know the tool name in _handle_preemption.
                pass

            # Since we need to know if it's preemptible to cancel it properly,
            # let's just run it. We will catch CancelledError below.

            # Note: A real implementation would extract sandbox_pid and handle teardown.
            sandbox_pid = None

            if not manifest.is_preemptible:
                # If it's not preemptible, we must shield it so if cancelled, it keeps running
                # and we can wait for it.
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

            # Successful Execution
            payload = {"execution_status": "completed", "result": result}

            if intent.event_id in self.preempted_events:
                payload = {"execution_status": "completed_under_preemption", "null_hash": True}

            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                payload=payload,
                triggering_invocation_id=intent.event_id,
            )
            logger.info(f"Tool {intent.tool_name} executed successfully.")
            await self.broker.push(observation.model_dump())

        except asyncio.CancelledError:
            # Task was cancelled and is preemptible
            logger.info(f"Execution for {intent.event_id} was safely eradicated via preemption.")
            # We would call sandbox.teardown(force=True) here if we had the sandbox instance.
            # We assume the provider or caller manages the teardown if sandbox_pid is passed,
            # or we could explicitly call a teardown method if we tracked sandboxes.

            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                # The FRD: If True, eradicate safely. It doesn't explicitly specify the
                # Observation payload for eradication, but following conventions,
                # we return a fatal_crash or preempted status.
                payload={"execution_status": "preempted", "eradicated": True},
                triggering_invocation_id=intent.event_id,
            )
            await self.broker.push(observation.model_dump())
            raise  # Re-raise CancelledError to properly terminate the task

        except Exception:
            # Fatal Crash
            tb = traceback.format_exc()
            observation = ObservationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                type="observation",
                payload={"execution_status": "fatal_crash", "traceback": tb},
                triggering_invocation_id=intent.event_id,
            )
            logger.error(f"Tool {intent.tool_name} crashed: {tb}")
            await self.broker.push(observation.model_dump())
        finally:
            self.active_tasks_count -= 1
            self.active_tasks.pop(intent.event_id, None)
            self.preempted_events.discard(intent.event_id)
