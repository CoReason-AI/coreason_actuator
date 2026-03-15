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
from typing import Any

from coreason_manifest.spec.ontology import (
    BackpressurePolicy,
    JSONRPCErrorResponseState,
    JSONRPCErrorState,
    ToolInvocationEvent,
)

from coreason_actuator.ingress import IPCValidator
from coreason_actuator.interfaces import IPCBrokerProtocol
from coreason_actuator.utils.logger import logger


class ActuatorDaemon:
    """The continuous worker daemon subscribing to an IPC message broker."""

    def __init__(
        self,
        broker: IPCBrokerProtocol,
        validator: IPCValidator,
        backpressure_policy: BackpressurePolicy,
    ) -> None:
        self.broker = broker
        self.validator = validator
        self.backpressure_policy = backpressure_policy
        self.active_tasks_count = 0
        self._is_running = False

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

        # Simulated Execution Hook (In future units, this will route to The Do-Operator)
        await self._dispatch_intent(result, raw_payload.get("id"))

    async def _dispatch_intent(self, intent: ToolInvocationEvent, request_id: Any) -> None:
        """
        Temporarily holds place for the execution routing logic.
        For now, we just acknowledge receipt and update metrics.
        """
        self.active_tasks_count += 1
        logger.info(f"Dispatched tool invocation: {intent.tool_name}", request_id=request_id)
        # We decrement it here for now to avoid the pool filling up during tests.
        # In reality, this decrement occurs when the tool finishes executing.
        self.active_tasks_count -= 1
