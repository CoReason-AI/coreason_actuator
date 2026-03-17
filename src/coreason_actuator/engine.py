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

from coreason_manifest.spec.ontology import EvictionPolicy, ToolInvocationEvent, ToolManifest

from coreason_actuator.daemon import ActuatorDaemon
from coreason_actuator.interfaces import IPCBrokerProtocol
from coreason_actuator.utils.logger import logger


class ActuatorEngine:
    """
    The primary execution engine class for the Kinetic Plane.

    Serves as the entry point for the Orchestrator, dispatching execution intents
    to the background ActuatorDaemon and waiting for verifiable JSON results.
    """

    def __init__(self, broker: IPCBrokerProtocol, daemon: ActuatorDaemon) -> None:
        self.broker = broker
        self.daemon = daemon
        self._bg_tasks: set[asyncio.Task[Any]] = set()

    async def _start_daemon_if_needed(self) -> None:
        """Starts the background ActuatorDaemon if it isn't running."""
        # Check if the daemon has a running task or loop
        if getattr(self.daemon, "_main_task", None) is None:
            logger.info("Starting ActuatorDaemon in background from ActuatorEngine")
            task = asyncio.create_task(self.daemon.start())
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            self.daemon._main_task = task  # type: ignore

    async def execute(
        self, intent: ToolInvocationEvent, manifest: ToolManifest, eviction_policy: EvictionPolicy | None = None
    ) -> dict[str, Any]:
        """
        Awaits an authorized tool execution and returns a raw, verifiable JSON result.

        Serializes the intent, seamlessly passing the hydrated state manifest to the kinetic worker,
        ensuring the worker applies the exact constraints required by the zero-trust sandbox providers.
        """
        await self._start_daemon_if_needed()

        # Serialize the intent into a raw payload for the IPC Broker
        payload = intent.model_dump()

        # Pass manifest and eviction policy in the payload so the worker can enforce constraints
        payload["manifest"] = manifest.model_dump()
        if eviction_policy is not None:
            payload["eviction_policy"] = eviction_policy.model_dump()

        # Hydrate the dynamic state manifest into the payload out-of-band
        if hasattr(intent, "state_hydration") and intent.state_hydration is not None:
            payload["state_hydration"] = intent.state_hydration.model_dump()

        # Construct JSON-RPC 2.0 packet
        packet = {
            "jsonrpc": "2.0",
            "id": intent.event_id,
            "method": intent.tool_name,
            "params": payload,
        }

        logger.info(f"Engine dispatching intent {intent.event_id} to broker.")
        await self.broker.push(packet)

        # Poll the broker for the corresponding observation
        while True:
            response = await self.broker.pull()

            # Check if it's the target ObservationEvent
            if response.get("triggering_invocation_id") == intent.event_id:
                logger.info(f"Engine received ObservationEvent for intent {intent.event_id}.")
                return response

            # Check if it's a JSONRPCErrorResponseState indicating failure or rejection
            if response.get("id") == intent.event_id and "error" in response:
                logger.error(f"Engine received JSONRPCErrorResponseState for intent {intent.event_id}.")
                return response

            # If the payload belongs to another event, yield and ideally put it back
            # For this simple polling mechanism without specific queues, we push it back.
            await self.broker.push(response)
            await asyncio.sleep(0.01)
