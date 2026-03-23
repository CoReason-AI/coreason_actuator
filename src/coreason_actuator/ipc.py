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
import contextlib
import json
from typing import Any
from urllib.parse import urlparse

from coreason_manifest.spec.ontology import (
    EphemeralNamespacePartitionState,
    EvictionPolicy,
    LatentSchemaInferenceIntent,
    ToolInvocationEvent,
    ToolManifest,
)

from coreason_actuator.interfaces import IPCBrokerProtocol
from coreason_actuator.utils.logger import logger


class IPCBrokerServer(IPCBrokerProtocol):
    """
    A real IPC broker server that listens on a given URI.
    Uses asyncio stream readers/writers to accept and handle JSON-RPC messages.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.StreamWriter] = set()

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:  # pragma: no cover
        """Handles an incoming IPC client connection."""
        self._clients.add(writer)  # pragma: no cover
        peer = writer.get_extra_info("peername")  # pragma: no cover
        logger.info(f"Client connected: {peer}")  # pragma: no cover

        try:  # pragma: no cover
            while True:  # pragma: no cover
                data = await reader.readline()  # pragma: no cover
                if not data:  # pragma: no cover
                    break  # pragma: no cover
                try:  # pragma: no cover
                    payload = json.loads(data.decode())  # pragma: no cover
                    await self.queue.put(payload)  # pragma: no cover
                except json.JSONDecodeError:  # pragma: no cover
                    logger.warning("Received invalid JSON from client")  # pragma: no cover
        except asyncio.CancelledError:  # pragma: no cover
            pass  # pragma: no cover
        finally:  # pragma: no cover
            self._clients.discard(writer)  # pragma: no cover
            writer.close()  # pragma: no cover
            with contextlib.suppress(ConnectionError):  # pragma: no cover
                await writer.wait_closed()  # pragma: no cover
            logger.info(f"Client disconnected: {peer}")  # pragma: no cover

    async def start(self) -> None:
        """Starts listening on the specified URI."""
        parsed = urlparse(self.uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5555

        if parsed.scheme == "tcp":
            self._server = await asyncio.start_server(self.handle_client, host, port)
            logger.info(f"IPCBrokerServer listening on {host}:{port}")
        else:
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")  # pragma: no cover

    async def close(self) -> None:
        """Closes the socket cleanly."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for writer in self._clients:  # pragma: no cover
            writer.close()  # pragma: no cover
            try:  # pragma: no cover
                await writer.wait_closed()  # pragma: no cover
            except Exception as e:  # pragma: no cover
                logger.debug(f"Error while closing client writer: {e}")  # pragma: no cover
        self._clients.clear()
        logger.info(f"IPCBrokerServer closed connection on URI: {self.uri}")

    async def pull(self) -> dict[str, Any]:
        """Pulls a message from the incoming queue."""
        if not self._server:
            raise RuntimeError("IPCBrokerServer is not running.")
        return await self.queue.get()

    async def push(self, message: dict[str, Any]) -> None:
        """Pushes a message back (e.g., a response) to all connected clients."""
        if not self._server:
            raise RuntimeError("IPCBrokerServer is not running.")

        payload = json.dumps(message) + "\n"
        data = payload.encode()
        for writer in self._clients:  # pragma: no cover
            try:  # pragma: no cover
                writer.write(data)  # pragma: no cover
                await writer.drain()  # pragma: no cover
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to push message to client: {e}")  # pragma: no cover
        logger.info(f"IPCBrokerServer pushed to {len(self._clients)} clients")


class RemoteKineticBrokerClient:
    """
    A network client acting as the ActuatorEngineProtocol.
    Connects to the IPCBrokerServer to proxy execution payloads to a remote physical daemon.
    """

    def __init__(self, broker_uri: str | None) -> None:
        self.broker_uri = broker_uri or "tcp://127.0.0.1:5555"

    async def execute(
        self,
        intent: ToolInvocationEvent,
        manifest: ToolManifest,
        eviction_policy: EvictionPolicy | None = None,
        partitions: list[EphemeralNamespacePartitionState] | None = None,
    ) -> dict[str, Any]:
        """Proxies the payload over TCP and strictly awaits the JSON-RPC response."""
        parsed = urlparse(self.broker_uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5555

        reader, writer = await asyncio.open_connection(host, port)
        try:
            payload = intent.model_dump()
            payload["manifest"] = manifest.model_dump()
            if eviction_policy is not None:
                payload["eviction_policy"] = eviction_policy.model_dump()

            if partitions is not None:
                payload["partitions"] = [p.model_dump() for p in partitions]

            if hasattr(intent, "state_hydration") and intent.state_hydration is not None:
                payload["state_hydration"] = intent.state_hydration.model_dump()

            packet = {
                "jsonrpc": "2.0",
                "id": intent.event_id,
                "method": intent.tool_name,
                "params": payload,
            }

            data = json.dumps(packet) + "\n"
            writer.write(data.encode())
            await writer.drain()

            while True:
                response_data = await reader.readline()
                if not response_data:
                    raise ConnectionError("Remote IPC Broker closed the connection unexpectedly.")

                response = json.loads(response_data.decode())
                if response.get("triggering_invocation_id") == intent.event_id or response.get("id") == intent.event_id:
                    return response  # type: ignore[no-any-return]
        finally:
            writer.close()
            import contextlib

            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def execute_research_intent(
        self, intent: LatentSchemaInferenceIntent, partitions: list[EphemeralNamespacePartitionState] | None = None
    ) -> dict[str, Any]:
        """Proxies a high-order research intent over TCP."""
        parsed = urlparse(self.broker_uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5555

        reader, writer = await asyncio.open_connection(host, port)
        try:
            payload = intent.model_dump()
            if partitions is not None:
                payload["partitions"] = [p.model_dump() for p in partitions]

            intent_id = getattr(intent, "event_id", getattr(intent, "target_buffer_id", "unknown"))
            packet = {
                "jsonrpc": "2.0",
                "id": intent_id,
                "method": "__LATENT_RESEARCH__",
                "params": payload,
            }

            data = json.dumps(packet) + "\n"
            writer.write(data.encode())
            await writer.drain()

            while True:
                response_data = await reader.readline()
                if not response_data:
                    raise ConnectionError("Remote IPC Broker closed the connection unexpectedly.")

                response = json.loads(response_data.decode())
                if response.get("triggering_invocation_id") == intent_id or response.get("id") == intent_id:
                    return response  # type: ignore[no-any-return]
        finally:
            writer.close()
            import contextlib

            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
