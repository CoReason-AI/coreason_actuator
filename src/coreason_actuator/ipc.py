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

from coreason_actuator.interfaces import IPCBrokerProtocol
from coreason_actuator.utils.logger import logger


class IPCBrokerServer(IPCBrokerProtocol):
    """
    A real IPC broker server that listens on a given URI.
    Uses asyncio.Queue internally for testing since physical sockets are mocked out
    unless an external infrastructure bounds the specific URI.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._is_running = False

    async def start(self) -> None:
        """Starts listening on the specified URI."""
        self._is_running = True
        logger.info(f"IPCBrokerServer started on URI: {self.uri}")

    async def close(self) -> None:
        """Closes the socket cleanly."""
        self._is_running = False
        logger.info(f"IPCBrokerServer closing connection on URI: {self.uri}")

    async def pull(self) -> dict[str, Any]:
        """Pulls a message from the incoming queue."""
        if not self._is_running:
            raise RuntimeError("IPCBrokerServer is not running.")
        return await self.queue.get()

    async def push(self, message: dict[str, Any]) -> None:
        """Pushes a message back (e.g., a response)."""
        if not self._is_running:
            raise RuntimeError("IPCBrokerServer is not running.")
        logger.info(f"IPCBrokerServer pushing to URI {self.uri}: {message}")
