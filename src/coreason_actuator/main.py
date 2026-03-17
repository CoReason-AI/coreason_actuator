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
import signal
from typing import Any

import typer
from coreason_manifest.spec.ontology import BackpressurePolicy, ToolInvocationEvent, ToolManifest

from coreason_actuator.daemon import ActuatorDaemon
from coreason_actuator.ingress import IPCValidator
from coreason_actuator.interfaces import (
    ActionSpaceRegistryProtocol,
    CryptographicVerifierProtocol,
    IPCBrokerProtocol,
)
from coreason_actuator.strategies import NativeExecutionStrategy
from coreason_actuator.utils.logger import logger

app = typer.Typer()


class DummyVerifier(CryptographicVerifierProtocol):
    """A dummy verifier for standalone CLI bootstrapping."""

    def verify(self, intent: ToolInvocationEvent) -> bool:
        _ = intent
        return True


class DummyIPCBroker(IPCBrokerProtocol):
    """A dummy IPC broker for standalone CLI bootstrapping."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def pull(self) -> dict[str, Any]:
        return await self.queue.get()

    async def push(self, message: dict[str, Any]) -> None:
        logger.info(f"Pushed to broker: {message}")


class DummyRegistry(ActionSpaceRegistryProtocol):
    """A dummy registry for standalone CLI bootstrapping."""

    def get_tool(self, tool_name: str) -> ToolManifest | None:
        _ = tool_name
        return None


class DummyLockManager:
    """A dummy lock manager for standalone CLI bootstrapping."""

    def acquire_lock(self, resource_id: str, timeout: float = 10.0) -> bool:
        _ = resource_id, timeout
        return True

    def release_lock(self, resource_id: str) -> None:
        _ = resource_id


@app.command()  # type: ignore
def run() -> None:
    """Bootstraps the ActuatorDaemon and starts the main loop."""
    logger.info("Bootstrapping ActuatorDaemon...")

    registry = DummyRegistry()
    verifier = DummyVerifier()
    broker = DummyIPCBroker()
    validator = IPCValidator(registry=registry, verifier=verifier)
    policy = BackpressurePolicy(max_queue_depth=100, max_concurrent_tool_invocations=10)
    lock_manager = DummyLockManager()
    strategy = NativeExecutionStrategy(registry=registry, lock_manager=lock_manager)  # type: ignore

    daemon = ActuatorDaemon(
        broker=broker,
        validator=validator,
        backpressure_policy=policy,
        execution_strategy=strategy,
        registry=registry,
    )

    loop = asyncio.get_event_loop()

    def handle_sigterm() -> None:
        logger.info("Received SIGTERM/SIGINT. Initiating graceful shutdown...")
        daemon.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_sigterm)

    try:
        loop.run_until_complete(daemon.start())
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("ActuatorDaemon terminated.")


if __name__ == "__main__":
    app()  # pragma: no cover
