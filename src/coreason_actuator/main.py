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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import typer
from coreason_manifest.spec.ontology import ActionSpaceManifest, BackpressurePolicy, ToolManifest

from coreason_actuator.daemon import ActuatorDaemon
from coreason_actuator.ingress import IPCValidator
from coreason_actuator.interfaces import (
    ActionSpaceRegistryProtocol,
)
from coreason_actuator.ipc import IPCBrokerServer
from coreason_actuator.security import CryptographicVerifier
from coreason_actuator.strategies import DistributedLockProtocol, NativeExecutionStrategy
from coreason_actuator.utils.logger import logger

app = typer.Typer()


class ActionSpaceRegistry(ActionSpaceRegistryProtocol):
    """A real registry that holds a collection of tools from an ActionSpaceManifest."""

    def __init__(self, manifest: ActionSpaceManifest | None = None) -> None:
        self.tools: dict[str, ToolManifest] = {}
        self._callables: dict[str, Callable[..., Awaitable[Any]]] = {}
        if manifest:
            for tool in getattr(manifest, "native_tools", []):
                name = getattr(tool, "tool_name", getattr(tool, "name", "unknown"))
                self.tools[name] = tool

    def get_tool(self, tool_name: str) -> ToolManifest | None:
        """Retrieves a tool from the registry if it exists."""
        return self.tools.get(tool_name)

    async def get_callable(self, tool_name: str) -> Callable[..., Awaitable[Any]] | None:
        """Retrieves the registered asynchronous callable for the tool."""
        return self._callables.get(tool_name)  # pragma: no cover


class AsyncLockManager(DistributedLockProtocol):
    """A real async lock manager using asyncio.Lock."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def acquire(self, lock_key: str, ttl: int | None = None) -> AsyncIterator[Any]:
        """Acquires a lock for the given key."""
        if lock_key not in self._locks:
            self._locks[lock_key] = asyncio.Lock()
        lock = self._locks[lock_key]

        if ttl is not None:
            # Wrap the lock acquisition with a timeout matching the requested TTL.
            try:
                await asyncio.wait_for(lock.acquire(), timeout=ttl / 1000.0)
            except TimeoutError as e:
                raise TimeoutError(f"Failed to acquire lock for {lock_key} within TTL {ttl}ms") from e
        else:
            await lock.acquire()

        try:
            yield lock
        finally:
            lock.release()
            # Optionally clean up unused locks
            if not lock.locked():
                self._locks.pop(lock_key, None)


@app.command()
def run() -> None:
    """Bootstraps the ActuatorDaemon and starts the main loop."""
    logger.info("Bootstrapping ActuatorDaemon...")

    # Instantiate real components
    registry = ActionSpaceRegistry()
    verifier = CryptographicVerifier()

    # Typically, URI would come from SystemConfigurationManifest
    broker = IPCBrokerServer(uri="tcp://0.0.0.0:5555")

    validator = IPCValidator(registry=registry, verifier=verifier)
    policy = BackpressurePolicy(max_queue_depth=100, max_concurrent_tool_invocations=10)
    lock_manager = AsyncLockManager()

    strategy = NativeExecutionStrategy(registry=registry, lock_manager=lock_manager)

    daemon = ActuatorDaemon(
        broker=broker,
        validator=validator,
        backpressure_policy=policy,
        execution_strategy=strategy,
        registry=registry,
    )

    loop = asyncio.get_event_loop()

    _bg_tasks: set[asyncio.Task[Any]] = set()

    def handle_sigterm() -> None:
        logger.info("Received SIGTERM/SIGINT. Initiating graceful shutdown...")
        # Add graceful socket closing to the shutdown handler via task
        if hasattr(broker, "close"):
            task = loop.create_task(broker.close())
            _bg_tasks.add(task)
            task.add_done_callback(_bg_tasks.discard)
        daemon.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_sigterm)

    try:
        # We need to start the broker before the daemon
        loop.run_until_complete(broker.start())
        loop.run_until_complete(daemon.start())
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("ActuatorDaemon terminated.")


if __name__ == "__main__":
    app()  # pragma: no cover
