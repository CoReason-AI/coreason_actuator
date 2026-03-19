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

import pytest
from coreason_manifest.spec.ontology import PermissionBoundaryPolicy, SideEffectProfile, ToolManifest

from coreason_actuator.ipc import IPCBrokerServer
from coreason_actuator.main import ActionSpaceRegistry, AsyncLockManager, app, run


def test_app_initialization() -> None:
    assert app is not None
    assert app.registered_commands
    # For typer command decorators without explicit name, name can be None or the function's name.
    # The callback is what we want to verify.
    cb = app.registered_commands[0].callback
    assert cb is not None
    assert cb.__name__ == "run"


@pytest.mark.asyncio
async def test_ipc_broker_server() -> None:
    broker = IPCBrokerServer("tcp://0.0.0.0:5555")
    await broker.start()
    await broker.queue.put({"test": "data"})
    res = await broker.pull()
    assert res == {"test": "data"}

    # Should not raise
    await broker.push({"status": "ok"})
    await broker.close()

    # When not running, push/pull should raise RuntimeError
    with pytest.raises(RuntimeError):
        await broker.pull()
    with pytest.raises(RuntimeError):
        await broker.push({})


def test_action_space_registry() -> None:
    reg = ActionSpaceRegistry()
    assert reg.get_tool("anything") is None

    from coreason_manifest.spec.ontology import ActionSpaceManifest

    mock_tool = ToolManifest.model_construct(
        tool_name="test_tool",
        description="test",
        input_schema={},
        side_effects=SideEffectProfile.model_construct(is_idempotent=True, mutates_state=False),
        permissions=PermissionBoundaryPolicy.model_construct(network_access=False, file_system_mutation_forbidden=True),
    )
    manifest = ActionSpaceManifest.model_construct(
        action_space_id="test",
        native_tools=[mock_tool],
        mcp_servers=[],
        ephemeral_partitions=[],
        kinetic_separation=None,
    )
    reg_with_tools = ActionSpaceRegistry(manifest)
    assert reg_with_tools.get_tool("test_tool") == mock_tool


@pytest.mark.asyncio
async def test_async_lock_manager() -> None:
    lock_manager = AsyncLockManager()

    # Test successful acquire
    async with lock_manager.acquire("test_lock"):
        assert "test_lock" in lock_manager._locks

    # Lock is cleaned up
    assert "test_lock" not in lock_manager._locks

    # Test TTL timeout
    async def hold_lock() -> None:
        async with lock_manager.acquire("test_lock2"):
            await asyncio.sleep(0.1)

    task = asyncio.create_task(hold_lock())
    # give it a moment to acquire
    await asyncio.sleep(0.01)

    with pytest.raises(TimeoutError, match="Failed to acquire lock for test_lock2 within TTL 10ms"):
        async with lock_manager.acquire("test_lock2", ttl=10):
            pass

    await task


@pytest.mark.asyncio
async def test_run_command_graceful_shutdown() -> None:
    import asyncio
    from typing import Any
    from unittest.mock import patch

    with patch("coreason_actuator.main.ActuatorDaemon") as mock_daemon:
        mock_daemon_instance = mock_daemon.return_value
        # Use an asyncio.Future to mock start
        f: asyncio.Future[Any] = asyncio.Future()
        f.set_result(None)
        mock_daemon_instance.start.return_value = f

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = mock_get_loop.return_value

            run()

            # Verify daemon was initialized
            mock_daemon.assert_called_once()
            # Verify the loop started it
            assert mock_loop.run_until_complete.call_count == 2

            # Verify signals were registered
            assert mock_loop.add_signal_handler.call_count == 2

            # Simulate a signal handler being called
            sig_handler = mock_loop.add_signal_handler.call_args_list[0][0][1]
            sig_handler()

            mock_daemon_instance.stop.assert_called_once()


@pytest.mark.asyncio
async def test_run_command_cancelled_error() -> None:
    import asyncio
    from typing import Any
    from unittest.mock import patch

    with patch("coreason_actuator.main.ActuatorDaemon") as mock_daemon:
        mock_daemon_instance = mock_daemon.return_value
        f: asyncio.Future[Any] = asyncio.Future()
        f.set_result(None)
        mock_daemon_instance.start.return_value = f

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = mock_get_loop.return_value
            mock_loop.run_until_complete.side_effect = asyncio.CancelledError()

            # Should silently catch asyncio.CancelledError
            run()

            assert mock_loop.run_until_complete.call_count >= 1
