# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import pytest

from coreason_actuator.main import DummyIPCBroker, DummyRegistry, DummyVerifier, app, run


def test_app_initialization() -> None:
    assert app is not None
    assert app.registered_commands
    # For typer command decorators without explicit name, name can be None or the function's name.
    # The callback is what we want to verify.
    cb = app.registered_commands[0].callback
    assert cb is not None
    assert cb.__name__ == "run"


@pytest.mark.asyncio
async def test_dummy_broker() -> None:
    broker = DummyIPCBroker()
    await broker.queue.put({"test": "data"})
    res = await broker.pull()
    assert res == {"test": "data"}

    # Should not raise
    await broker.push({"status": "ok"})


def test_dummy_registry() -> None:
    reg = DummyRegistry()
    assert reg.get_tool("anything") is None


def test_dummy_verifier() -> None:
    from unittest.mock import MagicMock

    verifier = DummyVerifier()
    assert verifier.verify(MagicMock()) is True


def test_dummy_lock_manager() -> None:
    from coreason_actuator.main import DummyLockManager

    lock_manager = DummyLockManager()
    assert lock_manager.acquire_lock("anything") is True
    # release_lock does nothing, just check it does not raise
    lock_manager.release_lock("anything")


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
            mock_loop.run_until_complete.assert_called_once()

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

            mock_loop.run_until_complete.assert_called_once()
