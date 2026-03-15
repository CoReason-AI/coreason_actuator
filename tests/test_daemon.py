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

import pytest
from coreason_manifest.spec.ontology import BackpressurePolicy

from coreason_actuator.daemon import ActuatorDaemon


class MockBroker:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.pushed: list[dict[str, Any]] = []

    async def pull(self) -> dict[str, Any]:
        if not self.payloads:
            # Raise an exception to simulate broker error/empty loop end, allowing test to stop gracefully
            raise RuntimeError("Broker empty")
        return self.payloads.pop(0)

    async def push(self, message: dict[str, Any]) -> None:
        self.pushed.append(message)


class MockValidator:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        from coreason_manifest.spec.ontology import JSONRPCErrorResponseState, JSONRPCErrorState

        self.error_response = JSONRPCErrorResponseState(
            jsonrpc="2.0",
            id="123",
            error=JSONRPCErrorState(code=400, message="Mock validation failure"),
        )

    def validate_intent(self, payload: dict[str, Any]) -> Any:
        _ = payload

        if self.should_fail:
            return self.error_response

        # Just return a dummy event
        # Note: In real life we'd mock cleanly, but we cheat here as _dispatch_intent only checks tool_name
        class DummyIntent:
            tool_name = "test_tool"

        return DummyIntent()


@pytest.mark.asyncio
async def test_daemon_successful_dispatch() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    daemon = ActuatorDaemon(broker, validator, policy)  # type: ignore

    await daemon.run_once()

    assert len(broker.pushed) == 0
    # active tasks count remains 0 because dispatch mock increments then decrements immediately


@pytest.mark.asyncio
async def test_daemon_backpressure_shedding() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    # Set max concurrent to 1, but seed active tasks to 1 to force shedding
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=1)

    daemon = ActuatorDaemon(broker, validator, policy)  # type: ignore
    daemon.active_tasks_count = 1

    await daemon.run_once()

    assert len(broker.pushed) == 1
    assert broker.pushed[0]["error"]["code"] == -32000
    assert "Too Many Requests" in broker.pushed[0]["error"]["message"]


@pytest.mark.asyncio
async def test_daemon_validation_failure() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=True)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    daemon = ActuatorDaemon(broker, validator, policy)  # type: ignore

    await daemon.run_once()

    assert len(broker.pushed) == 1
    assert broker.pushed[0]["error"]["code"] == 400
    assert "Mock validation failure" in broker.pushed[0]["error"]["message"]


@pytest.mark.asyncio
async def test_daemon_start_stop() -> None:
    broker = MockBroker([])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    daemon = ActuatorDaemon(broker, validator, policy)  # type: ignore

    # Run start as a background task
    task = asyncio.create_task(daemon.start())

    # Yield control to allow start to run
    await asyncio.sleep(0.01)

    assert daemon._is_running

    daemon.stop()

    await task

    assert not daemon._is_running
