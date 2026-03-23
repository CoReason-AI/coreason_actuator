import asyncio
from typing import Any, cast

import pytest

from coreason_actuator.engine import ActuatorEngine


class MockIPCBroker:
    def __init__(self) -> None:
        self.pushed_messages: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    async def push(self, message: dict[str, Any]) -> None:
        self.pushed_messages.append(message)

    async def pull(self) -> dict[str, Any]:
        if self.responses:
            return self.responses.pop(0)
        await asyncio.sleep(0.1)
        return {}


class MockActuatorDaemon:
    def __init__(self) -> None:
        self.started = False
        self._running_task: asyncio.Task[Any] | None = None

    @property
    def is_running(self) -> bool:
        return self._running_task is not None

    def register_task(self, task: asyncio.Task[Any]) -> None:
        self._running_task = task

    async def start(self) -> None:
        self.started = True
        while True:
            await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_engine_execute_research_intent_success() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    intent_dict = {
        "target_buffer_id": "research_id_123",
        "max_schema_depth": 5,
        "max_properties": 100,
    }

    partitions = [
        {
            "partition_id": "part_1",
            "execution_runtime": "wasm32-wasi",
            "allow_network_egress": False,
            "authorized_bytecode_hashes": [],
            "max_ttl_seconds": 100,
            "max_vram_mb": 512,
        }
    ]

    expected_response = {
        "event_id": "obs_1",
        "timestamp": 123.0,
        "type": "observation",
        "payload": {"research_result": "found something"},
        "triggering_invocation_id": "research_id_123",
    }
    broker.responses.append(expected_response)

    result = await engine.execute_research_intent(intent_dict, partitions)

    assert result == expected_response
    assert daemon.is_running is True

    assert len(broker.pushed_messages) == 1
    pushed = broker.pushed_messages[0]
    assert pushed["jsonrpc"] == "2.0"
    assert pushed["id"] == "research_id_123"
    assert pushed["method"] == "__LATENT_RESEARCH__"
    assert pushed["params"]["target_buffer_id"] == "research_id_123"
    assert pushed["params"]["partitions"] == partitions


@pytest.mark.asyncio
async def test_engine_execute_research_intent_handles_error() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    intent_dict = {
        "target_buffer_id": "research_id_123",
        "max_schema_depth": 5,
        "max_properties": 100,
    }

    expected_response = {
        "jsonrpc": "2.0",
        "id": "research_id_123",
        "error": {
            "code": -32000,
            "message": "research failed",
        },
    }
    broker.responses.append(expected_response)

    result = await engine.execute_research_intent(intent_dict)
    assert result == expected_response


@pytest.mark.asyncio
async def test_engine_execute_ignores_other_messages_execute() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    intent_dict = {
        "event_id": "test_id_1",
        "tool_name": "test_tool",
        "parameters": {},
        "manifest": {"tool_name": "test_tool", "description": "mock", "is_preemptible": False},
    }

    # Queue up a different message first, then the correct one
    other_message = {"triggering_invocation_id": "other_id", "payload": {}}
    expected_response = {
        "triggering_invocation_id": "test_id_1",
        "payload": {"result": "success"},
    }
    broker.responses.append(other_message)
    broker.responses.append(expected_response)

    result = await engine.execute(intent_dict, cast("dict[str, Any]", intent_dict["manifest"]))

    assert result == expected_response
    # Verify the other message was pushed back to the queue
    assert other_message in broker.pushed_messages


@pytest.mark.asyncio
async def test_engine_execute_research_intent_ignores_other_messages() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    intent_dict = {
        "target_buffer_id": "research_id_123",
        "max_schema_depth": 5,
        "max_properties": 100,
    }

    # Queue up a different message first, then the correct one
    other_message = {"triggering_invocation_id": "other_id", "payload": {}}
    expected_response = {
        "triggering_invocation_id": "research_id_123",
        "payload": {"result": "success"},
    }
    broker.responses.append(other_message)
    broker.responses.append(expected_response)

    result = await engine.execute_research_intent(intent_dict)

    assert result == expected_response
    # Verify the other message was pushed back to the queue
    assert other_message in broker.pushed_messages
