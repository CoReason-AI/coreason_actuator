import asyncio
import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from coreason_actuator.ipc import IPCBrokerServer, RemoteKineticBrokerClient

@pytest.mark.asyncio
async def test_remote_broker_client_execute() -> None:
    client = RemoteKineticBrokerClient("tcp://127.0.0.1:5555")

    intent = {
        "event_id": "test_id",
        "tool_name": "test_tool",
        "parameters": {},
        "state_hydration": {"some_state": True}
    }
    manifest = {"tool_name": "test_tool"}
    eviction_policy = {"strategy": "fifo"}
    partitions = [{"partition_id": "part1"}]

    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_reader.readline.side_effect = [
        json.dumps({"id": "test_id", "result": "success"}).encode() + b"\n"
    ]

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        result = await client.execute(intent, manifest, eviction_policy, partitions)

    assert result["id"] == "test_id"
    assert result["result"] == "success"

@pytest.mark.asyncio
async def test_remote_broker_client_research_intent() -> None:
    client = RemoteKineticBrokerClient("tcp://127.0.0.1:5555")

    intent = {
        "event_id": "test_id",
        "target_buffer_id": "target1"
    }
    partitions = [{"partition_id": "part1"}]

    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_reader.readline.side_effect = [
        json.dumps({"id": "test_id", "result": "success"}).encode() + b"\n"
    ]

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        result = await client.execute_research_intent(intent, partitions)

    assert result["id"] == "test_id"
    assert result["result"] == "success"

@pytest.mark.asyncio
async def test_remote_broker_client_execute_connection_closed() -> None:
    client = RemoteKineticBrokerClient("tcp://127.0.0.1:5555")

    intent = {
        "event_id": "test_id",
        "tool_name": "test_tool",
    }
    manifest = {"tool_name": "test_tool"}

    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_reader.readline.return_value = b""  # Connection closed

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        with pytest.raises(ConnectionError, match="Remote IPC Broker closed the connection unexpectedly."):
            await client.execute(intent, manifest)

@pytest.mark.asyncio
async def test_remote_broker_client_research_intent_connection_closed() -> None:
    client = RemoteKineticBrokerClient("tcp://127.0.0.1:5555")

    intent = {
        "event_id": "test_id",
    }

    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_reader.readline.return_value = b""  # Connection closed

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        with pytest.raises(ConnectionError, match="Remote IPC Broker closed the connection unexpectedly."):
            await client.execute_research_intent(intent)
