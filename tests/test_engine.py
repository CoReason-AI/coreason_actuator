import asyncio
import uuid
from typing import Any

import pytest
from coreason_manifest.spec.ontology import (
    AgentAttestationReceipt,
    EvictionPolicy,
    ExecutionSLA,
    PermissionBoundaryPolicy,
    SideEffectProfile,
    StateHydrationManifest,
    ToolInvocationEvent,
    ToolManifest,
    ZeroKnowledgeReceipt,
)

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
        # Block forever if no responses are queued, simulating empty broker
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


def create_mock_zk_proof() -> ZeroKnowledgeReceipt:
    return ZeroKnowledgeReceipt.model_construct(
        proof_protocol="zk-SNARK",
        public_inputs_hash="mock",
        verifier_key_id="mock",
        cryptographic_blob="mock",
    )


def create_mock_attestation() -> AgentAttestationReceipt:
    valid_hash = "a" * 64
    return AgentAttestationReceipt(
        training_lineage_hash=valid_hash,
        developer_signature="mock",
        capability_merkle_root=valid_hash,
        credential_presentations=[],
    )


def create_mock_manifest() -> ToolManifest:
    side_effects = SideEffectProfile(mutates_state=False, is_idempotent=False)
    permissions = PermissionBoundaryPolicy(
        network_access=False, allowed_domains=[], file_system_mutation_forbidden=True, auth_requirements=[]
    )
    sla = ExecutionSLA(max_execution_time_ms=1000, max_compute_footprint_mb=128)
    return ToolManifest(
        tool_name="test_tool",
        description="A test tool",
        input_schema={"type": "object"},
        side_effects=side_effects,
        permissions=permissions,
        sla=sla,
    )


@pytest.mark.asyncio
async def test_engine_execute_success() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    event_id = str(uuid.uuid4())
    intent = ToolInvocationEvent(
        event_id=event_id,
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"foo": "bar"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    manifest = create_mock_manifest()

    # Pre-populate the response queue in the broker
    expected_response = {
        "event_id": str(uuid.uuid4()),
        "timestamp": 1704067200.0,
        "type": "observation",
        "payload": {"execution_status": "completed", "result": "success"},
        "triggering_invocation_id": event_id,
    }
    broker.responses.append(expected_response)

    # Execute
    result = await engine.execute(intent, manifest)

    # Assertions
    assert result == expected_response
    assert daemon.is_running is True

    assert len(broker.pushed_messages) == 1
    pushed = broker.pushed_messages[0]
    assert pushed["jsonrpc"] == "2.0"
    assert pushed["id"] == event_id
    assert pushed["method"] == "test_tool"
    assert pushed["params"]["tool_name"] == "test_tool"


@pytest.mark.asyncio
async def test_engine_execute_with_state_hydration() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    event_id = str(uuid.uuid4())
    intent = ToolInvocationEvent(
        event_id=event_id,
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"foo": "bar"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    hydration = StateHydrationManifest.model_validate(
        {
            "epistemic_coordinate": "test_coordinate",
            "crystallized_ledger_cids": [],
            "working_context_variables": {},
            "max_retained_tokens": 1000,
        }
    )

    class DummySessionState:
        def __init__(self) -> None:
            self.session_id = "session_123"
            self.allowed_vault_keys: list[str] = []

        def model_dump(self) -> dict[str, Any]:
            return {"session_id": "session_123", "allowed_vault_keys": []}

    class DummyPartitionState:
        def __init__(self) -> None:
            self.partition_id = "part_123"
            self.execution_runtime = "wasm32-wasi"
            self.allow_network_egress = False

        def model_dump(self) -> dict[str, Any]:
            return {
                "partition_id": "part_123",
                "execution_runtime": "wasm32-wasi",
                "allow_network_egress": False,
            }

    session_state = DummySessionState()
    partition_state = DummyPartitionState()

    object.__setattr__(hydration, "session_state", session_state)
    object.__setattr__(hydration, "partition_state", partition_state)

    # Also patch state_hydration.model_dump to include these dynamic attributes
    original_model_dump = hydration.model_dump

    def patched_model_dump() -> dict[str, Any]:
        d: dict[str, Any] = original_model_dump()
        d["session_state"] = session_state.model_dump()
        d["partition_state"] = partition_state.model_dump()
        return d

    object.__setattr__(hydration, "model_dump", patched_model_dump)

    object.__setattr__(intent, "state_hydration", hydration)

    manifest = create_mock_manifest()

    # Expected response from daemon
    expected_response = {
        "event_id": str(uuid.uuid4()),
        "timestamp": 1704067200.0,
        "type": "observation",
        "payload": {"execution_status": "completed", "result": "success"},
        "triggering_invocation_id": event_id,
    }
    broker.responses.append(expected_response)

    eviction_policy = EvictionPolicy(strategy="fifo", max_retained_tokens=500)

    result = await engine.execute(intent, manifest, eviction_policy)

    assert result == expected_response

    # Verify out-of-band attribute extraction
    pushed = broker.pushed_messages[0]
    assert "state_hydration" in pushed["params"]
    assert pushed["params"]["state_hydration"]["session_state"]["session_id"] == "session_123"
    assert "eviction_policy" in pushed["params"]
    assert pushed["params"]["eviction_policy"]["strategy"] == "fifo"


@pytest.mark.asyncio
async def test_engine_execute_ignores_other_messages() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    event_id = str(uuid.uuid4())
    intent = ToolInvocationEvent(
        event_id=event_id,
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"foo": "bar"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    manifest = create_mock_manifest()

    # Queue up a different message first, then the correct one
    other_message = {"triggering_invocation_id": "other_id", "payload": {}}
    expected_response = {
        "event_id": str(uuid.uuid4()),
        "timestamp": 1704067200.0,
        "type": "observation",
        "payload": {"execution_status": "completed", "result": "success"},
        "triggering_invocation_id": event_id,
    }
    broker.responses.append(other_message)
    broker.responses.append(expected_response)

    result = await engine.execute(intent, manifest)

    assert result == expected_response
    # Verify the other message was pushed back to the queue
    assert other_message in broker.pushed_messages


@pytest.mark.asyncio
async def test_engine_execute_handles_jsonrpc_error() -> None:
    broker = MockIPCBroker()
    daemon = MockActuatorDaemon()
    engine = ActuatorEngine(broker=broker, daemon=daemon)  # type: ignore

    event_id = str(uuid.uuid4())
    intent = ToolInvocationEvent(
        event_id=event_id,
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"foo": "bar"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    manifest = create_mock_manifest()

    expected_response = {
        "jsonrpc": "2.0",
        "id": event_id,
        "error": {
            "code": -32601,
            "message": "Method not found",
        },
    }
    broker.responses.append(expected_response)

    result = await engine.execute(intent, manifest)
    assert result == expected_response
