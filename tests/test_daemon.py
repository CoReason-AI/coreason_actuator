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
from coreason_manifest.spec.ontology import BackpressurePolicy, ToolInvocationEvent, ToolManifest

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
        # Just return a proper mock event
        return ToolInvocationEvent(
            event_id="test_event_123",
            timestamp=12345.6,
            tool_name="test_tool",
            parameters={},
            zk_proof={
                "proof_protocol": "zk-SNARK",
                "public_inputs_hash": "a" * 64,
                "verifier_key_id": "b",
                "cryptographic_blob": "c",
                "latent_state_commitments": {},
            },
            agent_attestation={
                "training_lineage_hash": "a" * 64,
                "developer_signature": "b",
                "capability_merkle_root": "c" * 64,
                "credential_presentations": [],
            },
        )


class MockRegistry:
    def __init__(self, manifests: dict[str, ToolManifest] | None = None) -> None:
        self.manifests = manifests or {}

    def get_tool(self, tool_name: str) -> ToolManifest | None:
        return self.manifests.get(tool_name)


class MockExecutionStrategy:
    def __init__(self, should_crash: bool = False, result: Any = "success_result") -> None:
        self.should_crash = should_crash
        self.result = result

    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        _ = (intent, manifest, sandbox_pid)
        if self.should_crash:
            raise RuntimeError("Mock execution crash")
        # Add a sleep so we have time to cancel in preemption tests
        await asyncio.sleep(0.01)
        return self.result


def create_mock_manifest(tool_name: str = "test_tool", is_preemptible: bool = False) -> ToolManifest:
    from coreason_manifest.spec.ontology import PermissionBoundaryPolicy, SideEffectProfile

    return ToolManifest(
        tool_name=tool_name,
        description="A mock tool",
        input_schema={"type": "object"},
        side_effects=SideEffectProfile(is_idempotent=True, mutates_state=False),
        permissions=PermissionBoundaryPolicy(network_access=False, file_system_mutation_forbidden=True),
        is_preemptible=is_preemptible,
    )


@pytest.mark.asyncio
async def test_daemon_successful_dispatch() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy()
    registry = MockRegistry({"test_tool": create_mock_manifest()})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    await daemon.run_once()

    # Wait for background task to finish
    await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    assert broker.pushed[0]["type"] == "observation"
    assert broker.pushed[0]["payload"]["execution_status"] == "completed"
    assert broker.pushed[0]["payload"]["result"] == "success_result"


@pytest.mark.asyncio
async def test_daemon_backpressure_shedding() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    # Set max concurrent to 1, but seed active tasks to 1 to force shedding
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=1)
    strategy = MockExecutionStrategy()
    registry = MockRegistry()

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore
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
    strategy = MockExecutionStrategy()
    registry = MockRegistry()

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    await daemon.run_once()

    assert len(broker.pushed) == 1
    assert broker.pushed[0]["error"]["code"] == 400
    assert "Mock validation failure" in broker.pushed[0]["error"]["message"]


@pytest.mark.asyncio
async def test_daemon_start_stop() -> None:
    broker = MockBroker([])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy()
    registry = MockRegistry()

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    # Run start as a background task
    task = asyncio.create_task(daemon.start())

    # Yield control to allow start to run
    await asyncio.sleep(0.01)

    assert daemon._is_running

    daemon.stop()

    await task

    assert not daemon._is_running


@pytest.mark.asyncio
async def test_daemon_execution_success() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy(result={"some": "data"})
    registry = MockRegistry({"test_tool": create_mock_manifest()})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    await daemon.run_once()

    await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "completed"
    assert pushed["payload"]["result"] == {"some": "data"}
    assert pushed["triggering_invocation_id"] == "test_event_123"


@pytest.mark.asyncio
async def test_daemon_execution_crash() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy(should_crash=True)
    registry = MockRegistry({"test_tool": create_mock_manifest()})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    await daemon.run_once()

    await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "fatal_crash"
    assert "Mock execution crash" in pushed["payload"]["traceback"]
    assert pushed["triggering_invocation_id"] == "test_event_123"


@pytest.mark.asyncio
async def test_daemon_missing_manifest() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy()
    # Missing from registry
    registry = MockRegistry({})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    await daemon.run_once()
    await asyncio.sleep(0.01)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    # In reality, IPCValidator should reject it first, but if it reaches _dispatch_intent without a manifest,
    # it yields a JSONRPCErrorResponseState indicating Method not found.
    assert pushed["jsonrpc"] == "2.0"
    assert pushed["id"] == "1"
    # This responds immediately
    assert pushed["error"]["code"] == -32601
    assert "Method not found: Tool 'test_tool' missing from the registry." in pushed["error"]["message"]


@pytest.mark.asyncio
async def test_daemon_preemption_preemptible_task() -> None:
    broker = MockBroker(
        [
            {"jsonrpc": "2.0", "method": "test", "id": "1"},  # Intent
            {"type": "barge_in", "target_event_id": "test_event_123"},  # Preemption Signal
        ]
    )
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy(result={"some": "data"})
    # Create manifest with is_preemptible=True
    registry = MockRegistry({"test_tool": create_mock_manifest(is_preemptible=True)})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    # Run once to pull the intent
    await daemon.run_once()
    await asyncio.sleep(0.001)
    await asyncio.sleep(0.001)
    assert daemon.active_tasks_count == 1

    # Run again to pull the preemption signal
    await daemon.run_once()

    # Wait for background task to resolve its cancellation
    await asyncio.sleep(0.05)

    # It should have eradicated it
    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "preempted"
    assert pushed["payload"]["eradicated"] is True


@pytest.mark.asyncio
async def test_daemon_preemption_no_active_task() -> None:
    broker = MockBroker(
        [
            {"type": "barge_in", "target_event_id": "test_event_123"}  # Preemption Signal
        ]
    )
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy(result={"some": "data"})
    registry = MockRegistry({})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    # Run once to pull the preemption signal, task is not active
    await daemon.run_once()
    assert daemon.active_tasks_count == 0


@pytest.mark.asyncio
async def test_daemon_preemption_non_preemptible_task() -> None:
    broker = MockBroker(
        [
            {"jsonrpc": "2.0", "method": "test", "id": "1"},  # Intent
            {"type": "barge_in", "target_event_id": "test_event_123"},  # Preemption Signal
        ]
    )
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy(result={"some": "data"})
    # Create manifest with is_preemptible=False
    registry = MockRegistry({"test_tool": create_mock_manifest(is_preemptible=False)})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    # Run once to pull the intent
    await daemon.run_once()
    await asyncio.sleep(0.001)
    assert daemon.active_tasks_count == 1

    # Run again to pull the preemption signal
    await daemon.run_once()

    # Wait for background task to finish executing its logic since it shielded itself
    await asyncio.sleep(0.05)

    # It should have allowed it to complete
    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "completed_under_preemption"
    assert pushed["payload"]["null_hash"] is True
