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
from coreason_actuator.security import MaskingFunctor
from coreason_actuator.semantic_extractor import SemanticExtractor


class MockVault:
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets
        self.unsealed_keys: list[str] = []

    def unseal(self, auth_requirements: list[str]) -> dict[str, str]:
        self.unsealed_keys.extend(auth_requirements)
        return self.secrets


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
        from coreason_manifest.spec.ontology import (
            AgentAttestationReceipt,
            StateHydrationManifest,
            ZeroKnowledgeReceipt,
        )

        intent = ToolInvocationEvent(
            event_id="test_event_123",
            timestamp=12345.6,
            tool_name="test_tool",
            parameters={},
            zk_proof=ZeroKnowledgeReceipt.model_validate(
                {
                    "proof_protocol": "zk-SNARK",
                    "public_inputs_hash": "a" * 64,
                    "verifier_key_id": "b",
                    "cryptographic_blob": "c",
                    "latent_state_commitments": {},
                }
            ),
            agent_attestation=AgentAttestationReceipt.model_validate(
                {
                    "training_lineage_hash": "a" * 64,
                    "developer_signature": "b",
                    "capability_merkle_root": "c" * 64,
                    "credential_presentations": [],
                }
            ),
        )

        # Simulate state_hydration bound correctly
        # Since we just found out session_state does not exist on StateHydrationManifest
        # we have to attach it if we want to mimic the logic, or we inject a custom object.
        # Wait, the FRD says "FR-2.2 Stateful Sandbox Cache ... active sandboxes bound to specific session_id tags".
        # Let's mock a simpler object that acts as StateHydrationManifest but has session_state attached
        # OR actually instantiate StateHydrationManifest and set the attribute dynamically just like
        # we do for ToolInvocationEvent!

        state_hydration = StateHydrationManifest.model_validate(
            {
                "epistemic_coordinate": "test_coordinate",
                "crystallized_ledger_cids": [],
                "working_context_variables": {},
                "max_retained_tokens": 1000,
            }
        )

        from coreason_manifest.spec.ontology import SecureSubSessionState

        session_state = SecureSubSessionState.model_validate(
            {
                "session_id": "sess_123",
                "max_ttl_seconds": 300,
                "allowed_vault_keys": ["oauth2:github", "mtls:internal"],
                "description": "Test session",
            }
        )

        # FRD implies session_state is available on state_hydration
        object.__setattr__(state_hydration, "session_state", session_state)

        object.__setattr__(intent, "state_hydration", state_hydration)
        return intent


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
        await asyncio.sleep(0.2)
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
    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
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

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
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

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
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

    for _ in range(20):
        if daemon.active_tasks_count == 1:
            break
        await asyncio.sleep(0.01)
    assert daemon.active_tasks_count == 1

    # Run again to pull the preemption signal
    await daemon.run_once()

    # Wait for background task to resolve its cancellation
    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    # It should have eradicated it
    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "preempted"
    assert pushed["payload"]["eradicated"] is True


@pytest.mark.asyncio
async def test_daemon_preemption_sandbox_teardown() -> None:
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

    # Mock a sandbox for the task
    class MockSandbox:
        def __init__(self) -> None:
            self.teardown_called = False
            self.force = False

        async def teardown(self, force: bool = False) -> None:
            self.teardown_called = True
            self.force = force

    mock_sandbox = MockSandbox()

    # Run once to pull the intent
    await daemon.run_once()

    # Inject the mock sandbox directly into the active tracking dictionary
    daemon.active_sandboxes["test_event_123"] = mock_sandbox  # type: ignore

    for _ in range(20):
        if daemon.active_tasks_count == 1:
            break
        await asyncio.sleep(0.01)

    # Run again to pull the preemption signal
    await daemon.run_once()

    # Wait for background task to resolve its cancellation
    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert mock_sandbox.teardown_called is True
    assert mock_sandbox.force is True


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

    for _ in range(20):
        if daemon.active_tasks_count == 1:
            break
        await asyncio.sleep(0.01)
    assert daemon.active_tasks_count == 1

    # Run again to pull the preemption signal
    await daemon.run_once()

    # Wait for background task to finish executing its logic since it shielded itself
    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    # It should have allowed it to complete
    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "completed_under_preemption"
    assert pushed["payload"]["null_hash"] is True


@pytest.mark.asyncio
async def test_daemon_execution_success_with_scrubbing() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    # Execution strategy returns data with a secret
    strategy = MockExecutionStrategy(
        result={
            "some": "super_secret_token",
            "nested": {"key": "secret_abc"},
            "array": ["super_secret_token", 123, {"hidden": "super_secret_token"}],
            "super_secret_token_key": "val",
            "other": 456,
        }
    )
    registry = MockRegistry({"test_tool": create_mock_manifest()})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    # Set the masking functor
    functor = MaskingFunctor(["super_secret_token", "secret_abc"])
    daemon.set_masking_functor(functor)

    await daemon.run_once()

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "completed"

    result = pushed["payload"]["result"]
    assert result["some"] == MaskingFunctor.REDACTION_STRING
    assert result["nested"]["key"] == MaskingFunctor.REDACTION_STRING
    assert result["array"][0] == MaskingFunctor.REDACTION_STRING
    assert result["array"][1] == 123
    assert result["array"][2]["hidden"] == MaskingFunctor.REDACTION_STRING
    assert result[f"{MaskingFunctor.REDACTION_STRING}_key"] == "val"
    assert result["other"] == 456


@pytest.mark.asyncio
async def test_daemon_execution_crash_with_scrubbing() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    class CrasherStrategy:
        async def execute(self, intent: Any, manifest: Any, sandbox_pid: Any) -> Any:
            _ = (intent, manifest, sandbox_pid)
            raise RuntimeError("Failed because of super_secret_token in memory")

    strategy = CrasherStrategy()
    registry = MockRegistry({"test_tool": create_mock_manifest()})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    functor = MaskingFunctor(["super_secret_token"])
    daemon.set_masking_functor(functor)

    await daemon.run_once()

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]
    assert pushed["type"] == "observation"
    assert pushed["payload"]["execution_status"] == "fatal_crash"

    traceback_str = pushed["payload"]["traceback"]
    assert "super_secret_token" not in traceback_str
    assert MaskingFunctor.REDACTION_STRING in traceback_str


@pytest.mark.asyncio
async def test_daemon_vault_unsealing_and_injection() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)
    strategy = MockExecutionStrategy(result={"some": "data"})
    registry = MockRegistry({"test_tool": create_mock_manifest()})
    vault = MockVault({"oauth2:github": "secret_token"})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry, vault)  # type: ignore

    class MockSandbox:
        def __init__(self) -> None:
            self.injected_secrets: dict[str, str] = {}

        def inject_secrets(self, secrets: dict[str, str]) -> None:
            self.injected_secrets = secrets

    mock_sandbox = MockSandbox()
    daemon.active_sandboxes["test_event_123"] = mock_sandbox  # type: ignore

    await daemon.run_once()

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert sorted(vault.unsealed_keys) == sorted(["oauth2:github", "mtls:internal"])
    assert mock_sandbox.injected_secrets == {"oauth2:github": "secret_token"}


@pytest.mark.asyncio
async def test_daemon_semantic_truncation_routing() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    # Simulate native SemanticExtractor behavior via execution strategy
    truncated_result = {
        "status": "success",
        "data": [1, 2, 3],
        "truncation_metadata": {
            "semantic_truncation_applied": True,
            "items_omitted": 997,
        },
    }

    strategy = MockExecutionStrategy(result=truncated_result)
    registry = MockRegistry({"test_tool": create_mock_manifest()})

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry)  # type: ignore

    await daemon.run_once()

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]

    # Assert it was properly routed to root ObservationEvent
    assert pushed["type"] == "observation"
    assert "truncation_metadata" in pushed
    assert pushed["truncation_metadata"]["semantic_truncation_applied"] is True
    assert pushed["truncation_metadata"]["items_omitted"] == 997

    # Assert it was removed from payload body to avoid schema collisions
    assert "truncation_metadata" not in pushed["payload"]
    assert "truncation_metadata" not in pushed["payload"]["result"]


@pytest.mark.asyncio
async def test_daemon_semantic_extractor_integration() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    # Execution strategy returns data with a very large array
    large_array = list(range(100))
    strategy = MockExecutionStrategy(
        result={
            "large_data": large_array,
            "small_data": [1, 2, 3],
        }
    )
    registry = MockRegistry({"test_tool": create_mock_manifest()})
    semantic_extractor = SemanticExtractor(max_array_length=10)

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry, semantic_extractor=semantic_extractor)  # type: ignore

    await daemon.run_once()

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]

    # Assert it was properly routed to root ObservationEvent
    assert pushed["type"] == "observation"
    assert "truncation_metadata" in pushed
    assert pushed["truncation_metadata"]["semantic_truncation_applied"] is True
    assert pushed["truncation_metadata"]["items_omitted"] == 90

    # Assert payload was truncated
    assert "truncation_metadata" not in pushed["payload"]
    assert "truncation_metadata" not in pushed["payload"]["result"]
    assert len(pushed["payload"]["result"]["large_data"]) == 10
    assert len(pushed["payload"]["result"]["small_data"]) == 3


@pytest.mark.asyncio
async def test_daemon_semantic_extractor_native_combination() -> None:
    broker = MockBroker([{"jsonrpc": "2.0", "method": "test", "id": "1"}])
    validator = MockValidator(should_fail=False)
    policy = BackpressurePolicy(max_queue_depth=10, max_concurrent_tool_invocations=10)

    # Strategy returns some truncation natively
    truncated_result = {
        "status": "success",
        "data": [1, 2, 3],
        "more_data": list(range(100)),
        "truncation_metadata": {
            "semantic_truncation_applied": True,
            "items_omitted": 500,
        },
    }

    strategy = MockExecutionStrategy(result=truncated_result)
    registry = MockRegistry({"test_tool": create_mock_manifest()})
    semantic_extractor = SemanticExtractor(max_array_length=10)

    daemon = ActuatorDaemon(broker, validator, policy, strategy, registry, semantic_extractor=semantic_extractor)  # type: ignore

    await daemon.run_once()

    for _ in range(20):
        if len(broker.pushed) >= 1:
            break
        await asyncio.sleep(0.05)

    assert len(broker.pushed) == 1
    pushed = broker.pushed[0]

    # Assert it combined omitted counts
    assert pushed["type"] == "observation"
    assert "truncation_metadata" in pushed
    assert pushed["truncation_metadata"]["items_omitted"] == 590
    assert len(pushed["payload"]["result"]["more_data"]) == 10
