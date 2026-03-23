import json
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from coreason_manifest.spec.ontology import (
    EphemeralNamespacePartitionState,
    ObservationEvent,
    SecureSubSessionState,
    StateHydrationManifest,
    ToolInvocationEvent,
    ToolManifest,
)

from coreason_actuator.engine import ActuatorEngine
from coreason_actuator.sandbox import SandboxProviderFactory, SymbolicSandboxProvider


@pytest.mark.asyncio
async def test_symbolic_sandbox_provider() -> None:
    provider = SymbolicSandboxProvider()

    # Test provision
    part = EphemeralNamespacePartitionState.model_construct(
        partition_id="z3-part",
        execution_runtime=cast("Any", "z3-solver"),
        authorized_bytecode_hashes=[],
        max_ttl_seconds=10,
        max_vram_mb=512,
    )
    provider.provision(part)
    assert "--unshare-net" in provider.bwrap_cmd_array

    # Test network and immutability (no-ops or simple lists)
    provider.apply_network_egress_rules(["test"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108
    assert "--ro-bind" in provider.bwrap_cmd_array

    provider.inject_secrets({"sec": "val"})
    assert "--tmpfs" in provider.bwrap_cmd_array

    # Test execute SAT
    payload_sat = {
        "formal_grammar_payload": "x = Int('x')\ny = Int('y')\nsolve(x > 2, y < 10, x + 2*y == 7)",
        "expected_proof_schema": {"x": "integer", "y": "integer"},
        "timeout_ms": 1000,
    }
    bytecode_sat = json.dumps(payload_sat).encode("utf-8")
    result_sat = await provider.execute(bytecode_sat)
    assert result_sat["x"] == 7
    assert result_sat["y"] == 0

    # Test execute UNSAT
    payload_unsat = {
        "formal_grammar_payload": "x = Int('x')\nsolve(x > 2, x < 1)",
        "expected_proof_schema": {"x": "integer"},
        "timeout_ms": 1000,
    }
    bytecode_unsat = json.dumps(payload_unsat).encode("utf-8")
    with pytest.raises(RuntimeError, match="UNSAT"):
        await provider.execute(bytecode_unsat)

    # Test JSON decode error
    with pytest.raises(RuntimeError, match="Invalid payload"):
        await provider.execute(b"not json")

    await provider.teardown()


def test_sandbox_factory_z3() -> None:
    part = EphemeralNamespacePartitionState.model_construct(
        partition_id="z3-part",
        execution_runtime=cast("Any", "z3-solver"),
        authorized_bytecode_hashes=[],
        max_ttl_seconds=10,
        max_vram_mb=512,
    )
    provider = SandboxProviderFactory.create(part)
    assert isinstance(provider, SymbolicSandboxProvider)


@pytest.mark.asyncio
async def test_engine_zk_receipt() -> None:
    broker = AsyncMock()
    daemon = AsyncMock()

    engine = ActuatorEngine(broker=broker, daemon=daemon)

    # Setup mock push/pull
    from coreason_manifest.spec.ontology import (
        AgentAttestationReceipt,
        PermissionBoundaryPolicy,
        SideEffectProfile,
        ZeroKnowledgeReceipt,
    )

    intent = ToolInvocationEvent(
        event_id="test-zk-event",
        timestamp=0.0,
        tool_name="test_tool",
        parameters={},
        zk_proof=ZeroKnowledgeReceipt.model_construct(
            proof_protocol="zk-STARK", public_inputs_hash="mock", verifier_key_id="mock", cryptographic_blob="{}"
        ),
        agent_attestation=AgentAttestationReceipt.model_construct(
            training_lineage_hash="mock", developer_signature="mock", capability_merkle_root="mock"
        ),
    )
    # Add hydration for RISC-V ZKVM
    hydration = cast(
        "Any",
        StateHydrationManifest.model_construct(
            epistemic_coordinate="a", crystallized_ledger_cids=[], working_context_variables={}, max_retained_tokens=100
        ),
    )
    object.__setattr__(
        hydration,
        "session_state",
        SecureSubSessionState.model_construct(
            session_id="test-session", allowed_vault_keys=[], max_ttl_seconds=10, description=""
        ),
    )
    object.__setattr__(
        hydration,
        "ephemeral_partition",
        EphemeralNamespacePartitionState.model_construct(
            partition_id="zkvm-part",
            execution_runtime="riscv32-zkvm",
            authorized_bytecode_hashes=[],
            max_ttl_seconds=10,
            max_vram_mb=512,
        ),
    )
    object.__setattr__(intent, "state_hydration", hydration)

    manifest = ToolManifest.model_construct(
        tool_name="test_tool",
        description="mock",
        input_schema={},
        side_effects=SideEffectProfile.model_construct(
            mutates_state=False,
            is_idempotent=True,
        ),
        permissions=PermissionBoundaryPolicy.model_construct(
            network_access=False,
            file_system_mutation_forbidden=True,
        ),
    )

    observation = ObservationEvent.model_construct(
        event_id="obs-id",
        timestamp=0.0,
        type="observation",
        triggering_invocation_id="test-zk-event",
        payload={"result": ["actual", {"stdout": "blob"}]},
    )

    # Mock pull to return observation WITH zk_proof (simulating daemon.py)
    obs_dict = observation.model_dump()
    obs_dict["payload"] = {"result": "actual"}
    obs_dict["zk_proof"] = {
        "proof_protocol": "zk-STARK",
        "public_inputs_hash": "mock",
        "verifier_key_id": "mock",
        "cryptographic_blob": {"stdout": "blob"},
    }
    cast("AsyncMock", broker.pull).return_value = obs_dict

    # Mock engine method directly
    cast("AsyncMock", engine.broker.pull).return_value = obs_dict

    intent_dict = intent.model_dump()
    intent_dict["state_hydration"] = hydration.model_dump()

    response = await engine.execute(intent_dict, manifest.model_dump())

    assert "zk_proof" in response
    assert response["zk_proof"]["cryptographic_blob"] == {"stdout": "blob"}
    assert response["payload"]["result"] == "actual"
