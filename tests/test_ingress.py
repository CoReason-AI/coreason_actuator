# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

from typing import Any

from coreason_manifest.spec.ontology import (
    JSONRPCErrorResponseState,
    PermissionBoundaryPolicy,
    SideEffectProfile,
    TamperFaultEvent,
    ToolInvocationEvent,
    ToolManifest,
)

from coreason_actuator.ingress import IPCValidator


class MockRegistry:
    def __init__(self, tools: dict[str, ToolManifest]) -> None:
        self.tools = tools

    def get_tool(self, tool_name: str) -> dict[str, Any] | None:
        tool = self.tools.get(tool_name)
        return tool.model_dump() if tool else None


class MockVerifier:
    def __init__(self, should_pass: bool = True) -> None:
        self.should_pass = should_pass

    def verify(self, intent: dict[str, Any]) -> bool:
        _ = intent
        if not self.should_pass:
            raise TamperFaultEvent("Mock verification failed")
        return True


def get_valid_raw_payload() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "execute_tool",
        "id": 123,
        "params": {
            "event_id": "test_event_id",
            "timestamp": 1234567890.0,
            "type": "tool_invocation",
            "tool_name": "test_tool",
            "parameters": {"arg": "value"},
            "agent_attestation": {
                "training_lineage_hash": "a" * 64,
                "developer_signature": "sig",
                "capability_merkle_root": "b" * 64,
                "credential_presentations": [],
            },
            "zk_proof": {
                "proof_protocol": "zk-SNARK",
                "public_inputs_hash": "a" * 64,
                "verifier_key_id": "cid",
                "cryptographic_blob": "blob",
            },
            "state_hydration": {
                "epistemic_coordinate": "test_coordinate",
                "crystallized_ledger_cids": ["a" * 64, "b" * 64],
                "working_context_variables": {"context": "test_context"},
                "max_retained_tokens": 1000,
            },
        },
    }


def get_mock_tool() -> ToolManifest:
    return ToolManifest(
        tool_name="test_tool",
        description="A test tool",
        input_schema={"type": "object"},
        side_effects=SideEffectProfile(is_idempotent=True, mutates_state=False),
        permissions=PermissionBoundaryPolicy(network_access=False, file_system_mutation_forbidden=True),
    )


def test_validator_success() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()
    result = validator.validate_intent(payload)

    assert isinstance(result, ToolInvocationEvent)
    assert result.tool_name == "test_tool"
    assert result.parameters == {"arg": "value"}
    assert hasattr(result, "state_hydration")
    assert result.state_hydration.epistemic_coordinate == "test_coordinate"


def test_validator_missing_state_hydration() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()
    if "params" in payload and isinstance(payload["params"], dict):
        payload["params"].pop("state_hydration", None)

    result = validator.validate_intent(payload)

    assert isinstance(result, JSONRPCErrorResponseState)
    assert result.error.code == -32602


def test_validator_missing_tool_name_attribute() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()

    # The easiest way to hit the missing `tool_name` code path
    # without failing validation is to patch the constructed object.
    import unittest.mock

    with unittest.mock.patch("coreason_manifest.spec.ontology.ToolInvocationEvent.model_validate") as mock_validate:
        # Mock it so it returns an object without a `tool_name` attribute
        class BadToolInvocationEvent:
            def model_dump(self) -> dict[str, Any]:
                return {}

        mock_validate.return_value = BadToolInvocationEvent()
        result = validator.validate_intent(payload)

        assert isinstance(result, JSONRPCErrorResponseState)
        assert result.error.code == -32602
        assert "Missing tool_name in ToolInvocationEvent" in result.error.message


def test_validator_invalid_state_hydration() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()
    if "params" in payload and isinstance(payload["params"], dict):
        # Provide an invalid state_hydration (e.g. missing required epistemic_coordinate)
        payload["params"]["state_hydration"] = {"crystallized_ledger_cids": ["a" * 64]}

    result = validator.validate_intent(payload)

    assert isinstance(result, JSONRPCErrorResponseState)
    assert result.error.code == -32602


def test_validator_invalid_rpc_intent() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    # Missing method
    payload = {"jsonrpc": "2.0", "id": 123, "params": {}}
    result = validator.validate_intent(payload)

    assert isinstance(result, JSONRPCErrorResponseState)
    assert result.error.code == -32700


def test_validator_invalid_tool_invocation() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()
    # Replace valid tool payload with a bad type to test Pydantic validation failure
    if "params" in payload and isinstance(payload["params"], dict):
        payload["params"] = {"tool_name": "test_tool", "parameters": "INVALID_NOT_DICT"}

    result = validator.validate_intent(payload)

    assert isinstance(result, JSONRPCErrorResponseState)
    assert result.error.code == -32602


def test_validator_cryptographic_failure() -> None:
    registry = MockRegistry({"test_tool": get_mock_tool()})
    verifier = MockVerifier(should_pass=False)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()
    result = validator.validate_intent(payload)

    assert isinstance(result, JSONRPCErrorResponseState)
    assert result.error.code == 403


def test_validator_tool_not_found() -> None:
    registry = MockRegistry({})  # Empty registry
    verifier = MockVerifier(should_pass=True)
    validator = IPCValidator(registry, verifier)

    payload = get_valid_raw_payload()
    result = validator.validate_intent(payload)

    assert isinstance(result, JSONRPCErrorResponseState)
    assert result.error.code == -32601
