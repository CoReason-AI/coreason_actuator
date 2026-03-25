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

from pydantic import ValidationError

from coreason_actuator.adapters.dto import (
    InternalIntentDTO,
    InternalJSONRPCErrorResponseState,
    InternalJSONRPCErrorState,
    InternalStateHydrationDTO,
)
from coreason_actuator.interfaces import ActionSpaceRegistryProtocol, CryptographicVerifierProtocol


class IPCValidator:
    """Service for IPC Ingress & Pre-Flight Validation."""

    def __init__(
        self,
        registry: ActionSpaceRegistryProtocol,
        verifier: CryptographicVerifierProtocol,
    ) -> None:
        self.registry = registry
        self.verifier = verifier

    def validate_intent(self, raw_payload: dict[str, Any]) -> dict[str, Any] | InternalJSONRPCErrorResponseState:
        """
        Parses and validates the raw IPC payload.

        Args:
            raw_payload: The raw dictionary from the IPC broker.

        Returns:
            The extracted payload dictionary if valid, or a InternalJSONRPCErrorResponseState.
        """
        try:
            intent = InternalIntentDTO.model_validate(raw_payload)
        except (ValidationError, TypeError) as e:
            err_details = e.errors() if hasattr(e, "errors") else [{"msg": str(e)}]
            return InternalJSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=raw_payload.get("id"),
                error=InternalJSONRPCErrorState(
                    code=-32700,
                    message="Parse error: Invalid BoundedJSONRPCIntent payload.",
                    data={"details": str(err_details)},
                ),
            )

        # Extract params
        params = intent.params.copy() if intent.params else {}

        # The protocol method expects BoundedJSONRPCIntent to map method to tool_name
        tool_name = intent.method or params.get("tool_name")
        if not tool_name:
            return InternalJSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=InternalJSONRPCErrorState(
                    code=-32602,
                    message="Invalid params: Missing tool_name in intent.",
                ),
            )

        # Ensure tool_name is in params for downstream
        params["tool_name"] = tool_name

        # Extract and validate StateHydrationManifest
        state_hydration_dict = params.get("state_hydration", None)
        if state_hydration_dict is None:
            return InternalJSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=InternalJSONRPCErrorState(
                    code=-32602,
                    message="Invalid params: Payload is missing strictly required state_hydration.",
                ),
            )

        try:
            state_hydration_manifest = InternalStateHydrationDTO.model_validate(state_hydration_dict)
            params["state_hydration"] = state_hydration_manifest
        except (ValidationError, TypeError) as e:
            err_details = e.errors() if hasattr(e, "errors") else [{"msg": str(e)}]
            return InternalJSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=InternalJSONRPCErrorState(
                    code=-32602,
                    message="Invalid params: Payload does not conform to StateHydrationManifest bounds.",
                    data={"details": str(err_details)},
                ),
            )

        # Mathematical verification of ZK proof and agent attestation
        try:
            self.verifier.verify(params)
        except Exception as e:
            # We trap broad Exception since we removed coreason_manifest TamperFaultEvent dependency
            return InternalJSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=InternalJSONRPCErrorState(
                    code=403,
                    message=f"Cryptographic verification failed: {e!s}",
                ),
            )

        # Topological Registry Verification
        tool_manifest = self.registry.get_tool(tool_name)
        if tool_manifest is None:
            return InternalJSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=InternalJSONRPCErrorState(
                    code=-32601,
                    message=f"Method not found: Tool '{tool_name}' missing from the registry.",
                ),
            )

        return params
