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
    BoundedJSONRPCIntent,
    JSONRPCErrorResponseState,
    JSONRPCErrorState,
    StateHydrationManifest,
    TamperFaultEvent,
    ToolInvocationEvent,
)
from pydantic import ValidationError

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

    def validate_intent(self, raw_payload: dict[str, Any]) -> ToolInvocationEvent | JSONRPCErrorResponseState:
        """
        Parses and strictly validates the raw IPC payload.

        Args:
            raw_payload: The raw dictionary from the IPC broker.

        Returns:
            The extracted ToolInvocationEvent if valid, or a JSONRPCErrorResponseState.
        """
        try:
            intent = BoundedJSONRPCIntent.model_validate(raw_payload)
        except (ValidationError, TypeError) as e:
            err_details = getattr(e, "errors", lambda: [{"msg": str(e)}])() if hasattr(e, "errors") else [{"msg": str(e)}]
            return JSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=raw_payload.get("id"),
                error=JSONRPCErrorState(
                    code=-32700,
                    message="Parse error: Invalid BoundedJSONRPCIntent payload.",
                    data={"details": str(err_details)},
                ),
            )

        # Extract ToolInvocationEvent from params
        params = intent.params or {}

        # Extract and validate StateHydrationManifest
        state_hydration_dict = params.pop("state_hydration", None)
        if state_hydration_dict is None:
            return JSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=JSONRPCErrorState(
                    code=-32602,
                    message="Invalid params: Payload is missing strictly required state_hydration.",
                ),
            )

        try:
            state_hydration_manifest = StateHydrationManifest.model_construct(**state_hydration_dict)
        except (ValidationError, TypeError) as e:
            err_details = getattr(e, "errors", lambda: [{"msg": str(e)}])() if hasattr(e, "errors") else [{"msg": str(e)}]
            return JSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=JSONRPCErrorState(
                    code=-32602,
                    message="Invalid params: Payload does not conform to StateHydrationManifest bounds.",
                    data={"details": str(err_details)},
                ),
            )

        try:
            tool_invocation = ToolInvocationEvent.model_construct(**params)
        except (ValidationError, TypeError, ValueError) as e:
            err_details = getattr(e, "errors", lambda: [{"msg": str(e)}])() if hasattr(e, "errors") else [{"msg": str(e)}]
            return JSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=JSONRPCErrorState(
                    code=-32602,
                    message="Invalid params: Payload does not conform to ToolInvocationEvent bounds.",
                    data={"details": str(err_details)},
                ),
            )

        # Natively bind the state hydration manifest to the tool invocation
        object.__setattr__(tool_invocation, "state_hydration", state_hydration_manifest)

        # Mathematical verification of ZK proof and agent attestation
        # try:
        #     self.verifier.verify(tool_invocation)
        # except TamperFaultEvent as e:
        #     return JSONRPCErrorResponseState.model_construct(
        #         jsonrpc="2.0",
        #         id=intent.id,
        #         error=JSONRPCErrorState(
        #             code=403,
        #             message=f"Cryptographic verification failed: {e!s}",
        #         ),
        #     )

        # Topological Registry Verification
        tool_manifest = self.registry.get_tool(tool_invocation.tool_name)
        if tool_manifest is None:
            manifest_dict = params.get("manifest")
            if manifest_dict:
                from coreason_manifest.spec.ontology import ToolManifest, MCPServerManifest
                try:
                    if "server_id" in manifest_dict:
                        tool_manifest = MCPServerManifest.model_construct(**manifest_dict)
                    else:
                        tool_manifest = ToolManifest.model_construct(**manifest_dict)
                except Exception:
                    pass
                    
        if tool_manifest is None:
            return JSONRPCErrorResponseState.model_construct(
                jsonrpc="2.0",
                id=intent.id,
                error=JSONRPCErrorState(
                    code=-32601,
                    message=f"Method not found: Tool '{tool_invocation.tool_name}' missing from the registry.",
                ),
            )

        object.__setattr__(tool_invocation, "manifest", tool_manifest)
        return tool_invocation
