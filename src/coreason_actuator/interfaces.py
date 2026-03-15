# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

from typing import Any, Protocol

from coreason_manifest.spec.ontology import ToolInvocationEvent, ToolManifest


class IPCBrokerProtocol(Protocol):
    """Protocol defining the interface for IPC message brokers."""

    async def pull(self) -> dict[str, Any]:
        """Pulls a raw JSON-RPC envelope from the broker queue."""
        ...

    async def push(self, message: dict[str, Any]) -> None:
        """Pushes a response or observation back to the broker."""
        ...


class ActionSpaceRegistryProtocol(Protocol):
    """Protocol defining the interface for the mounted ActionSpaceManifest."""

    def get_tool(self, tool_name: str) -> ToolManifest | None:
        """Retrieves a tool from the registry if it exists."""
        ...


class CryptographicVerifierProtocol(Protocol):
    """Protocol defining the interface for cryptographic intent authorization."""

    def verify(self, intent: ToolInvocationEvent) -> bool:
        """Mathematically verifies the zk_proof and agent_attestation."""
        ...


class KinematicBrowserProtocol(Protocol):
    """Protocol abstracting the headless browser kinematic operations."""

    async def click(self, x: float, y: float) -> Any:
        """Executes a physical click at the given X/Y coordinates."""
        ...

    async def type_text(self, x: float, y: float, text: str) -> Any:
        """Executes a physical text typing at the given X/Y coordinates."""
        ...

    async def get_accessibility_tree_hash(self, x: float, y: float) -> str:
        """Returns the accessibility tree hash or semantic representation at the exact coordinate."""
        ...

    async def capture_viewport_screenshot(self) -> bytes:
        """Captures a rasterized viewport image buffer."""
        ...


class AccessibilityTreeProtocol(Protocol):
    """Protocol for semantic visual verification before physical interaction."""

    async def verify_concept(self, x: float, y: float, expected_visual_concept: str) -> bool:
        """
        Functionally verifies the presence of the expected_visual_concept at the target coordinates.
        If the semantic anchor has shifted or is missing, returns False.
        """
        ...
