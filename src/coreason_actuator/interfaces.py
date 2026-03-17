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

from coreason_manifest.spec.ontology import EvictionPolicy, ToolInvocationEvent, ToolManifest


class ActuatorEngineProtocol(Protocol):
    """Protocol defining the interface for the primary execution engine class."""

    async def execute(
        self, intent: ToolInvocationEvent, manifest: ToolManifest, eviction_policy: EvictionPolicy | None = None
    ) -> dict[str, Any]:
        """Awaits an authorized tool execution and returns a raw, verifiable JSON result."""
        ...


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
    """Protocol abstracting the headless browser kinematic operations using strictly atomic locators."""

    async def click(self, x: float, y: float, expected_visual_concept: str, timeout: int = 100) -> Any:
        """
        Executes a physical click at the given X/Y coordinates ONLY if the semantic anchor
        (expected_visual_concept) is mathematically verified in the same operation.
        Eliminates TOCTOU race conditions.
        """
        ...

    async def type_text(self, x: float, y: float, text: str, expected_visual_concept: str, timeout: int = 100) -> Any:
        """
        Executes a physical text typing at the given X/Y coordinates ONLY if the semantic anchor
        (expected_visual_concept) is mathematically verified in the same operation.
        Eliminates TOCTOU race conditions.
        """
        ...

    async def get_current_url(self) -> str:
        """Returns the current URL of the active browser session."""
        ...

    async def get_viewport_size(self) -> tuple[int, int]:
        """Returns the current viewport dimensions (width, height)."""
        ...

    async def get_dom_hash(self) -> str:
        """Returns the SHA-256 hash of the active DOM tree."""
        ...

    async def capture_viewport_screenshot(self) -> bytes:
        """Captures a rasterized viewport image buffer."""
        ...
