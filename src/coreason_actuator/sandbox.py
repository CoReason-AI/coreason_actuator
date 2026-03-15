# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import collections
import hashlib
from typing import Any, Protocol

from coreason_manifest.spec.ontology import (
    EphemeralNamespacePartitionState,
    SecureSubSessionState,
    ToolManifest,
)

from coreason_actuator.utils.logger import logger


class EnterpriseVaultProtocol(Protocol):
    """Protocol for the Enterprise Secret Manager."""

    def unseal(self, auth_requirements: list[str]) -> dict[str, str]:
        """Dynamically parses auth_requirements and retrieves secrets."""
        ...


class SandboxProviderProtocol(Protocol):
    """Abstract implementations MUST strictly map to the coreason-manifest bounds."""

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        """Dynamically provisions the execution boundary."""
        ...

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        """Injects unsealed secrets via in-memory tmpfs mounts."""
        ...

    def execute(self, bytecode: bytes) -> Any:
        """Executes the target binary within the physical boundary."""
        ...

    async def teardown(self, force: bool = False) -> None:
        """Eradicates the partition safely and frees host VRAM."""
        ...


class WasmSandboxProvider:
    """WASM sandboxing execution provider (wasm32-wasi)."""

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        logger.info(f"Provisioning WASM sandbox: {partition_state.partition_id}")

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        logger.info(f"Injecting {len(secrets)} secrets into WASM tmpfs")

    def execute(self, bytecode: bytes) -> Any:
        logger.info(f"Executing WASM bytecode ({len(bytecode)} bytes)")
        return "WASM execution simulated"

    async def teardown(self, force: bool = False) -> None:
        logger.info(f"Tearing down WASM sandbox (force={force})")


class RiscvZkvmSandboxProvider:
    """RISC-V ZKVM execution provider for generating hardware ZK proofs."""

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        logger.info(f"Provisioning RISC-V ZKVM sandbox: {partition_state.partition_id}")

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        logger.info(f"Injecting {len(secrets)} secrets into RISC-V ZKVM tmpfs")

    def execute(self, bytecode: bytes) -> Any:
        logger.info(f"Executing RISC-V bytecode ({len(bytecode)} bytes)")
        return "RISC-V execution simulated"

    async def teardown(self, force: bool = False) -> None:
        logger.info(f"Tearing down RISC-V ZKVM sandbox (force={force})")


class BpfSandboxProvider:
    """BPF execution provider for kernel-level execution."""

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        logger.info(f"Provisioning BPF sandbox: {partition_state.partition_id}")

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        logger.info(f"Injecting {len(secrets)} secrets into BPF tmpfs")

    def execute(self, bytecode: bytes) -> Any:
        logger.info(f"Executing BPF bytecode ({len(bytecode)} bytes)")
        return "BPF execution simulated"

    async def teardown(self, force: bool = False) -> None:
        logger.info(f"Tearing down BPF sandbox (force={force})")


class SandboxProviderFactory:
    """Factory to route provisioning to the correct hypervisor adapter."""

    @staticmethod
    def create(partition_state: EphemeralNamespacePartitionState) -> SandboxProviderProtocol:
        runtime = partition_state.execution_runtime
        if runtime == "wasm32-wasi":
            return WasmSandboxProvider()
        if runtime == "riscv32-zkvm":
            return RiscvZkvmSandboxProvider()
        if runtime == "bpf":
            return BpfSandboxProvider()
        # Mathematical impossibility per schema bounds, but included for completeness
        raise ValueError(f"Legacy containerization strictly prohibited: {runtime}")


class StatefulSandboxCache:
    """Maintains an internal LRU cache of active sandboxes bound to specific session_id tags."""

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._cache: collections.OrderedDict[str, SandboxProviderProtocol] = collections.OrderedDict()

    def get_or_create(
        self,
        session_state: SecureSubSessionState,
        partition_state: EphemeralNamespacePartitionState,
    ) -> SandboxProviderProtocol:
        session_id = session_state.session_id

        if session_id in self._cache:
            logger.info(f"Warm start: Reusing sandbox for session {session_id}")
            # Move to end to indicate recent use
            self._cache.move_to_end(session_id)
            return self._cache[session_id]

        logger.info(f"Cold start: Provisioning new sandbox for session {session_id}")
        provider = SandboxProviderFactory.create(partition_state)
        provider.provision(partition_state)

        if len(self._cache) >= self.max_size:
            # Evict least recently used
            evicted_id, evicted_provider = self._cache.popitem(last=False)
            logger.info(f"Cache full: Evicting sandbox for session {evicted_id}")
            # Note: Cache eviction is synchronous here, but teardown is async.
            # In a real environment, this might need an async task or an async cache.
            # For the interface change, we'll log a warning or handle it properly later.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(evicted_provider.teardown(force=True))
                # Store the task to avoid RUF006 and prevent GC of the task
                if not hasattr(self, "_bg_tasks"):
                    self._bg_tasks = set()
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
            except RuntimeError:
                pass  # no running loop

        self._cache[session_id] = provider
        return provider

    async def teardown_all(self) -> None:
        """Safely tears down all cached sandboxes."""
        for session_id, provider in self._cache.items():
            logger.info(f"Tearing down cached sandbox for session {session_id}")
            await provider.teardown(force=True)
        self._cache.clear()


def verify_bytecode_safety(bytecode: bytes, authorized_hashes: list[str]) -> bool:
    """
    Computes the SHA-256 hash of the target binary and asserts it matches
    a hash within the authorized_bytecode_hashes whitelist.
    """
    bytecode_hash = hashlib.sha256(bytecode).hexdigest()
    is_safe = bytecode_hash in authorized_hashes
    if not is_safe:
        logger.warning(
            f"Bytecode validation failed. Hash {bytecode_hash} not in authorized list.",
            computed_hash=bytecode_hash,
            authorized=authorized_hashes,
        )
    return is_safe


def verify_network_access(manifest: ToolManifest, partition_state: EphemeralNamespacePartitionState) -> bool:
    """
    Evaluates network egress using a defense-in-depth lock.
    To authorize a network socket, BOTH ToolManifest.permissions.network_access
    AND the sandbox's EphemeralNamespacePartitionState.allow_network_egress must evaluate to True.
    A conflict mathematically blocks the execution by raising a PermissionError.
    """
    tool_network = manifest.permissions.network_access
    sandbox_network = partition_state.allow_network_egress

    if tool_network and not sandbox_network:
        raise PermissionError(
            "Dual-Evaluation Permission Boundary conflict: "
            "Tool requests network access, but the active ephemeral partition explicitly denies egress."
        )

    return bool(tool_network and sandbox_network)
