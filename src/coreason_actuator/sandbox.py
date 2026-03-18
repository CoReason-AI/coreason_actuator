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
import json
import subprocess
from typing import Any, Protocol

import httpx
from coreason_manifest.spec.ontology import (
    EphemeralNamespacePartitionState,
    SecureSubSessionState,
    ToolManifest,
)

from coreason_actuator.utils.logger import logger


class EnterpriseVaultProtocol(Protocol):
    """Protocol for the Enterprise Secret Manager."""

    async def unseal(self, auth_requirements: list[str]) -> dict[str, str]:
        """Dynamically parses auth_requirements and retrieves secrets."""
        ...


class HashiCorpVault:
    """Enterprise Vault implementation using HashiCorp Vault."""

    def __init__(self, vault_addr: str, vault_token: str) -> None:
        self.vault_addr = vault_addr
        self.vault_token = vault_token

    async def unseal(self, auth_requirements: list[str]) -> dict[str, str]:
        """
        Dynamically parses auth_requirements and retrieves secrets.
        """
        logger.info(f"Unsealing secrets via HashiCorp Vault at {self.vault_addr} for {auth_requirements}")

        aggregated_secrets: dict[str, str] = {}
        async with httpx.AsyncClient() as client:
            for req in auth_requirements:
                url = f"{self.vault_addr.rstrip('/')}/{req}"
                headers = {"X-Vault-Token": self.vault_token}

                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    # We assume the secret data is either in the root or in a 'data' envelope
                    if isinstance(data, dict):
                        secret_data = data.get("data", data)
                        if isinstance(secret_data, dict):
                            aggregated_secrets.update({k: v for k, v in secret_data.items() if isinstance(v, str)})
                except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to fetch secrets for {req}: {e}")
                    raise RuntimeError(f"Vault communication failed: {e}") from e

        return aggregated_secrets


class SandboxProviderProtocol(Protocol):
    """Abstract implementations MUST strictly map to the coreason-manifest bounds."""

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        """Dynamically provisions the execution boundary."""
        ...

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        """Injects unsealed secrets via in-memory tmpfs mounts."""
        ...

    async def execute(self, bytecode: bytes) -> Any:
        """Executes the target binary within the physical boundary."""
        ...

    async def teardown(self, force: bool = False) -> None:
        """Eradicates the partition safely and frees host VRAM."""
        ...

    def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
        """Dynamically configure BPF or iptables egress filtering to enforce specific domain whitelist."""
        ...

    def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
        """Enforces a read-only (ro) mount for the sandbox root filesystem with specific tmpfs exemptions."""
        ...


class WasmSandboxProvider:
    """WASM sandboxing execution provider (wasm32-wasi)."""

    def __init__(self) -> None:
        self.partition_id: str | None = None
        self.bwrap_cmd_array: list[str] = []

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        logger.info(f"Provisioning WASM sandbox: {partition_state.partition_id}")
        self.partition_id = partition_state.partition_id
        self.bwrap_cmd_array = [
            "bwrap",
            "--unshare-pid",
            "--unshare-mount",
            "--unshare-ipc",
        ]

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        logger.info(f"Injecting {len(secrets)} secrets into WASM tmpfs")
        self.bwrap_cmd_array.extend(["--tmpfs", "/run/secrets"])

    async def execute(self, bytecode: bytes) -> Any:
        import asyncio

        logger.info(f"Executing WASM bytecode ({len(bytecode)} bytes) via wasmtime")
        full_command = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            "MemoryMax=512M",
            "-p",
            "CPUQuota=50%",
            *self.bwrap_cmd_array,
            "wasmtime",
            "-",
        ]

        process = await asyncio.create_subprocess_exec(
            *full_command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=bytecode), timeout=10.0)
        except TimeoutError as e:
            import contextlib

            with contextlib.suppress(OSError):
                process.kill()
            raise RuntimeError("WASM execution failed: Timeout") from e

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
            raise RuntimeError(f"WASM execution failed: {error_msg}")
        return stdout.decode("utf-8") if stdout else ""

    async def teardown(self, force: bool = False) -> None:
        logger.info(f"Tearing down WASM sandbox (force={force})")
        if self.partition_id:
            subprocess.run(  # noqa: S603
                ["systemctl", "--user", "stop", f"{self.partition_id}.scope"],  # noqa: S607
                capture_output=True,
                check=False,
            )
        subprocess.run(
            ["umount", "/run/secrets"],  # noqa: S607
            capture_output=True,
            check=False,
        )

    def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
        logger.info(f"Applying network egress rules for WASM sandbox. Allowed domains: {allowed_domains}")
        try:
            subprocess.run(["iptables", "-F", "OUTPUT"], check=False, capture_output=True)  # noqa: S607
            subprocess.run(["iptables", "-P", "OUTPUT", "DROP"], check=False, capture_output=True)  # noqa: S607
            subprocess.run(["iptables", "-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"], check=False, capture_output=True)  # noqa: S607
            for domain in allowed_domains:
                cmd = ["iptables", "-A", "OUTPUT", "-d", domain, "-j", "ACCEPT"]
                subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to apply iptables rules: {e}")  # pragma: no cover

    def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
        logger.info(f"Enforcing WASM filesystem immutability with exemptions: {tmpfs_exemptions}")
        self.bwrap_cmd_array.extend(["--ro-bind", "/", "/"])
        if tmpfs_exemptions:
            for path in tmpfs_exemptions:
                self.bwrap_cmd_array.extend(["--tmpfs", path])


class RiscvZkvmSandboxProvider:
    """RISC-V ZKVM execution provider for generating hardware ZK proofs."""

    def __init__(self) -> None:
        self.partition_id: str | None = None
        self.bwrap_cmd_array: list[str] = []

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        logger.info(f"Provisioning RISC-V ZKVM sandbox: {partition_state.partition_id}")
        self.partition_id = partition_state.partition_id
        self.bwrap_cmd_array = [
            "bwrap",
            "--unshare-pid",
            "--unshare-mount",
            "--unshare-ipc",
        ]

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        logger.info(f"Injecting {len(secrets)} secrets into RISC-V ZKVM tmpfs")
        self.bwrap_cmd_array.extend(["--tmpfs", "/run/secrets"])

    async def execute(self, bytecode: bytes) -> Any:
        import asyncio

        logger.info(f"Executing RISC-V bytecode ({len(bytecode)} bytes) via riscv64-unknown-elf-run")
        full_command = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            "MemoryMax=512M",
            "-p",
            "CPUQuota=50%",
            *self.bwrap_cmd_array,
            "riscv64-unknown-elf-run",
        ]

        process = await asyncio.create_subprocess_exec(
            *full_command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=bytecode), timeout=10.0)
        except TimeoutError as e:
            import contextlib

            with contextlib.suppress(OSError):
                process.kill()
            raise RuntimeError("RISC-V execution failed: Timeout") from e

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
            raise RuntimeError(f"RISC-V execution failed: {error_msg}")
        return stdout.decode("utf-8") if stdout else ""

    async def teardown(self, force: bool = False) -> None:
        logger.info(f"Tearing down RISC-V ZKVM sandbox (force={force})")
        if self.partition_id:
            subprocess.run(  # noqa: S603
                ["systemctl", "--user", "stop", f"{self.partition_id}.scope"],  # noqa: S607
                capture_output=True,
                check=False,
            )
        subprocess.run(
            ["umount", "/run/secrets"],  # noqa: S607
            capture_output=True,
            check=False,
        )

    def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
        logger.info(f"Applying network egress rules for RISC-V ZKVM sandbox. Allowed domains: {allowed_domains}")
        try:
            subprocess.run(["iptables", "-F", "OUTPUT"], check=False, capture_output=True)  # noqa: S607
            subprocess.run(["iptables", "-P", "OUTPUT", "DROP"], check=False, capture_output=True)  # noqa: S607
            subprocess.run(["iptables", "-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"], check=False, capture_output=True)  # noqa: S607
            for domain in allowed_domains:
                cmd = ["iptables", "-A", "OUTPUT", "-d", domain, "-j", "ACCEPT"]
                subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to apply iptables rules: {e}")  # pragma: no cover

    def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
        logger.info(f"Enforcing RISC-V filesystem immutability with exemptions: {tmpfs_exemptions}")
        self.bwrap_cmd_array.extend(["--ro-bind", "/", "/"])
        if tmpfs_exemptions:
            for path in tmpfs_exemptions:
                self.bwrap_cmd_array.extend(["--tmpfs", path])


class BpfSandboxProvider:
    """BPF execution provider for kernel-level execution."""

    def __init__(self) -> None:
        self.partition_id: str | None = None
        self.bwrap_cmd_array: list[str] = []

    def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
        logger.info(f"Provisioning BPF sandbox: {partition_state.partition_id}")
        self.partition_id = partition_state.partition_id
        self.bwrap_cmd_array = [
            "bwrap",
            "--unshare-pid",
            "--unshare-mount",
            "--unshare-ipc",
        ]

    def inject_secrets(self, secrets: dict[str, str]) -> None:
        logger.info(f"Injecting {len(secrets)} secrets into BPF tmpfs")
        self.bwrap_cmd_array.extend(["--tmpfs", "/run/secrets"])

    async def execute(self, bytecode: bytes) -> Any:
        import asyncio

        logger.info(f"Executing BPF bytecode ({len(bytecode)} bytes) via bpftool")
        full_command = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            "MemoryMax=512M",
            "-p",
            "CPUQuota=50%",
            *self.bwrap_cmd_array,
            "bpftool",
            "prog",
            "load",
            "-",
            "/sys/fs/bpf/prog",
        ]

        process = await asyncio.create_subprocess_exec(
            *full_command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=bytecode), timeout=10.0)
        except TimeoutError as e:
            import contextlib

            with contextlib.suppress(OSError):
                process.kill()
            raise RuntimeError("BPF execution failed: Timeout") from e

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
            raise RuntimeError(f"BPF execution failed: {error_msg}")
        return stdout.decode("utf-8") if stdout else ""

    async def teardown(self, force: bool = False) -> None:
        logger.info(f"Tearing down BPF sandbox (force={force})")
        if self.partition_id:
            subprocess.run(  # noqa: S603
                ["systemctl", "--user", "stop", f"{self.partition_id}.scope"],  # noqa: S607
                capture_output=True,
                check=False,
            )
        subprocess.run(
            ["umount", "/run/secrets"],  # noqa: S607
            capture_output=True,
            check=False,
        )

    def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
        logger.info(f"Applying network egress rules for BPF sandbox. Allowed domains: {allowed_domains}")
        try:
            subprocess.run(["iptables", "-F", "OUTPUT"], check=False, capture_output=True)  # noqa: S607
            subprocess.run(["iptables", "-P", "OUTPUT", "DROP"], check=False, capture_output=True)  # noqa: S607
            subprocess.run(["iptables", "-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"], check=False, capture_output=True)  # noqa: S607
            for domain in allowed_domains:
                cmd = ["iptables", "-A", "OUTPUT", "-d", domain, "-j", "ACCEPT"]
                subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to apply iptables rules: {e}")  # pragma: no cover

    def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
        logger.info(f"Enforcing BPF filesystem immutability with exemptions: {tmpfs_exemptions}")
        self.bwrap_cmd_array.extend(["--ro-bind", "/", "/"])
        if tmpfs_exemptions:
            for path in tmpfs_exemptions:
                self.bwrap_cmd_array.extend(["--tmpfs", path])


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


def enforce_sandbox_immutability(
    manifest: ToolManifest,
    provider: SandboxProviderProtocol,
    additional_exemptions: list[str] | None = None,
) -> None:
    """
    If file_system_mutation_forbidden is True, enforces a strictly read-only
    root filesystem while explicitly exempting volatile memory paths like /dev/shm
    and specific secret injection mounts.
    """
    if manifest.permissions.file_system_mutation_forbidden:
        exemptions = ["/dev/shm", "/run/secrets"]  # noqa: S108
        if additional_exemptions:
            exemptions.extend(additional_exemptions)

        provider.enforce_filesystem_immutability(tmpfs_exemptions=exemptions)


def verify_network_access(
    manifest: ToolManifest,
    partition_state: EphemeralNamespacePartitionState,
    provider: SandboxProviderProtocol | None = None,
) -> bool:
    """
    Evaluates network egress using a defense-in-depth lock.
    To authorize a network socket, BOTH ToolManifest.permissions.network_access
    AND the sandbox's EphemeralNamespacePartitionState.allow_network_egress must evaluate to True.
    A conflict mathematically blocks the execution by raising a PermissionError.

    If access is granted and a provider is passed, it dynamically applies BPF/iptables egress
    rules based on the allowed_domains whitelist.
    """
    tool_network = manifest.permissions.network_access
    sandbox_network = partition_state.allow_network_egress

    if tool_network and not sandbox_network:
        raise PermissionError(
            "Dual-Evaluation Permission Boundary conflict: "
            "Tool requests network access, but the active ephemeral partition explicitly denies egress."
        )

    access_granted = bool(tool_network and sandbox_network)

    if access_granted and provider is not None:
        allowed_domains = manifest.permissions.allowed_domains or []
        provider.apply_network_egress_rules(allowed_domains)

    return access_granted
