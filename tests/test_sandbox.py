# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import hashlib
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from coreason_manifest.spec.ontology import (
    EphemeralNamespacePartitionState,
    PermissionBoundaryPolicy,
    SecureSubSessionState,
    SideEffectProfile,
    ToolManifest,
)

from coreason_actuator.sandbox import (
    BpfSandboxProvider,
    EnterpriseVaultProtocol,
    HashiCorpVault,
    RiscvZkvmSandboxProvider,
    SandboxProviderFactory,
    StatefulSandboxCache,
    WasmSandboxProvider,
    enforce_sandbox_immutability,
    verify_bytecode_safety,
    verify_network_access,
)


class MockVault:
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets

    async def unseal(self, auth_requirements: list[str]) -> dict[str, str]:
        _ = auth_requirements
        return self.secrets


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_hashicorp_vault(mock_get: MagicMock) -> None:
    # Setup mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": {"real_secret_key": "real_secret_value"}}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    vault = HashiCorpVault(vault_addr="http://localhost:8200", vault_token="test-token")  # noqa: S106
    secrets = await vault.unseal(["oauth2:github"])
    assert secrets == {"real_secret_key": "real_secret_value"}

    # Verify the request was made correctly
    mock_get.assert_called_once_with("http://localhost:8200/oauth2:github", headers={"X-Vault-Token": "test-token"})


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_hashicorp_vault_error(mock_get: MagicMock) -> None:
    import httpx

    mock_get.side_effect = httpx.RequestError("Network error")

    vault = HashiCorpVault(vault_addr="http://localhost:8200", vault_token="test-token")  # noqa: S106
    with pytest.raises(RuntimeError, match="Vault communication failed: Network error"):
        await vault.unseal(["oauth2:github"])


@pytest.mark.asyncio
async def test_vault_protocol_mock() -> None:
    vault: EnterpriseVaultProtocol = MockVault({"key": "val"})
    secrets = await vault.unseal(["oauth2:github"])
    assert secrets == {"key": "val"}


def create_partition_state(runtime: Any) -> EphemeralNamespacePartitionState:
    return EphemeralNamespacePartitionState(
        partition_id="test_part",
        execution_runtime=runtime,
        authorized_bytecode_hashes=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
        max_ttl_seconds=300,
        max_vram_mb=512,
        allow_network_egress=False,
        allow_subprocess_spawning=False,
    )


def test_sandbox_factory_wasm() -> None:
    state = create_partition_state("wasm32-wasi")
    provider = SandboxProviderFactory.create(state)
    assert isinstance(provider, WasmSandboxProvider)


def test_sandbox_factory_riscv() -> None:
    state = create_partition_state("riscv32-zkvm")
    provider = SandboxProviderFactory.create(state)
    assert isinstance(provider, RiscvZkvmSandboxProvider)


def test_sandbox_factory_bpf() -> None:
    state = create_partition_state("bpf")
    provider = SandboxProviderFactory.create(state)
    assert isinstance(provider, BpfSandboxProvider)


def test_sandbox_factory_invalid() -> None:
    # We bypass Pydantic validation here to test the ValueError explicitly.
    # In reality, Pydantic would catch this first.
    state = create_partition_state("wasm32-wasi")
    object.__setattr__(state, "execution_runtime", "docker")
    with pytest.raises(ValueError, match="Legacy containerization strictly prohibited"):
        SandboxProviderFactory.create(state)


@pytest.mark.asyncio
@patch("subprocess.run")
async def test_wasm_provider_methods(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(stdout=b"WASM execution mock success")
    provider = WasmSandboxProvider()
    state = create_partition_state("wasm32-wasi")
    provider.provision(state)
    assert provider.partition_id == "test_part"
    assert provider.bwrap_cmd_array == ["bwrap", "--unshare-pid", "--unshare-mount", "--unshare-ipc"]

    provider.apply_network_egress_rules(["coreason.ai"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108

    provider.inject_secrets({"secret": "value"})
    assert "--tmpfs" in provider.bwrap_cmd_array
    assert "/run/secrets" in provider.bwrap_cmd_array

    # Reset mock since apply_network_egress_rules uses subprocess.run
    mock_run.reset_mock()
    res = provider.execute(b"test")
    assert res == "WASM execution mock success"

    # Verify execute called with systemd-run
    execute_call = mock_run.call_args_list[0]
    called_cmd = execute_call[0][0]
    assert called_cmd[0] == "systemd-run"
    assert "wasmtime" in called_cmd

    # Reset mock for teardown
    mock_run.reset_mock()
    await provider.teardown()

    # Teardown should have 2 calls: systemctl stop, and umount
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0][0][0] == ["systemctl", "--user", "stop", "test_part.scope"]
    assert mock_run.call_args_list[1][0][0] == ["umount", "/run/secrets"]

    # Test subprocess failure fallback
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
    with pytest.raises(RuntimeError, match="WASM execution failed"):
        provider.execute(b"test")


@pytest.mark.asyncio
@patch("subprocess.run")
async def test_riscv_provider_methods(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(stdout=b"RISC-V execution mock success")
    provider = RiscvZkvmSandboxProvider()
    state = create_partition_state("riscv32-zkvm")
    provider.provision(state)
    assert provider.partition_id == "test_part"

    provider.apply_network_egress_rules(["coreason.ai"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108

    provider.inject_secrets({"secret": "value"})
    assert "--tmpfs" in provider.bwrap_cmd_array

    # Reset mock since apply_network_egress_rules uses subprocess.run
    mock_run.reset_mock()
    res = provider.execute(b"test")
    assert res == "RISC-V execution mock success"

    # Verify execute called with systemd-run
    execute_call = mock_run.call_args_list[0]
    called_cmd = execute_call[0][0]
    assert called_cmd[0] == "systemd-run"
    assert "riscv64-unknown-elf-run" in called_cmd

    # Reset mock for teardown
    mock_run.reset_mock()
    await provider.teardown()

    # Teardown calls
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0][0][0] == ["systemctl", "--user", "stop", "test_part.scope"]
    assert mock_run.call_args_list[1][0][0] == ["umount", "/run/secrets"]

    # Test subprocess failure fallback
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
    with pytest.raises(RuntimeError, match="RISC-V execution failed"):
        provider.execute(b"test")


@pytest.mark.asyncio
@patch("subprocess.run")
async def test_bpf_provider_methods(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(stdout=b"BPF execution mock success")
    provider = BpfSandboxProvider()
    state = create_partition_state("bpf")
    provider.provision(state)
    assert provider.partition_id == "test_part"

    provider.apply_network_egress_rules(["coreason.ai"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108

    provider.inject_secrets({"secret": "value"})
    assert "--tmpfs" in provider.bwrap_cmd_array

    # Reset mock since apply_network_egress_rules uses subprocess.run
    mock_run.reset_mock()
    res = provider.execute(b"test")
    assert res == "BPF execution mock success"

    # Verify execute called with systemd-run
    execute_call = mock_run.call_args_list[0]
    called_cmd = execute_call[0][0]
    assert called_cmd[0] == "systemd-run"
    assert "bpftool" in called_cmd

    # Reset mock for teardown
    mock_run.reset_mock()
    await provider.teardown()

    # Teardown calls
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0][0][0] == ["systemctl", "--user", "stop", "test_part.scope"]
    assert mock_run.call_args_list[1][0][0] == ["umount", "/run/secrets"]

    # Test subprocess failure fallback
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
    with pytest.raises(RuntimeError, match="BPF execution failed"):
        provider.execute(b"test")


def test_stateful_sandbox_cache_warm_start() -> None:
    cache = StatefulSandboxCache(max_size=2)
    session_state = SecureSubSessionState(
        session_id="session_1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test"
    )
    partition_state = create_partition_state("wasm32-wasi")

    # Cold start
    provider1 = cache.get_or_create(session_state, partition_state)

    # Warm start
    provider2 = cache.get_or_create(session_state, partition_state)

    assert provider1 is provider2
    assert len(cache._cache) == 1


@pytest.mark.asyncio
async def test_stateful_sandbox_cache_eviction() -> None:
    cache = StatefulSandboxCache(max_size=2)
    partition_state = create_partition_state("wasm32-wasi")

    session1 = SecureSubSessionState(session_id="s1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    session2 = SecureSubSessionState(session_id="s2", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    session3 = SecureSubSessionState(session_id="s3", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")

    from unittest.mock import AsyncMock

    p1 = cache.get_or_create(session1, partition_state)
    p1.teardown = AsyncMock()  # type: ignore

    _ = cache.get_or_create(session2, partition_state)

    assert len(cache._cache) == 2

    # Adding third should evict first (s1)
    _ = cache.get_or_create(session3, partition_state)

    assert len(cache._cache) == 2
    assert "s1" not in cache._cache
    assert "s2" in cache._cache
    assert "s3" in cache._cache

    # Wait for the background task of teardown to finish
    import asyncio

    await asyncio.sleep(0.01)

    p1.teardown.assert_called_once()  # Eviction calls teardown


@pytest.mark.asyncio
async def test_stateful_sandbox_cache_teardown_all() -> None:
    cache = StatefulSandboxCache(max_size=2)
    partition_state = create_partition_state("wasm32-wasi")

    session1 = SecureSubSessionState(session_id="s1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    from unittest.mock import AsyncMock

    p1 = cache.get_or_create(session1, partition_state)
    p1.teardown = AsyncMock()  # type: ignore

    await cache.teardown_all()

    assert len(cache._cache) == 0
    p1.teardown.assert_called_once()


@pytest.mark.asyncio
async def test_stateful_sandbox_cache_eviction_no_loop() -> None:
    # We simulate a runtime error when getting the event loop
    cache = StatefulSandboxCache(max_size=2)
    partition_state = create_partition_state("wasm32-wasi")

    session1 = SecureSubSessionState(session_id="s1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    session2 = SecureSubSessionState(session_id="s2", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    session3 = SecureSubSessionState(session_id="s3", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")

    from unittest.mock import AsyncMock, patch

    p1 = cache.get_or_create(session1, partition_state)
    p1.teardown = AsyncMock()  # type: ignore

    _ = cache.get_or_create(session2, partition_state)

    assert len(cache._cache) == 2

    # Force RuntimeError from get_running_loop
    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
        _ = cache.get_or_create(session3, partition_state)

    assert len(cache._cache) == 2
    assert "s1" not in cache._cache
    assert "s2" in cache._cache
    assert "s3" in cache._cache

    # Since there was no loop, teardown was never scheduled
    p1.teardown.assert_not_called()


def test_verify_bytecode_safety_success() -> None:
    bytecode = b"print('hello')"
    bytecode_hash = hashlib.sha256(bytecode).hexdigest()

    assert verify_bytecode_safety(bytecode, [bytecode_hash, "other_hash"]) is True


def test_verify_bytecode_safety_failure() -> None:
    bytecode = b"print('hello')"

    assert verify_bytecode_safety(bytecode, ["invalid_hash"]) is False


def create_tool_manifest(network_access: bool) -> ToolManifest:
    return ToolManifest(
        tool_name="test_tool",
        description="A test tool",
        input_schema={"type": "object"},
        side_effects=SideEffectProfile(mutates_state=False, is_idempotent=True),
        permissions=PermissionBoundaryPolicy(
            network_access=network_access,
            file_system_mutation_forbidden=True,
        ),
    )


def test_verify_network_access_both_true() -> None:
    manifest = create_tool_manifest(network_access=True)
    partition_state = create_partition_state("wasm32-wasi")
    # Need to override allow_network_egress
    object.__setattr__(partition_state, "allow_network_egress", True)

    assert verify_network_access(manifest, partition_state) is True


def test_verify_network_access_with_provider_rules_applied() -> None:
    manifest = create_tool_manifest(network_access=True)
    # Give the manifest some allowed domains
    object.__setattr__(manifest.permissions, "allowed_domains", ["coreason.ai", "example.com"])

    partition_state = create_partition_state("wasm32-wasi")
    object.__setattr__(partition_state, "allow_network_egress", True)

    class MockProvider:
        def __init__(self) -> None:
            self.applied_domains: list[str] = []

        def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
            pass

        def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
            self.applied_domains = allowed_domains

        def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
            pass

        def inject_secrets(self, secrets: dict[str, str]) -> None:
            pass

        def execute(self, bytecode: bytes) -> Any:
            pass

        async def teardown(self, force: bool = False) -> None:
            pass

    provider = MockProvider()

    assert verify_network_access(manifest, partition_state, provider) is True
    assert provider.applied_domains == ["coreason.ai", "example.com"]


def test_verify_network_access_both_false() -> None:
    manifest = create_tool_manifest(network_access=False)
    partition_state = create_partition_state("wasm32-wasi")
    object.__setattr__(partition_state, "allow_network_egress", False)

    assert verify_network_access(manifest, partition_state) is False


def test_verify_network_access_tool_false_sandbox_true() -> None:
    manifest = create_tool_manifest(network_access=False)
    partition_state = create_partition_state("wasm32-wasi")
    object.__setattr__(partition_state, "allow_network_egress", True)

    assert verify_network_access(manifest, partition_state) is False


def test_verify_network_access_conflict_raises() -> None:
    manifest = create_tool_manifest(network_access=True)
    partition_state = create_partition_state("wasm32-wasi")
    object.__setattr__(partition_state, "allow_network_egress", False)

    with pytest.raises(PermissionError, match="Dual-Evaluation Permission Boundary conflict"):
        verify_network_access(manifest, partition_state)


def test_enforce_sandbox_immutability() -> None:
    manifest = create_tool_manifest(network_access=False)

    class MockProvider:
        def __init__(self) -> None:
            self.exemptions: list[str] | None = None

        def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
            pass

        def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
            pass

        def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
            self.exemptions = tmpfs_exemptions

        def inject_secrets(self, secrets: dict[str, str]) -> None:
            pass

        def execute(self, bytecode: bytes) -> Any:
            pass

        async def teardown(self, force: bool = False) -> None:
            pass

    provider = MockProvider()

    enforce_sandbox_immutability(manifest, provider, additional_exemptions=["/extra"])

    assert provider.exemptions == ["/dev/shm", "/run/secrets", "/extra"]  # noqa: S108


def test_enforce_sandbox_immutability_not_forbidden() -> None:
    manifest = create_tool_manifest(network_access=False)
    object.__setattr__(manifest.permissions, "file_system_mutation_forbidden", False)

    class MockProvider:
        def __init__(self) -> None:
            self.exemptions: list[str] | None = None

        def provision(self, partition_state: EphemeralNamespacePartitionState) -> None:
            pass

        def apply_network_egress_rules(self, allowed_domains: list[str]) -> None:
            pass

        def enforce_filesystem_immutability(self, tmpfs_exemptions: list[str] | None = None) -> None:
            self.exemptions = tmpfs_exemptions

        def inject_secrets(self, secrets: dict[str, str]) -> None:
            pass

        def execute(self, bytecode: bytes) -> Any:
            pass

        async def teardown(self, force: bool = False) -> None:
            pass

    provider = MockProvider()

    enforce_sandbox_immutability(manifest, provider)

    assert provider.exemptions is None
