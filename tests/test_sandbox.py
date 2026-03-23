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
    object.__setattr__(state, "execution_runtime", "invalid_runtime")
    with pytest.raises(ValueError, match="Legacy containerization strictly prohibited"):
        SandboxProviderFactory.create(state)


@pytest.mark.asyncio
async def test_wasm_provider_methods() -> None:
    provider = WasmSandboxProvider()
    state = create_partition_state("wasm32-wasi")
    provider.provision(state)
    assert provider.partition_id == "test_part"

    provider.apply_network_egress_rules(["coreason.ai"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108

    provider.inject_secrets({"secret": "value"})
    from unittest.mock import AsyncMock, patch

    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = "WASM execution mock success"

        res = await provider.execute(b"(module)")
        assert res == "WASM execution mock success"

    # Test apply_network_egress_rules
    with patch("subprocess.run") as mock_run:
        provider.apply_network_egress_rules(["coreason.ai"])
        # wasmtime implicitly blocks egress

    with patch("subprocess.run") as mock_run:
        await provider.teardown()
        assert mock_run.call_count == 0


@pytest.mark.asyncio
async def test_riscv_provider_methods() -> None:
    provider = RiscvZkvmSandboxProvider()
    state = create_partition_state("riscv32-zkvm")
    provider.provision(state)
    assert provider.partition_id == "test_part"

    provider.apply_network_egress_rules(["coreason.ai"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108

    provider.inject_secrets({"secret": "value"})
    assert "--tmpfs" in provider.bwrap_cmd_array

    from unittest.mock import AsyncMock, patch

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"RISC-V execution mock success", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        res = await provider.execute(b"(module)")
        assert res == ("RISC-V execution mock success", {"stdout": "RISC-V execution mock success"})

        execute_call = mock_exec.call_args_list[0]
        called_cmd = execute_call[0]
        assert called_cmd[0] == "systemd-run"
        assert "riscv64-unknown-elf-run" in called_cmd

    # Test apply_network_egress_rules
    with patch("subprocess.run") as mock_run:
        provider.apply_network_egress_rules(["coreason.ai"])
        assert mock_run.call_count > 0

    with patch("subprocess.run") as mock_run:
        await provider.teardown()
        assert mock_run.call_count == 2

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec2:
            mock_process2 = AsyncMock()
            mock_process2.returncode = 1
            mock_process2.communicate.return_value = (b"", b"RISC-V execution failed: some error")
            mock_exec2.return_value = mock_process2
            with pytest.raises(RuntimeError, match="RISC-V execution failed:"):
                await provider.execute(b"(module)")


@pytest.mark.asyncio
async def test_bpf_provider_methods() -> None:
    provider = BpfSandboxProvider()
    state = create_partition_state("bpf")
    provider.provision(state)
    assert provider.partition_id == "test_part"

    provider.apply_network_egress_rules(["coreason.ai"])
    provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108

    provider.inject_secrets({"secret": "value"})
    assert "--tmpfs" in provider.bwrap_cmd_array

    from unittest.mock import AsyncMock, patch

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"BPF execution mock success", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        res = await provider.execute(b"(module)")
        assert res == "BPF execution mock success"

        execute_call = mock_exec.call_args_list[0]
        called_cmd = execute_call[0]
        assert called_cmd[0] == "systemd-run"
        assert "bpftool" in called_cmd

    # Test apply_network_egress_rules
    with patch("subprocess.run") as mock_run:
        provider.apply_network_egress_rules(["coreason.ai"])
        assert mock_run.call_count > 0

    with patch("subprocess.run") as mock_run:
        await provider.teardown()
        assert mock_run.call_count == 2

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec2:
            mock_process2 = AsyncMock()
            mock_process2.returncode = 1
            mock_process2.communicate.return_value = (b"", b"BPF execution failed: some error")
            mock_exec2.return_value = mock_process2
            with pytest.raises(RuntimeError, match="BPF execution failed:"):
                await provider.execute(b"(module)")


@pytest.mark.asyncio
async def test_stateful_sandbox_cache_warm_start() -> None:
    cache = StatefulSandboxCache(max_size=2)
    session_state = SecureSubSessionState(
        session_id="session_1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test"
    )
    partition_state = create_partition_state("wasm32-wasi")

    # Cold start
    provider1 = await cache.get_or_create(session_state, partition_state)

    # Warm start
    provider2 = await cache.get_or_create(session_state, partition_state)

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

    p1 = await cache.get_or_create(session1, partition_state)
    p1.teardown = AsyncMock()  # type: ignore

    _ = await cache.get_or_create(session2, partition_state)

    assert len(cache._cache) == 2

    # Adding third should evict first (s1)
    _ = await cache.get_or_create(session3, partition_state)

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

    p1 = await cache.get_or_create(session1, partition_state)
    p1.teardown = AsyncMock()  # type: ignore

    await cache.teardown_all()

    assert len(cache._cache) == 0
    p1.teardown.assert_called_once()


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
