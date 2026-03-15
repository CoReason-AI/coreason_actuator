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
from unittest.mock import MagicMock

import pytest
from coreason_manifest.spec.ontology import (
    EphemeralNamespacePartitionState,
    SecureSubSessionState,
)

from coreason_actuator.sandbox import (
    BpfSandboxProvider,
    EnterpriseVaultProtocol,
    RiscvZkvmSandboxProvider,
    SandboxProviderFactory,
    StatefulSandboxCache,
    WasmSandboxProvider,
    verify_bytecode_safety,
)


class MockVault:
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets

    def unseal(self, auth_requirements: list[str]) -> dict[str, str]:
        _ = auth_requirements
        return self.secrets


def test_vault_protocol_mock() -> None:
    vault: EnterpriseVaultProtocol = MockVault({"key": "val"})
    secrets = vault.unseal(["oauth2:github"])
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


def test_wasm_provider_methods() -> None:
    provider = WasmSandboxProvider()
    state = create_partition_state("wasm32-wasi")
    provider.provision(state)
    provider.inject_secrets({"secret": "value"})
    res = provider.execute(b"test")
    assert res == "WASM execution simulated"
    provider.teardown()


def test_riscv_provider_methods() -> None:
    provider = RiscvZkvmSandboxProvider()
    state = create_partition_state("riscv32-zkvm")
    provider.provision(state)
    provider.inject_secrets({"secret": "value"})
    res = provider.execute(b"test")
    assert res == "RISC-V execution simulated"
    provider.teardown()


def test_bpf_provider_methods() -> None:
    provider = BpfSandboxProvider()
    state = create_partition_state("bpf")
    provider.provision(state)
    provider.inject_secrets({"secret": "value"})
    res = provider.execute(b"test")
    assert res == "BPF execution simulated"
    provider.teardown()


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


def test_stateful_sandbox_cache_eviction() -> None:
    cache = StatefulSandboxCache(max_size=2)
    partition_state = create_partition_state("wasm32-wasi")

    session1 = SecureSubSessionState(session_id="s1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    session2 = SecureSubSessionState(session_id="s2", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    session3 = SecureSubSessionState(session_id="s3", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")

    p1 = cache.get_or_create(session1, partition_state)
    p1.teardown = MagicMock()  # type: ignore

    _ = cache.get_or_create(session2, partition_state)

    assert len(cache._cache) == 2

    # Adding third should evict first (s1)
    _ = cache.get_or_create(session3, partition_state)

    assert len(cache._cache) == 2
    assert "s1" not in cache._cache
    assert "s2" in cache._cache
    assert "s3" in cache._cache
    p1.teardown.assert_called_once()  # Eviction calls teardown


def test_stateful_sandbox_cache_teardown_all() -> None:
    cache = StatefulSandboxCache(max_size=2)
    partition_state = create_partition_state("wasm32-wasi")

    session1 = SecureSubSessionState(session_id="s1", allowed_vault_keys=[], max_ttl_seconds=300, description="Test")
    p1 = cache.get_or_create(session1, partition_state)
    p1.teardown = MagicMock()  # type: ignore

    cache.teardown_all()

    assert len(cache._cache) == 0
    p1.teardown.assert_called_once()


def test_verify_bytecode_safety_success() -> None:
    bytecode = b"print('hello')"
    bytecode_hash = hashlib.sha256(bytecode).hexdigest()

    assert verify_bytecode_safety(bytecode, [bytecode_hash, "other_hash"]) is True


def test_verify_bytecode_safety_failure() -> None:
    bytecode = b"print('hello')"

    assert verify_bytecode_safety(bytecode, ["invalid_hash"]) is False
