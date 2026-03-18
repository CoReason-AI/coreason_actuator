# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator
import asyncio
import concurrent.futures
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_manifest.spec.ontology import (
    AgentAttestationReceipt,
    ExecutionSLA,
    MCPCapabilityWhitelistPolicy,
    MCPServerManifest,
    PermissionBoundaryPolicy,
    SideEffectProfile,
    StdioTransportProfile,
    ToolInvocationEvent,
    ToolManifest,
    VerifiableCredentialPresentationReceipt,
    ZeroKnowledgeReceipt,
)
from hypothesis import given
from hypothesis import strategies as st

from coreason_actuator.strategies import (
    KinematicExecutionStrategy,
    MCPClientStrategy,
    NativeExecutionStrategy,
    RedisDistributedLock,
)


class MockLockManager:
    def __init__(self) -> None:
        self.acquired_locks: list[tuple[str, int | None]] = []
        self.released_locks: list[str] = []

    @asynccontextmanager
    async def acquire(self, lock_key: str, ttl: int | None = None) -> AsyncIterator[Any]:
        self.acquired_locks.append((lock_key, ttl))
        try:
            yield self
        finally:
            self.released_locks.append(lock_key)


class MockRegistry:
    def __init__(self, callable_map: dict[str, Callable[..., Awaitable[Any]]]):
        self.callable_map = callable_map

    async def get_callable(self, tool_name: str) -> Callable[..., Awaitable[Any]] | None:
        return self.callable_map.get(tool_name)


class MockMCPServerRegistry:
    def __init__(self, server_map: dict[str, MCPServerManifest]):
        self.server_map = server_map

    async def get_server_manifest(self, tool_name: str) -> MCPServerManifest | None:
        return self.server_map.get(tool_name)


class MockMCPTransport:
    def __init__(self) -> None:
        self.dispatched_packets: list[tuple[MCPServerManifest, dict[str, Any]]] = []
        self.mock_response = "mock_response"

    async def dispatch(self, server_manifest: MCPServerManifest, packet: dict[str, Any]) -> Any:
        self.dispatched_packets.append((server_manifest, packet))
        return self.mock_response


def create_mock_credential_receipt() -> VerifiableCredentialPresentationReceipt:
    return VerifiableCredentialPresentationReceipt(
        presentation_format="jwt_vc",
        issuer_did="did:coreason:mcp-authority",
        cryptographic_proof_blob="mock_blob",
        authorization_claims={"mock_claim": "mock_value"},
    )


def create_mock_attestation() -> AgentAttestationReceipt:
    valid_hash = "a" * 64
    return AgentAttestationReceipt(
        training_lineage_hash=valid_hash,
        developer_signature="mock",
        capability_merkle_root=valid_hash,
        credential_presentations=[],
    )


def create_mock_zk_proof() -> ZeroKnowledgeReceipt:
    return ZeroKnowledgeReceipt.model_construct(
        proof_protocol="zk-SNARK",
        public_inputs_hash="mock",
        verifier_key_id="mock",
        cryptographic_blob="mock",
    )


def create_mock_manifest(mutates_state: bool = False, is_idempotent: bool = False) -> ToolManifest:
    side_effects = SideEffectProfile(mutates_state=mutates_state, is_idempotent=is_idempotent)
    permissions = PermissionBoundaryPolicy(
        network_access=False, allowed_domains=[], file_system_mutation_forbidden=True, auth_requirements=[]
    )
    sla = ExecutionSLA(max_execution_time_ms=1000, max_compute_footprint_mb=128)
    return ToolManifest(
        tool_name="test_tool",
        description="A test tool",
        input_schema={"type": "object"},
        side_effects=side_effects,
        permissions=permissions,
        sla=sla,
        is_preemptible=False,
    )


class MockAsyncpgConnection:
    def __init__(self) -> None:
        self.executed_queries: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed_queries.append((query, args))


class MockAsyncpgPool:
    def __init__(self, conn: Any = None) -> None:
        self.conn = conn or MockAsyncpgConnection()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        yield self.conn


class MockRedisClient:
    def __init__(self, set_return: bool = True, set_side_effect: Exception | None = None) -> None:
        self.set_return = set_return
        self.set_side_effect = set_side_effect
        self.set_called = False
        self.delete_called = False
        self.aclose_called = False
        self.set_args: Any = None

    async def set(self, *args: Any, **kwargs: Any) -> Any:
        self.set_called = True
        self.set_args = (args, kwargs)
        if self.set_side_effect:
            raise self.set_side_effect
        return self.set_return

    async def delete(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.delete_called = True

    async def aclose(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.aclose_called = True


@pytest.mark.asyncio
async def test_redis_distributed_lock_acquire() -> None:
    mock_redis = MockRedisClient(set_return=True)
    from unittest.mock import patch

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        lock = RedisDistributedLock()
        async with lock.acquire("test_key", ttl=1000):
            pass
        assert mock_redis.set_called
        assert mock_redis.delete_called
        assert mock_redis.aclose_called


@pytest.mark.asyncio
async def test_redis_distributed_lock_acquire_timeout() -> None:
    mock_redis = MockRedisClient(set_return=False)
    from unittest.mock import patch

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        lock = RedisDistributedLock()
        with pytest.raises(TimeoutError, match="Failed to acquire lock for test_key within TTL 10ms"):
            async with lock.acquire("test_key", ttl=10):
                pass
        assert mock_redis.set_called
        assert mock_redis.aclose_called


@pytest.mark.asyncio
async def test_redis_distributed_lock_acquire_other_error() -> None:
    mock_redis = MockRedisClient(set_side_effect=ValueError("some other error"))
    from unittest.mock import patch

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        lock = RedisDistributedLock()
        with pytest.raises(ValueError, match="some other error"):
            async with lock.acquire("test_key", ttl=10):
                pass
        assert mock_redis.set_called


@pytest.mark.asyncio
async def test_redis_distributed_lock_acquire_no_ttl() -> None:
    mock_redis = MockRedisClient(set_return=True)
    from unittest.mock import patch

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        lock = RedisDistributedLock()
        async with lock.acquire("test_key_no_ttl"):
            pass
        assert mock_redis.set_called
        assert mock_redis.delete_called
        assert mock_redis.aclose_called


@pytest.mark.asyncio
async def test_native_execution_strategy_success() -> None:
    mock_callable = AsyncMock(return_value="success")
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()

    strategy = NativeExecutionStrategy(registry, lock_manager)
    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"arg1": "safe_value"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid=None)

    assert result == "success"
    mock_callable.assert_called_once_with(arg1="safe_value")
    assert len(lock_manager.acquired_locks) == 0


@pytest.mark.asyncio
async def test_native_execution_strategy_missing_tool() -> None:
    registry = MockRegistry({})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)
    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="missing_tool",
        parameters={},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(ValueError, match="Tool missing_tool not found in native registry"):
        await strategy.execute(intent, manifest, sandbox_pid=None)


@pytest.mark.asyncio
async def test_native_execution_strategy_ast_safety_failure() -> None:
    mock_callable = AsyncMock()
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    # This string is valid syntax but will fail verify_ast_safety due to allowed node limits
    # e.g., using Call or something if not allowed, or Attribute.
    unsafe_payload = "os.system('echo hacked')"

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"code": unsafe_payload},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid=None)
    assert result.payload["execution_status"] == "fatal_crash"
    assert (
        "Kinetic execution bleed detected" in result.payload["traceback"]
        or "AST safety verification failed" in result.payload["traceback"]
    )


@pytest.mark.asyncio
async def test_native_execution_strategy_ast_safety_explicit_false() -> None:
    mock_callable = AsyncMock()
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"code": "dummy"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.MonkeyPatch.context() as m:
        m.setattr("coreason_actuator.strategies.verify_ast_safety", MagicMock(return_value=False))
        result = await strategy.execute(intent, manifest, sandbox_pid=None)
        assert result.payload["execution_status"] == "fatal_crash"
        assert "AST safety verification failed for parameter 'code'" in result.payload["traceback"]


@pytest.mark.asyncio
async def test_native_execution_strategy_ast_safety_unknown_error() -> None:
    mock_callable = AsyncMock()
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"code": "dummy"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.MonkeyPatch.context() as m:
        m.setattr("coreason_actuator.strategies.verify_ast_safety", MagicMock(side_effect=Exception("Unknown Error")))
        result = await strategy.execute(intent, manifest, sandbox_pid=None)
        assert result.payload["execution_status"] == "fatal_crash"
        assert "Unknown Error" in result.payload["traceback"]


@pytest.mark.asyncio
async def test_native_execution_strategy_ast_safety_success_with_string() -> None:
    mock_callable = AsyncMock(return_value="success")
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    # This string is valid syntax and allowed AST nodes (just a string representation of list or simple math)
    safe_payload = "[1, 2, 3]"

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"code": safe_payload},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid=None)
    assert result == "success"
    mock_callable.assert_called_once_with(code=safe_payload)


@pytest.mark.asyncio
async def test_native_execution_strategy_idempotency_retry() -> None:
    # We simulate a failure that succeeds on the second try
    call_count = 0

    async def side_effect(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Transient network error")
        return "success_on_retry"

    mock_callable = AsyncMock(side_effect=side_effect)
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"arg1": "safe_value"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest(is_idempotent=True)

    # Note: wait_exponential min=1 means it will sleep for at least 1s.
    # To avoid slow tests, we would normally patch wait_exponential or retry configuration,
    # but for simplicity we let it run or patch tenacity's sleep.
    # Since we can't easily patch decorator arguments after definition,
    # we'll just mock asyncio.sleep.
    import tenacity.nap

    with pytest.MonkeyPatch.context() as m:
        m.setattr(tenacity.nap, "sleep", MagicMock())
        result = await strategy.execute(intent, manifest, sandbox_pid=None)

    assert result == "success_on_retry"
    assert mock_callable.call_count == 2
    assert len(lock_manager.acquired_locks) == 0


@pytest.mark.asyncio
async def test_native_execution_strategy_idempotency_no_retry_on_other_errors() -> None:
    # We simulate a failure that is NOT a ConnectionError or TimeoutError
    call_count = 0

    async def side_effect(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        raise ValueError("Some other error")

    mock_callable = AsyncMock(side_effect=side_effect)
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"arg1": "safe_value"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest(is_idempotent=True)

    import tenacity.nap

    with pytest.MonkeyPatch.context() as m:
        m.setattr(tenacity.nap, "sleep", MagicMock())
        with pytest.raises(ValueError, match="Some other error"):
            await strategy.execute(intent, manifest, sandbox_pid=None)

    # Because ValueError is not in the retry_if_exception_type list, it should only be called once
    assert mock_callable.call_count == 1
    assert len(lock_manager.acquired_locks) == 0


@pytest.mark.asyncio
async def test_native_execution_strategy_state_mutation_locking() -> None:
    mock_callable = AsyncMock(return_value="mutated")
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    params = {"target": "database", "action": "drop"}
    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters=params,
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest(mutates_state=True)

    result = await strategy.execute(intent, manifest, sandbox_pid=None)

    assert result == "mutated"
    mock_callable.assert_called_once_with(**params)

    # Check lock acquisition
    expected_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    expected_lock_key = f"lock:test_tool:{expected_hash}:test_event"

    # default max_execution_time_ms is 1000 from create_mock_manifest
    expected_ttl = 1000

    assert len(lock_manager.acquired_locks) == 1
    assert lock_manager.acquired_locks[0] == (expected_lock_key, expected_ttl)
    assert len(lock_manager.released_locks) == 1
    assert lock_manager.released_locks[0] == expected_lock_key


@given(st.text(alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1))
@pytest.mark.asyncio
async def test_native_execution_strategy_fuzzing_params(text_param: str) -> None:
    # A property-based test to ensure we don't crash on various string inputs,
    # assuming verify_ast_safety handles them correctly.
    # We must mock verify_ast_safety since random strings aren't valid Python code usually
    # and would otherwise fail.
    mock_callable = AsyncMock(return_value="success")
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"fuzz": text_param},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    # If the fuzz string is valid Python AST, it might get passed to verify_ast_safety.
    # To be safe, we just execute and ensure it doesn't crash on standard non-code strings.
    # If it does happen to be valid syntax that verify_ast_safety rejects, it'll raise.
    # To avoid flakiness, we patch verify_ast_safety to return True.
    with pytest.MonkeyPatch.context() as m:
        m.setattr("coreason_actuator.strategies.verify_ast_safety", MagicMock(return_value=True))
        result = await strategy.execute(intent, manifest, sandbox_pid=None)
        assert result == "success"
        mock_callable.assert_called_once_with(fuzz=text_param)


@pytest.mark.asyncio
async def test_native_execution_strategy_ast_safety_allows_plain_strings() -> None:
    mock_callable = AsyncMock(return_value="success")
    registry = MockRegistry({"test_tool": mock_callable})
    lock_manager = MockLockManager()
    strategy = NativeExecutionStrategy(registry, lock_manager)

    plain_string_payload = "This is just a regular string, not code."

    intent = ToolInvocationEvent(
        event_id="test_event",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"description": plain_string_payload},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid=None)
    assert result == "success"
    mock_callable.assert_called_once_with(description=plain_string_payload)


@pytest.mark.asyncio
async def test_mcp_client_strategy_success() -> None:
    server_manifest = MCPServerManifest(
        server_id="mock_id",
        transport=StdioTransportProfile(command="mock"),
        binary_hash="a" * 64,
        capability_whitelist=MCPCapabilityWhitelistPolicy(
            allowed_tools=["test_tool"], allowed_prompts=[], allowed_resources=[]
        ),
        attestation_receipt=create_mock_credential_receipt(),
    )
    registry = MockMCPServerRegistry({"test_tool": server_manifest})
    transport = MockMCPTransport()

    strategy = MCPClientStrategy(registry, transport)
    intent = ToolInvocationEvent(
        event_id="test_event_id_123",
        timestamp=1704067200.0,
        tool_name="test_tool",
        parameters={"arg1": "value1"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid="mock_pid")

    assert result == "mock_response"
    assert len(transport.dispatched_packets) == 1

    dispatched_server_manifest, dispatched_packet = transport.dispatched_packets[0]
    assert dispatched_server_manifest == server_manifest
    assert dispatched_packet == {
        "jsonrpc": "2.0",
        "id": "test_event_id_123",
        "method": "test_tool",
        "params": {"arg1": "value1"},
    }


@pytest.mark.asyncio
async def test_mcp_client_strategy_missing_tool() -> None:
    registry = MockMCPServerRegistry({})
    transport = MockMCPTransport()

    strategy = MCPClientStrategy(registry, transport)
    intent = ToolInvocationEvent(
        event_id="test_event_id_123",
        timestamp=1704067200.0,
        tool_name="missing_tool",
        parameters={"arg1": "value1"},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(ValueError, match="MCPServerManifest not found for tool: missing_tool"):
        await strategy.execute(intent, manifest, sandbox_pid="mock_pid")

    assert len(transport.dispatched_packets) == 0


@pytest.mark.asyncio
async def test_mcp_client_strategy_none_parameters() -> None:
    server_manifest = MCPServerManifest(
        server_id="mock_id",
        transport=StdioTransportProfile(command="mock"),
        binary_hash="a" * 64,
        capability_whitelist=MCPCapabilityWhitelistPolicy(
            allowed_tools=["test_tool_no_params"], allowed_prompts=[], allowed_resources=[]
        ),
        attestation_receipt=create_mock_credential_receipt(),
    )
    registry = MockMCPServerRegistry({"test_tool_no_params": server_manifest})
    transport = MockMCPTransport()

    strategy = MCPClientStrategy(registry, transport)
    intent = ToolInvocationEvent(
        event_id="test_event_id_456",
        timestamp=1704067200.0,
        tool_name="test_tool_no_params",
        parameters={},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid="mock_pid")

    assert result == "mock_response"
    assert len(transport.dispatched_packets) == 1

    dispatched_server_manifest, dispatched_packet = transport.dispatched_packets[0]
    assert dispatched_server_manifest == server_manifest
    assert dispatched_packet == {
        "jsonrpc": "2.0",
        "id": "test_event_id_456",
        "method": "test_tool_no_params",
        "params": {},
    }


class MockTensorStorageProtocol:
    def __init__(self) -> None:
        self.streamed_chunks: list[bytes] = []

    async def stream_to_storage(self, data_stream: AsyncIterator[bytes]) -> str:
        async for chunk in data_stream:
            self.streamed_chunks.append(chunk)
        return "ipfs://mock_screenshot_cid_123"


class MockKinematicBrowser:
    def __init__(self, should_fail_verification: bool = False) -> None:
        self.clicked_coords: list[tuple[float, float, str, int]] = []
        self.typed_texts: list[tuple[float, float, str, str, int]] = []
        self.screenshots: list[bytes] = []
        self.should_fail_verification = should_fail_verification

    async def click(self, x: float, y: float, expected_visual_concept: str, timeout: int = 100) -> Any:
        if self.should_fail_verification:
            raise RuntimeError(f"Visual verification failed: Expected concept '{expected_visual_concept}' not found")
        self.clicked_coords.append((x, y, expected_visual_concept, timeout))
        return "click_success"

    async def type_text(self, x: float, y: float, text: str, expected_visual_concept: str, timeout: int = 100) -> Any:
        if self.should_fail_verification:
            raise RuntimeError(f"Visual verification failed: Expected concept '{expected_visual_concept}' not found")
        self.typed_texts.append((x, y, text, expected_visual_concept, timeout))
        return "type_success"

    async def get_current_url(self) -> str:
        return "https://example.com/test"

    async def get_viewport_size(self) -> tuple[int, int]:
        return (1920, 1080)

    async def get_dom_hash(self) -> str:
        return "a" * 64

    async def capture_viewport_screenshot(self) -> bytes:
        return b"mock_image_bytes"


@pytest.mark.asyncio
async def test_kinematic_strategy_click_success() -> None:
    browser = MockKinematicBrowser()
    tensor_storage = MockTensorStorageProtocol()
    strategy = KinematicExecutionStrategy(browser, tensor_storage)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="click_tool",
        parameters={
            "x": 100.0,
            "y": 200.0,
            "expected_visual_concept": "Submit Button",
            "action": "click",
            "timeout": 200,
        },
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid="mock_pid")

    assert result.type == "browser"
    assert result.current_url == "https://example.com/test"
    assert result.viewport_size == (1920, 1080)
    assert result.dom_hash == "a" * 64
    assert result.accessibility_tree_hash == "a" * 64
    assert result.screenshot_cid == "ipfs://mock_screenshot_cid_123"

    assert len(browser.clicked_coords) == 1
    assert browser.clicked_coords[0] == (100.0, 200.0, "Submit Button", 200)
    assert len(tensor_storage.streamed_chunks) == 1
    assert tensor_storage.streamed_chunks[0] == b"mock_image_bytes"


@pytest.mark.asyncio
async def test_kinematic_strategy_type_text_success() -> None:
    browser = MockKinematicBrowser()
    tensor_storage = MockTensorStorageProtocol()
    strategy = KinematicExecutionStrategy(browser, tensor_storage)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="type_tool",
        parameters={
            "x": 100.0,
            "y": 200.0,
            "expected_visual_concept": "Username Field",
            "action": "type_text",
            "text": "testuser",
        },
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    result = await strategy.execute(intent, manifest, sandbox_pid="mock_pid")

    assert result.type == "browser"
    assert result.current_url == "https://example.com/test"
    assert result.viewport_size == (1920, 1080)
    assert result.dom_hash == "a" * 64
    assert result.accessibility_tree_hash == "a" * 64
    assert result.screenshot_cid == "ipfs://mock_screenshot_cid_123"

    assert len(browser.typed_texts) == 1
    assert browser.typed_texts[0] == (100.0, 200.0, "testuser", "Username Field", 100)
    assert len(tensor_storage.streamed_chunks) == 1
    assert tensor_storage.streamed_chunks[0] == b"mock_image_bytes"


@pytest.mark.asyncio
async def test_kinematic_strategy_verification_failure() -> None:
    browser = MockKinematicBrowser(should_fail_verification=True)
    strategy = KinematicExecutionStrategy(browser)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="click_tool",
        parameters={
            "x": 100.0,
            "y": 200.0,
            "expected_visual_concept": "Submit Button",
            "action": "click",
        },
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(RuntimeError, match="Visual verification failed: Expected concept 'Submit Button' not found"):
        await strategy.execute(intent, manifest, sandbox_pid="mock_pid")

    assert len(browser.clicked_coords) == 0


@pytest.mark.asyncio
async def test_kinematic_strategy_missing_params() -> None:
    browser = MockKinematicBrowser()
    strategy = KinematicExecutionStrategy(browser)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="click_tool",
        parameters={"x": 100.0, "y": 200.0},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(
        ValueError,
        match=r"Kinematic interaction requires 'x', 'y', 'expected_visual_concept', and 'action' parameters\.",
    ):
        await strategy.execute(intent, manifest, sandbox_pid="mock_pid")


@pytest.mark.asyncio
async def test_kinematic_strategy_invalid_coords() -> None:
    browser = MockKinematicBrowser()
    strategy = KinematicExecutionStrategy(browser)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="click_tool",
        parameters={
            "x": "invalid",
            "y": 200.0,
            "expected_visual_concept": "Submit Button",
            "action": "click",
        },
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(ValueError, match=r"Coordinates 'x', 'y' and 'timeout' must be valid numbers\."):
        await strategy.execute(intent, manifest, sandbox_pid="mock_pid")


@pytest.mark.asyncio
async def test_kinematic_strategy_missing_text_param() -> None:
    browser = MockKinematicBrowser()
    strategy = KinematicExecutionStrategy(browser)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="type_tool",
        parameters={
            "x": 100.0,
            "y": 200.0,
            "expected_visual_concept": "Username Field",
            "action": "type_text",
        },
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(ValueError, match=r"Kinematic interaction 'type_text' requires a 'text' parameter\."):
        await strategy.execute(intent, manifest, sandbox_pid="mock_pid")


@pytest.mark.asyncio
async def test_kinematic_strategy_unsupported_action() -> None:
    browser = MockKinematicBrowser()
    strategy = KinematicExecutionStrategy(browser)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="unsupported_tool",
        parameters={
            "x": 100.0,
            "y": 200.0,
            "expected_visual_concept": "Username Field",
            "action": "drag_and_drop",
        },
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )
    manifest = create_mock_manifest()

    with pytest.raises(ValueError, match="Unsupported kinematic action: drag_and_drop"):
        await strategy.execute(intent, manifest, sandbox_pid="mock_pid")


class MockExecutionStrategy:
    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        _ = (intent, manifest, sandbox_pid)
        return "sync_result"


class MockIPCBroker:
    def __init__(self) -> None:
        self.pushed_messages: list[dict[str, Any]] = []

    async def push(self, message: dict[str, Any]) -> None:
        self.pushed_messages.append(message)

    async def pull(self) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_background_polling_sync() -> None:
    from coreason_manifest.spec.ontology import ExecutionSLA

    from coreason_actuator.strategies import BackgroundPollingStrategy

    mock_strategy = MockExecutionStrategy()
    mock_broker = MockIPCBroker()
    polling_strategy = BackgroundPollingStrategy(mock_strategy, mock_broker)

    # recreate manifest because it's frozen
    manifest_data = create_mock_manifest().model_dump()
    manifest_data["sla"] = ExecutionSLA(max_execution_time_ms=29999, max_compute_footprint_mb=100)
    manifest = ToolManifest.model_validate(manifest_data)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="sync_tool",
        parameters={},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    result = await polling_strategy.execute(intent, manifest, "mock_pid")
    assert result == "sync_result"
    assert len(mock_broker.pushed_messages) == 0


@pytest.mark.asyncio
async def test_background_polling_sync_no_sla() -> None:
    from coreason_actuator.strategies import BackgroundPollingStrategy

    mock_strategy = MockExecutionStrategy()
    mock_broker = MockIPCBroker()
    polling_strategy = BackgroundPollingStrategy(mock_strategy, mock_broker)

    manifest_data = create_mock_manifest().model_dump()
    manifest_data["sla"] = None
    manifest = ToolManifest.model_validate(manifest_data)

    intent = ToolInvocationEvent(
        event_id="test_event_id",
        timestamp=1704067200.0,
        tool_name="sync_tool",
        parameters={},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    result = await polling_strategy.execute(intent, manifest, "mock_pid")
    assert result == "sync_result"
    assert len(mock_broker.pushed_messages) == 0


class MockExecutionStrategySlow:
    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        _ = (intent, manifest, sandbox_pid)
        await asyncio.sleep(0.01)
        return "async_result"


class MockExecutionStrategyCrash:
    async def execute(self, intent: ToolInvocationEvent, manifest: ToolManifest, sandbox_pid: Any) -> Any:
        _ = (intent, manifest, sandbox_pid)
        await asyncio.sleep(0.01)
        raise RuntimeError("Something went wrong!")


@pytest.mark.asyncio
async def test_background_polling_async() -> None:
    from coreason_manifest.spec.ontology import ExecutionSLA

    from coreason_actuator.strategies import BackgroundPollingStrategy

    mock_strategy = MockExecutionStrategySlow()
    mock_broker = MockIPCBroker()
    polling_strategy = BackgroundPollingStrategy(mock_strategy, mock_broker)

    manifest_data = create_mock_manifest().model_dump()
    manifest_data["sla"] = ExecutionSLA(max_execution_time_ms=30000, max_compute_footprint_mb=100)
    manifest = ToolManifest.model_validate(manifest_data)

    intent = ToolInvocationEvent(
        event_id="test_event_id_async",
        timestamp=1704067200.0,
        tool_name="async_tool",
        parameters={},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    import asyncio
    from unittest.mock import patch

    # We patch run_in_executor to avoid spawning a real process and pickling mocks
    async def mock_run_in_executor(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        # Return whatever the strategy returns
        return await mock_strategy.execute(intent, manifest, "mock_pid")

    with patch.object(asyncio.get_running_loop(), "run_in_executor", new=mock_run_in_executor):
        result = await polling_strategy.execute(intent, manifest, "mock_pid")

        assert result.type == "observation"
        assert result.payload["status"] == "pending_async_execution"
        assert "job_id" in result.payload
        assert result.triggering_invocation_id == "test_event_id_async"

        if hasattr(polling_strategy, "_background_tasks") and polling_strategy._background_tasks:
            await next(iter(polling_strategy._background_tasks))
        await asyncio.sleep(0.01)

        assert len(mock_broker.pushed_messages) == 1
    pushed_msg = mock_broker.pushed_messages[0]
    assert pushed_msg["type"] == "observation"
    assert pushed_msg["payload"]["status"] == "completed"
    assert pushed_msg["payload"]["job_id"] == result.payload["job_id"]
    assert pushed_msg["payload"]["result"] == "async_result"
    assert pushed_msg["triggering_invocation_id"] == "test_event_id_async"


@pytest.mark.asyncio
async def test_background_polling_async_crash() -> None:
    from coreason_manifest.spec.ontology import ExecutionSLA

    from coreason_actuator.strategies import BackgroundPollingStrategy

    mock_strategy = MockExecutionStrategyCrash()
    mock_broker = MockIPCBroker()
    polling_strategy = BackgroundPollingStrategy(mock_strategy, mock_broker)

    manifest_data = create_mock_manifest().model_dump()
    manifest_data["sla"] = ExecutionSLA(max_execution_time_ms=30000, max_compute_footprint_mb=100)
    manifest = ToolManifest.model_validate(manifest_data)

    intent = ToolInvocationEvent(
        event_id="test_event_id_crash",
        timestamp=1704067200.0,
        tool_name="async_tool_crash",
        parameters={},
        zk_proof=create_mock_zk_proof(),
        agent_attestation=create_mock_attestation(),
    )

    import asyncio
    from unittest.mock import patch

    async def mock_run_in_executor(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        return await mock_strategy.execute(intent, manifest, "mock_pid")

    with patch.object(asyncio.get_running_loop(), "run_in_executor", new=mock_run_in_executor):
        result = await polling_strategy.execute(intent, manifest, "mock_pid")

        assert result.type == "observation"
        assert result.payload["status"] == "pending_async_execution"

        if hasattr(polling_strategy, "_background_tasks") and polling_strategy._background_tasks:
            await next(iter(polling_strategy._background_tasks))
        await asyncio.sleep(0.01)

        assert len(mock_broker.pushed_messages) == 1
    pushed_msg = mock_broker.pushed_messages[0]
    assert pushed_msg["type"] == "observation"
    assert pushed_msg["payload"]["status"] == "fatal_crash"
    # # assert pushed_msg["payload"]["error_type"] == "RuntimeError"
    assert pushed_msg["payload"]["error_message"] == "Something went wrong!"
    assert pushed_msg["triggering_invocation_id"] == "test_event_id_crash"


class SyncProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "SyncProcessPoolExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future
