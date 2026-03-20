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
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from coreason_manifest.spec.ontology import TensorStructuralFormatProfile

from coreason_actuator.semantic_extractor import SemanticExtractor, TensorRouter, TensorStorageProtocol


def test_semantic_extractor_no_truncation() -> None:
    extractor = SemanticExtractor(max_array_length=50)
    raw_payload = {"status": "success", "data": [1, 2, 3, 4, 5], "metadata": {"key": "value"}}

    truncated, metadata = extractor.truncate_payload(raw_payload)

    assert metadata is None
    assert len(truncated["data"]) == 5
    assert truncated["status"] == "success"


def test_semantic_extractor_with_truncation() -> None:
    extractor = SemanticExtractor(max_array_length=5)
    raw_payload = {"status": "success", "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "other_data": ["a", "b", "c"]}

    truncated, metadata = extractor.truncate_payload(raw_payload)

    assert metadata is not None
    assert metadata["semantic_truncation_applied"] is True
    assert metadata["items_omitted"] == 5
    assert len(truncated["data"]) == 5
    assert truncated["data"] == [1, 2, 3, 4, 5]
    assert len(truncated["other_data"]) == 3
    assert truncated["status"] == "success"


def test_semantic_extractor_multiple_arrays() -> None:
    extractor = SemanticExtractor(max_array_length=2)
    raw_payload = {"data1": [1, 2, 3, 4], "data2": ["a", "b", "c"]}

    truncated, metadata = extractor.truncate_payload(raw_payload)

    assert metadata is not None
    assert metadata["items_omitted"] == 3  # (4-2) + (3-2)
    assert len(truncated["data1"]) == 2
    assert len(truncated["data2"]) == 2


def test_semantic_extractor_root_array_truncation() -> None:
    extractor = SemanticExtractor(max_array_length=3)
    raw_payload = [1, 2, 3, 4, 5, 6]

    truncated, metadata = extractor.truncate_payload(raw_payload)

    assert metadata is not None
    assert metadata["items_omitted"] == 3
    assert len(truncated) == 3
    assert truncated == [1, 2, 3]


def test_semantic_extractor_depth_bomb() -> None:
    extractor = SemanticExtractor()
    payload: dict[str, Any] = {}
    current = payload
    for _ in range(15):
        current["a"] = {}
        current = current["a"]

    truncated, metadata = extractor.truncate_payload(payload, max_depth=10)

    assert metadata is not None
    assert metadata["items_omitted"] == 1

    # Traverse truncated payload
    curr_trunc = truncated
    depth_reached = 0
    while isinstance(curr_trunc, dict) and "a" in curr_trunc:
        curr_trunc = curr_trunc["a"]
        depth_reached += 1

    assert curr_trunc == "<TRUNCATED_MAX_DEPTH>"
    assert depth_reached == 10


def test_semantic_extractor_node_bomb() -> None:
    extractor = SemanticExtractor()
    # Create a payload with 50 nodes
    payload = [1] * 50

    truncated, metadata = extractor.truncate_payload(payload, max_nodes=20)

    assert metadata is not None
    assert metadata["items_omitted"] > 0
    assert len(truncated) == 20
    assert truncated[-1] == "<TRUNCATED_MAX_NODES>"


def test_semantic_extractor_dict_node_bomb() -> None:
    extractor = SemanticExtractor()
    # Create a dict payload with 50 elements
    payload = {f"k{i}": i for i in range(50)}

    truncated, metadata = extractor.truncate_payload(payload, max_nodes=20)

    assert metadata is not None
    assert metadata["items_omitted"] > 0
    assert "<TRUNCATED>" in truncated
    assert truncated["<TRUNCATED>"] == "<TRUNCATED_MAX_NODES>"


class MockTensorStorage(TensorStorageProtocol):
    def __init__(self) -> None:
        self.received_chunks: list[bytes] = []

    async def stream_to_storage(self, data_stream: AsyncGenerator[bytes]) -> str:
        async for chunk in data_stream:
            self.received_chunks.append(chunk)
        return "s3://mock-bucket/tensor-blob"


@pytest.mark.asyncio
async def test_tensor_router() -> None:
    storage = MockTensorStorage()
    router = TensorRouter(storage)

    async def mock_stream() -> AsyncGenerator[bytes]:
        yield b"chunk1"
        yield b"chunk2"

    manifest = await router.route_tensor(
        data_stream=mock_stream(),
        shape=(2, 2),
        vram_footprint_bytes=16,
        structural_type=TensorStructuralFormatProfile.FLOAT32,
    )

    expected_hash = hashlib.sha256(b"chunk1chunk2").hexdigest()

    assert manifest.merkle_root == expected_hash
    assert manifest.storage_uri == "s3://mock-bucket/tensor-blob"
    assert manifest.shape == (2, 2)
    assert manifest.structural_type == TensorStructuralFormatProfile.FLOAT32
    assert manifest.vram_footprint_bytes == 16

    assert len(storage.received_chunks) == 2
    assert storage.received_chunks[0] == b"chunk1"
    assert storage.received_chunks[1] == b"chunk2"
