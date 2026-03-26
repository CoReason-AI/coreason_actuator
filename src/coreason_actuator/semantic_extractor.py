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
from typing import Any, Protocol

from coreason_actuator.utils.logger import logger


class SemanticExtractor:
    """Service for Lossless Semantic Extraction & Truncation."""

    def __init__(self, max_array_length: int = 50) -> None:
        self.max_array_length = max_array_length

    def truncate_payload(
        self, raw_payload: Any, max_nodes: int = 10000, max_depth: int = 10
    ) -> tuple[Any, dict[str, Any] | None]:
        """
        Applies Structural Semantic Truncation to large arrays within the payload.
        To prevent JSON-bomb memory exhaustion without violating strict Pydantic schema bounds,
        it slices the raw array and returns a strictly typed metadata dictionary to be appended
        as an out-of-band sibling key to the root ObservationEvent schema.
        """
        items_omitted = 0
        node_count = 0

        # We need to find arrays that exceed the limit.
        # Recursively search for large arrays within the payload to truncate them.
        def _truncate(node: Any, current_depth: int) -> Any:
            nonlocal items_omitted, node_count
            node_count += 1

            if current_depth >= max_depth:
                items_omitted += 1
                return "<TRUNCATED_MAX_DEPTH>"

            if node_count >= max_nodes:
                items_omitted += 1
                return "<TRUNCATED_MAX_NODES>"

            if isinstance(node, dict):
                truncated_dict: dict[str, Any] = {}
                for k, v in node.items():
                    if node_count >= max_nodes:
                        items_omitted += len(node) - len(truncated_dict)
                        # We just stop adding things since it's a dict. We can add a generic key if we want.
                        truncated_dict["<TRUNCATED>"] = "<TRUNCATED_MAX_NODES>"
                        break
                    truncated_dict[k] = _truncate(v, current_depth + 1)
                return truncated_dict
            if isinstance(node, list):
                truncated_list: list[Any] = []
                # First handle max_array_length
                if len(node) > self.max_array_length:
                    omitted = len(node) - self.max_array_length
                    items_omitted += omitted
                    nodes_to_process = node[: self.max_array_length]
                else:
                    nodes_to_process = node

                for item in nodes_to_process:
                    if node_count >= max_nodes:
                        items_omitted += len(nodes_to_process) - len(truncated_list)
                        truncated_list.append("<TRUNCATED_MAX_NODES>")
                        break
                    truncated_list.append(_truncate(item, current_depth + 1))
                return truncated_list
            return node

        # Handle deep copy automatically through recursive creation or explicitly if needed
        # Since _truncate creates new dicts and lists, it essentially acts as a deep copy
        # for mutatable structures.
        truncated_payload = _truncate(raw_payload, 0)

        truncation_metadata = None
        if items_omitted > 0:
            truncation_metadata = {
                "semantic_truncation_applied": True,
                "items_omitted": items_omitted,
            }
            logger.info(f"Applied semantic truncation, omitted {items_omitted} items.")

        return truncated_payload, truncation_metadata


class TensorStorageProtocol(Protocol):
    """Protocol for streaming massive binary blobs to cold storage."""

    async def stream_to_storage(self, data_stream: AsyncGenerator[bytes]) -> str:
        """
        Streams binary data to a pre-signed URI (e.g., S3).
        Returns the storage URI.
        """
        ...


class TensorRouter:
    """Service for Tensor & Binary Stream Routing."""

    def __init__(self, storage: TensorStorageProtocol) -> None:
        self.storage = storage

    async def route_tensor(
        self,
        data_stream: AsyncGenerator[bytes],
        shape: tuple[int, ...],  # noqa: ARG002
        vram_footprint_bytes: int,  # noqa: ARG002
        structural_type: str = "FLOAT32",  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Intercepts raw binary stream, computes its SHA-256 hash in chunks, streams the bytes
        directly to cold storage, and returns an NDimensionalTensorManifest reference.
        """
        hasher = hashlib.sha256()

        async def _hashing_generator() -> AsyncGenerator[bytes]:
            async for chunk in data_stream:
                hasher.update(chunk)
                yield chunk

        # Route to storage
        storage_uri = await self.storage.stream_to_storage(_hashing_generator())

        # Construct Manifest
        manifest: dict[str, Any] = {
            "type": "tensor_manifest",
            "storage_uri": storage_uri,
            "sha256_hash": hasher.hexdigest(),
        }

        logger.info(f"Routed massive tensor to {storage_uri} with root {manifest['sha256_hash']}")
        return manifest
