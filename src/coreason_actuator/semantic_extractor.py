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

from coreason_manifest.spec.ontology import NDimensionalTensorManifest, TensorStructuralFormatProfile

from coreason_actuator.utils.logger import logger


class SemanticExtractor:
    """Service for Lossless Semantic Extraction & Truncation."""

    def __init__(self, max_array_length: int = 50) -> None:
        self.max_array_length = max_array_length

    def truncate_payload(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """
        Applies Structural Semantic Truncation to large arrays within the payload.
        To prevent JSON-bomb memory exhaustion without violating strict Pydantic schema bounds,
        it slices the raw array and appends a strictly typed metadata dictionary as an out-of-band
        sibling key to the root ObservationEvent schema.
        """
        truncated_payload = raw_payload.copy()
        items_omitted = 0

        # We need to find arrays that exceed the limit.
        # Assuming the payload is a flat dictionary or we only truncate top-level arrays for now.
        for key, value in truncated_payload.items():
            if isinstance(value, list) and len(value) > self.max_array_length:
                omitted = len(value) - self.max_array_length
                items_omitted += omitted
                truncated_payload[key] = value[: self.max_array_length]

        if items_omitted > 0:
            truncated_payload["truncation_metadata"] = {
                "semantic_truncation_applied": True,
                "items_omitted": items_omitted,
            }
            logger.info(f"Applied semantic truncation, omitted {items_omitted} items.")

        return truncated_payload


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
        shape: tuple[int, ...],
        vram_footprint_bytes: int,
        structural_type: TensorStructuralFormatProfile = TensorStructuralFormatProfile.FLOAT32,
    ) -> NDimensionalTensorManifest:
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
        manifest = NDimensionalTensorManifest(
            structural_type=structural_type,
            shape=shape,
            vram_footprint_bytes=vram_footprint_bytes,
            merkle_root=hasher.hexdigest(),
            storage_uri=storage_uri,
        )

        logger.info(f"Routed massive tensor to {storage_uri} with root {manifest.merkle_root}")
        return manifest
