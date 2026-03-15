# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

from typing import Literal, Protocol

from coreason_manifest.spec.ontology import EmbodiedSensoryVectorProfile

from coreason_actuator.utils.logger import logger


class SensoryTrackerProtocol(Protocol):
    """Protocol defining the interface for sensory triggers."""

    def ingest_delta(
        self,
        modality: Literal["video", "audio", "spatial_telemetry"],
        temporal_duration_ms: int,
        delta_score: float,
    ) -> EmbodiedSensoryVectorProfile | None:
        """
        Ingests a continuous physical delta.
        Returns an EmbodiedSensoryVectorProfile if the anomaly threshold is breached,
        otherwise None.
        """
        ...


class MultimodalSensoryTracker:
    """
    Tracks continuous physical deltas for Multimodal Sensory Ingestion.
    Computes the bayesian_surprise_score. It ONLY interrupts the Orchestrator
    by returning an EmbodiedSensoryVectorProfile when this score exceeds
    the defined anomaly threshold.
    """

    def __init__(self, anomaly_threshold: float) -> None:
        """
        Initializes the sensory tracker.

        Args:
            anomaly_threshold: The threshold above which a bayesian_surprise_score
                triggers an event.
        """
        if anomaly_threshold < 0.0:
            raise ValueError("Anomaly threshold must be non-negative.")
        self.anomaly_threshold = anomaly_threshold
        # In a real implementation, this might maintain a running average
        # or historical buffer to compute true KL divergence.
        # For this requirement, we treat the ingested delta_score directly
        # as or mapped to the bayesian_surprise_score.

    def ingest_delta(
        self,
        modality: Literal["video", "audio", "spatial_telemetry"],
        temporal_duration_ms: int,
        delta_score: float,
    ) -> EmbodiedSensoryVectorProfile | None:
        """
        Ingests a sensory delta.

        If the computed surprise score exceeds the anomaly threshold,
        generates and returns an EmbodiedSensoryVectorProfile.
        """
        if temporal_duration_ms <= 0:
            raise ValueError("Temporal duration must be greater than 0.")

        if delta_score < 0.0:
            raise ValueError("Delta score must be non-negative.")

        # For this implementation, the incoming delta score directly maps
        # to the Bayesian surprise score representing KL divergence.
        bayesian_surprise_score = delta_score

        if bayesian_surprise_score > self.anomaly_threshold:
            logger.info(
                f"Sensory anomaly detected in modality '{modality}': "
                f"score {bayesian_surprise_score} > threshold {self.anomaly_threshold}"
            )
            return EmbodiedSensoryVectorProfile(
                sensory_modality=modality,
                bayesian_surprise_score=bayesian_surprise_score,
                temporal_duration_ms=temporal_duration_ms,
                salience_threshold_breached=True,
            )

        # Threshold not breached, no event generated
        return None
