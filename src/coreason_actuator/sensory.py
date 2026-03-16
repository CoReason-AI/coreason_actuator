# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import math
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

    def __init__(self, anomaly_threshold: float, history_size: int = 10) -> None:
        """
        Initializes the sensory tracker.

        Args:
            anomaly_threshold: The threshold above which a bayesian_surprise_score
                triggers an event.
            history_size: Maximum size of the historical buffer per modality.
        """
        if anomaly_threshold < 0.0:
            raise ValueError("Anomaly threshold must be non-negative.")
        self.anomaly_threshold = anomaly_threshold
        self.history_size = history_size
        self._history: dict[str, list[float]] = {}

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

        if modality not in self._history:
            self._history[modality] = []

        history = self._history[modality]

        # Compute simple KL divergence proxy: incoming score vs running average
        if not history:
            bayesian_surprise_score = delta_score
        else:
            average_score = sum(history) / len(history)
            # Mathematical approximation of KL divergence (P || Q) for scalars:
            # P * log(P / Q) - P + Q
            p = delta_score + 1e-6  # Add epsilon to prevent log(0) or division by 0
            q = average_score + 1e-6
            bayesian_surprise_score = p * math.log(p / q) - p + q

        # Update historical buffer
        history.append(delta_score)
        if len(history) > self.history_size:
            history.pop(0)

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
