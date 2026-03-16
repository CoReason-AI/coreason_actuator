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
from unittest.mock import patch

import pytest
from coreason_manifest.spec.ontology import EmbodiedSensoryVectorProfile

from coreason_actuator.sensory import MultimodalSensoryTracker


def test_sensory_tracker_initialization() -> None:
    tracker = MultimodalSensoryTracker(anomaly_threshold=0.5, history_size=20)
    assert tracker.anomaly_threshold == 0.5
    assert tracker.history_size == 20
    assert tracker._history == {}


def test_sensory_tracker_invalid_initialization() -> None:
    with pytest.raises(ValueError, match=r"Anomaly threshold must be non-negative\."):
        MultimodalSensoryTracker(anomaly_threshold=-1.0)


def test_sensory_tracker_threshold_not_breached() -> None:
    tracker = MultimodalSensoryTracker(anomaly_threshold=0.8)

    # First ingest has no history, uses delta_score directly
    result = tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=0.5)
    assert result is None

    # Exact threshold should not breach (strict inequality)
    result_exact = tracker.ingest_delta(modality="audio", temporal_duration_ms=100, delta_score=0.8)
    assert result_exact is None


def test_sensory_tracker_threshold_breached() -> None:
    tracker = MultimodalSensoryTracker(anomaly_threshold=0.5)

    # First ingest uses delta_score directly
    result = tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=0.9)
    assert result is not None
    assert isinstance(result, EmbodiedSensoryVectorProfile)
    assert result.sensory_modality == "video"
    assert result.bayesian_surprise_score == 0.9
    assert result.temporal_duration_ms == 100
    assert result.salience_threshold_breached is True


def test_sensory_tracker_kl_divergence() -> None:
    tracker = MultimodalSensoryTracker(anomaly_threshold=2.0, history_size=3)

    # Ingest baseline
    tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=0.1)
    tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=0.1)
    tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=0.1)

    # History is [0.1, 0.1, 0.1], average = 0.1
    # Now ingest a huge anomaly
    delta_score = 5.0

    # Calculate expected
    p = delta_score + 1e-6
    q = 0.1 + 1e-6
    expected_surprise = p * math.log(p / q) - p + q

    with patch("coreason_actuator.sensory.logger.info") as mock_logger:
        result = tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=delta_score)
        assert result is not None
        assert math.isclose(result.bayesian_surprise_score, expected_surprise)
        assert result.salience_threshold_breached is True
        mock_logger.assert_called_once()

    # Check history eviction
    assert tracker._history["video"] == [0.1, 0.1, 5.0]


def test_sensory_tracker_different_modalities() -> None:
    tracker = MultimodalSensoryTracker(anomaly_threshold=0.3)

    result_audio = tracker.ingest_delta(modality="audio", temporal_duration_ms=50, delta_score=0.4)
    assert result_audio is not None
    assert result_audio.sensory_modality == "audio"

    result_spatial = tracker.ingest_delta(modality="spatial_telemetry", temporal_duration_ms=200, delta_score=1.5)
    assert result_spatial is not None
    assert result_spatial.sensory_modality == "spatial_telemetry"


def test_sensory_tracker_invalid_inputs() -> None:
    tracker = MultimodalSensoryTracker(anomaly_threshold=0.5)

    # Zero temporal duration
    with pytest.raises(ValueError, match=r"Temporal duration must be greater than 0\."):
        tracker.ingest_delta(modality="video", temporal_duration_ms=0, delta_score=0.8)

    # Negative temporal duration
    with pytest.raises(ValueError, match=r"Temporal duration must be greater than 0\."):
        tracker.ingest_delta(modality="video", temporal_duration_ms=-10, delta_score=0.8)

    # Negative delta score
    with pytest.raises(ValueError, match=r"Delta score must be non-negative\."):
        tracker.ingest_delta(modality="video", temporal_duration_ms=100, delta_score=-0.1)
