from datetime import datetime, timedelta, timezone

import pytest

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.detection.advanced_detector import (
    AdvancedDetectionResult,
    detect_multivariate_anomalies,
    detect_rolling_z_scores,
    detect_z_scores,
)
from anomaly_detector.evaluation_engine import BinaryEvaluation, evaluate_advanced_results
from anomaly_detector.utils.exceptions import DataValidationError


def record(value, *, metric="cpu", entity="host-a", minute=0, label=None):
    return TelemetryRecord(
        timestamp=datetime(2026, 11, 2, 12, minute, tzinfo=timezone.utc),
        source_type="csv", metric_name=metric, metric_value=value,
        entity_id=entity, is_anomaly=label,
    )


def test_z_score_normal_and_anomalous_values():
    records = [record(value) for value in [10, 10, 10, 10, 10, 100]]
    results = detect_z_scores(records, z_threshold=2.0, min_samples=3)
    assert results[-1].predicted_is_anomaly is True
    assert results[-1].detection_method == "z_score"
    assert results[-1].reference_value == pytest.approx(25.0)
    assert "exceeds" in results[-1].explanation


def test_z_score_insufficient_samples_and_zero_variance():
    insufficient = detect_z_scores([record(1), record(2)], min_samples=3)
    assert all(item.explanation == "insufficient_samples" for item in insufficient)
    constant = detect_z_scores([record(5), record(5), record(5)], min_samples=3)
    assert all(not item.predicted_is_anomaly for item in constant)
    assert all(item.statistical_score == 0 for item in constant)


def test_rolling_detection_uses_prior_window_and_entities():
    records = [record(value, minute=index, entity="a") for index, value in enumerate([10, 10, 10, 50])]
    records += [record(50, minute=0, entity="b")]
    results = detect_rolling_z_scores(records, window_size=3, z_threshold=2.0, min_samples=3)
    assert results[3].predicted_is_anomaly is True
    assert results[4].explanation == "insufficient_prior_samples"


def test_multivariate_detection_handles_missing_combinations_and_entities():
    records = [
        record(10, metric="cpu", minute=0), record(10, metric="memory", minute=0),
        record(11, metric="cpu", minute=1), record(11, metric="memory", minute=1),
        record(10, metric="cpu", minute=2), record(10, metric="memory", minute=2),
        record(100, metric="cpu", minute=3), record(100, metric="memory", minute=3),
        record(10, metric="cpu", minute=4, entity="other"),
    ]
    results = detect_multivariate_anomalies(records, ["cpu", "memory"], sensitivity=2.0, min_samples=3)
    assert results[6].predicted_is_anomaly is True
    assert results[8].explanation == "missing_metric_combination"


def test_invalid_advanced_configuration_is_rejected():
    with pytest.raises(DataValidationError):
        detect_z_scores([record(1), record(2)], z_threshold=0)
    with pytest.raises(DataValidationError):
        detect_rolling_z_scores([record(1), record(2)], window_size=0)
    with pytest.raises(DataValidationError):
        detect_multivariate_anomalies([record(1)], [])


def test_evaluation_metrics_and_missing_labels():
    records = [record(1, label=False), record(2, label=True), record(3, label=False), record(4, label=True)]
    results = [
        AdvancedDetectionResult(records[0], False, "normal", "test", 1),
        AdvancedDetectionResult(records[1], True, "anomaly", "test", 2),
        AdvancedDetectionResult(records[2], True, "false_positive", "test", 3),
        AdvancedDetectionResult(records[3], False, "false_negative", "test", 4),
    ]
    evaluation = evaluate_advanced_results(results)
    assert evaluation == BinaryEvaluation(1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5)
    assert evaluate_advanced_results([AdvancedDetectionResult(record(1), False, "unknown", "test", 1)]) is None
