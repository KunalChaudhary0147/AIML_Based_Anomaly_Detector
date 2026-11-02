from datetime import datetime, timezone

import pytest

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.detection.generic_detector import (
    GenericDetectionResult,
    NumericThreshold,
    detect_numeric_telemetry,
)
from anomaly_detector.utils.exceptions import DataValidationError


def telemetry(metric_name="CPUUtilization", metric_value=50.0, is_anomaly=None):
    return TelemetryRecord(
        timestamp=datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
        source_type="aws_cloudwatch",
        metric_name=metric_name,
        metric_value=metric_value,
        is_anomaly=is_anomaly,
        namespace="AWS/EC2",
        dimensions={"InstanceId": "i-unseen"},
        account_id="123",
        region="us-east-1",
    )


def test_normal_value_is_not_anomalous_and_keeps_record_association():
    record = telemetry(metric_value=50.0)
    result = detect_numeric_telemetry([record], {"CPUUtilization": NumericThreshold(0, 100)})
    assert result == [GenericDetectionResult(record, False, "within_configured_thresholds")]
    assert result[0].record is record


def test_upper_and_lower_violations_are_flagged():
    records = [telemetry(metric_value=101.0), telemetry(metric_value=-1.0)]
    results = detect_numeric_telemetry(records, {"CPUUtilization": NumericThreshold(0, 100)})
    assert [item.predicted_is_anomaly for item in results] == [True, True]
    assert results[0].explanation.startswith("above_upper_threshold")
    assert results[1].explanation.startswith("below_lower_threshold")


def test_unconfigured_metric_is_explicitly_marked():
    result = detect_numeric_telemetry([telemetry(metric_name="Latency", metric_value=999)], {})[0]
    assert result.predicted_is_anomaly is False
    assert result.explanation == "unconfigured_metric"


@pytest.mark.parametrize(
    "thresholds",
    [
        {"CPUUtilization": NumericThreshold(float("nan"), 100)},
        {"CPUUtilization": NumericThreshold(0, float("inf"))},
        {"CPUUtilization": NumericThreshold(100, 0)},
    ],
)
def test_invalid_thresholds_are_rejected(thresholds):
    with pytest.raises(DataValidationError):
        detect_numeric_telemetry([telemetry()], thresholds)


def test_unseen_observation_and_ground_truth_do_not_change_detection():
    false_label = telemetry(metric_value=75.0, is_anomaly=False)
    true_label = telemetry(metric_value=75.0, is_anomaly=True)
    results = detect_numeric_telemetry(
        [false_label, true_label],
        {"CPUUtilization": NumericThreshold(0, 100)},
    )
    assert [item.predicted_is_anomaly for item in results] == [False, False]
    assert [item.record for item in results] == [false_label, true_label]


def test_input_records_are_not_mutated():
    records = [telemetry(metric_value=10.0)]
    original = list(records)
    detect_numeric_telemetry(records, {"CPUUtilization": NumericThreshold(0, 100)})
    assert records == original


def test_invalid_record_input_is_rejected():
    with pytest.raises(DataValidationError):
        detect_numeric_telemetry([object()], {"CPUUtilization": NumericThreshold(0, 100)})
