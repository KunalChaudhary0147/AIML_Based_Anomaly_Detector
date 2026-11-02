from datetime import datetime, timezone

import pytest

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_deduplication import deduplicate_telemetry_records
from anomaly_detector.utils.exceptions import DataValidationError


def record(
    *,
    timestamp=datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
    metric_value=42.0,
    is_anomaly=None,
    period=300,
    statistic="Average",
    metric_name="CPUUtilization",
):
    return TelemetryRecord(
        timestamp=timestamp,
        source_type="aws_cloudwatch",
        metric_name=metric_name,
        metric_value=metric_value,
        is_anomaly=is_anomaly,
        namespace="AWS/EC2",
        dimensions={"InstanceId": "i-123"},
        account_id="123456789012",
        region="us-east-1",
        period=period,
        statistic=statistic,
    )


def test_empty_input_returns_new_empty_list():
    source = []
    result = deduplicate_telemetry_records(source)
    assert result == []
    assert result is not source


def test_no_duplicates_preserves_order_and_returns_new_list():
    first = record(metric_name="CPUUtilization")
    second = record(metric_name="NetworkIn")
    source = [first, second]
    result = deduplicate_telemetry_records(source)
    assert result == source
    assert result is not source
    assert result[0] is first
    assert result[1] is second


def test_duplicate_identical_values_keeps_first_record():
    first = record(metric_value=42.0)
    duplicate = record(metric_value=42.0)
    result = deduplicate_telemetry_records([first, duplicate])
    assert result == [first]
    assert result[0] is first


def test_duplicate_identity_with_different_value_keeps_first_record():
    first = record(metric_value=42.0)
    later = record(metric_value=99.0)
    result = deduplicate_telemetry_records([first, later])
    assert result == [first]
    assert result[0].metric_value == 42.0


def test_duplicate_identity_with_different_label_keeps_first_record():
    first = record(is_anomaly=False)
    later = record(is_anomaly=True)
    result = deduplicate_telemetry_records([first, later])
    assert result == [first]
    assert result[0].is_anomaly is False


def test_first_seen_order_is_preserved_across_duplicates():
    first = record(metric_name="First")
    second = record(metric_name="Second")
    duplicate_of_first = record(metric_name="First", metric_value=123.0)
    result = deduplicate_telemetry_records([first, second, duplicate_of_first])
    assert [item.metric_name for item in result] == ["First", "Second"]


@pytest.mark.parametrize(
    "changes",
    [
        {"timestamp": datetime(2026, 10, 25, 13, tzinfo=timezone.utc)},
        {"period": 60},
        {"statistic": "Maximum"},
        {"metric_name": "DifferentMetric"},
    ],
)
def test_different_datapoint_identity_fields_are_not_deduplicated(changes):
    first = record()
    second = record(**changes)
    result = deduplicate_telemetry_records([first, second])
    assert result == [first, second]


def test_repeated_calls_are_deterministic_and_input_is_unchanged():
    first = record(metric_value=1.0)
    duplicate = record(metric_value=2.0)
    third = record(timestamp=datetime(2026, 10, 25, 13, tzinfo=timezone.utc))
    source = [first, duplicate, third]
    original = list(source)
    result_one = deduplicate_telemetry_records(source)
    result_two = deduplicate_telemetry_records(source)
    assert result_one == result_two
    assert result_one[0] is first
    assert source == original


def test_invalid_input_is_rejected():
    with pytest.raises(DataValidationError):
        deduplicate_telemetry_records([record(), object()])
    with pytest.raises(DataValidationError):
        deduplicate_telemetry_records(None)
