from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import DataValidationError


def test_valid_minimal_record_normalizes_timestamp_to_utc():
    record = TelemetryRecord(
        timestamp=datetime(2026, 10, 25, 12, 0, tzinfo=timezone(timedelta(hours=2))),
        source_type="aws_cloudwatch",
        metric_name="CPUUtilization",
        metric_value=42,
    )
    assert record.timestamp == datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
    assert record.metric_value == 42.0
    assert record.dimensions == {}
    assert record.is_anomaly is None


def test_valid_aws_record_preserves_metadata_and_canonical_identity():
    dimensions = {"InstanceId": "i-123", "Environment": "test"}
    record = TelemetryRecord(
        timestamp=datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc),
        source_type="aws_cloudwatch",
        metric_name="CPUUtilization",
        metric_value=np.float64(42.5),
        metric_unit="Percent",
        entity_id="i-123",
        is_anomaly=True,
        namespace="AWS/EC2",
        dimensions=dimensions,
        resource_arn="arn:aws:ec2:region:123456789012:instance/i-123",
        account_id="123456789012",
        region="us-east-1",
        period=300,
        statistic="Average",
    )
    dimensions["AddedLater"] = "must not appear"
    assert record.dimensions["InstanceId"] == "i-123"
    assert "AddedLater" not in record.dimensions
    assert record.canonical_identity[4] == (("Environment", "test"), ("InstanceId", "i-123"))
    assert record.canonical_identity[7] == record.timestamp


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 10, 25, 12, 0), "2026-10-25T12:00:00Z", None],
)
def test_invalid_or_naive_timestamps_are_rejected(timestamp):
    with pytest.raises(DataValidationError):
        TelemetryRecord(timestamp, "aws_cloudwatch", "CPUUtilization", 1.0)


@pytest.mark.parametrize("field", ["source_type", "metric_name"])
def test_required_text_fields_must_not_be_empty(field):
    values = {"timestamp": datetime.now(timezone.utc), "source_type": "aws_cloudwatch", "metric_name": "CPU", "metric_value": 1.0}
    values[field] = "  "
    with pytest.raises(DataValidationError):
        TelemetryRecord(**values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, False, "1.0"])
def test_metric_value_must_be_finite_numeric_and_not_boolean(value):
    with pytest.raises(DataValidationError):
        TelemetryRecord(datetime.now(timezone.utc), "aws_cloudwatch", "CPU", value)


def test_boolean_ground_truth_is_optional_but_strict():
    base = {"timestamp": datetime.now(timezone.utc), "source_type": "aws_cloudwatch", "metric_name": "CPU", "metric_value": 1.0}
    assert TelemetryRecord(**base, is_anomaly=False).is_anomaly is False
    assert TelemetryRecord(**base, is_anomaly=True).is_anomaly is True
    assert TelemetryRecord(**base, is_anomaly=None).is_anomaly is None
    for invalid in ("False", "True", 0, 1, np.bool_(True)):
        with pytest.raises(DataValidationError):
            TelemetryRecord(**base, is_anomaly=invalid)


@pytest.mark.parametrize(
    "dimensions",
    [None, {"": "value"}, {"key": ""}, {1: "value"}, {"key": 1}, [("key", "value")]],
)
def test_dimensions_must_be_string_mapping(dimensions):
    with pytest.raises(DataValidationError):
        TelemetryRecord(datetime.now(timezone.utc), "aws_cloudwatch", "CPU", 1.0, dimensions=dimensions)


def test_period_must_be_positive_integer_and_optional_metadata_must_be_nonempty():
    base = {"timestamp": datetime.now(timezone.utc), "source_type": "aws_cloudwatch", "metric_name": "CPU", "metric_value": 1.0}
    for period in (0, -1, 1.5, True):
        with pytest.raises(DataValidationError):
            TelemetryRecord(**base, period=period)
    with pytest.raises(DataValidationError):
        TelemetryRecord(**base, namespace=" ")


def _identity_record(**overrides):
    values = {
        "timestamp": datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
        "source_type": "aws_cloudwatch",
        "metric_name": "CPUUtilization",
        "metric_value": 42.0,
        "metric_unit": "Percent",
        "entity_id": "instance-1",
        "namespace": "AWS/EC2",
        "dimensions": {"Environment": "test", "InstanceId": "i-123"},
        "account_id": "123456789012",
        "region": "us-east-1",
        "period": 300,
        "statistic": "Average",
    }
    values.update(overrides)
    return TelemetryRecord(**values)


def test_time_series_identity_is_dimension_order_independent():
    first = _identity_record(dimensions={"A": "1", "B": "2"})
    second = _identity_record(dimensions={"B": "2", "A": "1"})
    assert first.time_series_identity == second.time_series_identity


@pytest.mark.parametrize(
    "field_values",
    [
        {"source_type": "other_source"},
        {"metric_unit": "Count"},
        {"account_id": "other-account"},
        {"region": "us-west-2"},
    ],
)
def test_time_series_identity_distinguishes_series_metadata(field_values):
    assert _identity_record().time_series_identity != _identity_record(**field_values).time_series_identity


def test_resource_arn_takes_precedence_over_entity_id():
    record = _identity_record(resource_arn="arn:resource", entity_id="entity-1")
    assert record.time_series_identity[6] == "arn:resource"
    assert _identity_record(resource_arn="arn:other", entity_id="entity-1").time_series_identity != record.time_series_identity


def test_entity_id_is_used_when_resource_arn_is_absent():
    record = _identity_record(resource_arn=None, entity_id="entity-1")
    assert record.time_series_identity[6] == "entity-1"


def test_missing_resource_identifiers_use_none():
    record = _identity_record(resource_arn=None, entity_id=None)
    assert record.time_series_identity[6] is None
    assert record.resource_arn is None
    assert record.entity_id is None


def test_datapoint_identity_distinguishes_timestamp_period_and_statistic():
    base = _identity_record()
    assert base.datapoint_identity != _identity_record(timestamp=base.timestamp.replace(hour=13)).datapoint_identity
    assert base.datapoint_identity != _identity_record(period=60).datapoint_identity
    assert base.datapoint_identity != _identity_record(statistic="Maximum").datapoint_identity


def test_datapoint_identity_excludes_value_and_ground_truth_label():
    base = _identity_record(is_anomaly=False)
    changed = _identity_record(metric_value=999.0, is_anomaly=True)
    assert base.datapoint_identity == changed.datapoint_identity


def test_canonical_identity_remains_backward_compatible():
    record = _identity_record()
    assert record.canonical_identity == (
        record.account_id,
        record.region,
        record.namespace,
        record.metric_name,
        (("Environment", "test"), ("InstanceId", "i-123")),
        record.resource_arn,
        record.entity_id,
        record.timestamp,
        record.period,
        record.statistic,
    )
