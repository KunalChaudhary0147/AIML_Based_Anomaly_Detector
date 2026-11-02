from datetime import datetime, timedelta, timezone

import pytest

from anomaly_detector.data_ingestion.cloudwatch_adapter import (
    CLOUDWATCH_SOURCE_TYPE,
    cloudwatch_response_to_records,
)
from anomaly_detector.utils.exceptions import DataValidationError


def metadata(metric_name="CPUUtilization", **overrides):
    value = {
        "namespace": "AWS/EC2",
        "metric_name": metric_name,
        "dimensions": {"InstanceId": "i-123"},
        "metric_unit": "Percent",
        "account_id": "123456789012",
        "region": "us-east-1",
        "entity_id": "i-123",
        "period": 300,
        "statistic": "Average",
    }
    value.update(overrides)
    return value


def result(result_id="cpu", timestamps=None, values=None, **overrides):
    value = {
        "Id": result_id,
        "Timestamps": [datetime(2026, 10, 25, 12, tzinfo=timezone.utc)] if timestamps is None else timestamps,
        "Values": [42.0] if values is None else values,
        "StatusCode": "Complete",
    }
    value.update(overrides)
    return value


def test_one_result_with_multiple_pairs_preserves_metadata_and_label():
    timestamps = [
        datetime(2026, 10, 25, 12, tzinfo=timezone(timedelta(hours=2))),
        datetime(2026, 10, 25, 13, tzinfo=timezone.utc),
    ]
    records = cloudwatch_response_to_records(
        {"MetricDataResults": [result(timestamps=timestamps, values=[42.0, 43.5])]},
        {"cpu": metadata()},
    )
    assert len(records) == 2
    assert records[0].source_type == CLOUDWATCH_SOURCE_TYPE
    assert records[0].timestamp == datetime(2026, 10, 25, 10, tzinfo=timezone.utc)
    assert records[1].metric_value == 43.5
    assert records[0].namespace == "AWS/EC2"
    assert dict(records[0].dimensions) == {"InstanceId": "i-123"}
    assert records[0].is_anomaly is None


def test_numeric_unix_timestamps_match_the_documented_response_shape():
    records = cloudwatch_response_to_records(
        {"MetricDataResults": [result(timestamps=[1792929600], values=[42.0])]},
        {"cpu": metadata()},
    )
    assert records[0].timestamp == datetime(2026, 10, 25, 12, tzinfo=timezone.utc)


def test_multiple_results_use_exact_id_to_metadata_mapping():
    response = {"MetricDataResults": [result("cpu", values=[1.0]), result("latency", values=[250.0])]}
    records = cloudwatch_response_to_records(
        response,
        {"cpu": metadata("CPUUtilization"), "latency": metadata("Latency", metric_unit="Milliseconds")},
    )
    assert [record.metric_name for record in records] == ["CPUUtilization", "Latency"]
    assert records[1].metric_unit == "Milliseconds"


def test_empty_response_and_complete_empty_result_return_empty_list():
    assert cloudwatch_response_to_records({}, {}) == []
    assert cloudwatch_response_to_records({"MetricDataResults": []}, {}) == []
    assert cloudwatch_response_to_records(
        {"MetricDataResults": [result(timestamps=[], values=[])]},
        {"cpu": metadata()},
    ) == []


@pytest.mark.parametrize(
    "response",
    [
        {"MetricDataResults": [result(timestamps=[datetime.now(timezone.utc)], values=[])]},
        {"MetricDataResults": [result(timestamps=[datetime.now(timezone.utc)], values=[1.0, 2.0])]},
    ],
)
def test_mismatched_pairs_are_rejected(response):
    with pytest.raises(DataValidationError, match="length mismatch"):
        cloudwatch_response_to_records(response, {"cpu": metadata()})


def test_missing_mapping_is_rejected_without_guessing_from_id():
    with pytest.raises(DataValidationError, match="No metadata mapping"):
        cloudwatch_response_to_records({"MetricDataResults": [result("cpu")]}, {})


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"MetricDataResults": "not-a-list"},
        {"MetricDataResults": ["not-a-result"]},
        {"MetricDataResults": [{"Id": "cpu"}]},
    ],
)
def test_malformed_response_structures_are_rejected(response):
    with pytest.raises(DataValidationError):
        cloudwatch_response_to_records(response, {"cpu": metadata()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "42"])
def test_invalid_values_are_rejected(value):
    with pytest.raises(DataValidationError):
        cloudwatch_response_to_records(
            {"MetricDataResults": [result(values=[value])]},
            {"cpu": metadata()},
        )


def test_invalid_timestamps_are_rejected():
    with pytest.raises(DataValidationError):
        cloudwatch_response_to_records(
            {"MetricDataResults": [result(timestamps=[datetime(2026, 10, 25, 12)])]},
            {"cpu": metadata()},
        )


def test_partial_data_is_accepted_for_client_pagination():
    records = cloudwatch_response_to_records(
        {"MetricDataResults": [result(status_code="PartialData", values=[42.0])]},
        {"cpu": metadata()},
    )
    assert records[0].metric_value == 42.0


@pytest.mark.parametrize("status", ["InternalError", "Forbidden"])
def test_failed_status_messages_are_rejected(status):
    response = {
        "MetricDataResults": [
            result(status_code=status, Messages=[{"Code": "Failure", "Value": "not complete"}]),
        ],
    }
    response["MetricDataResults"][0]["StatusCode"] = status
    with pytest.raises(DataValidationError, match=status):
        cloudwatch_response_to_records(response, {"cpu": metadata()})


def test_input_response_and_metadata_are_not_mutated():
    response = {"MetricDataResults": [result()]}
    supplied_metadata = {"cpu": metadata()}
    response_before = {"MetricDataResults": [dict(response["MetricDataResults"][0])]}
    metadata_before = {"cpu": dict(supplied_metadata["cpu"])}
    cloudwatch_response_to_records(response, supplied_metadata)
    assert response == response_before
    assert supplied_metadata == metadata_before
