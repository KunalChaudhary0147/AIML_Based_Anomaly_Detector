from dataclasses import replace
from datetime import datetime, timezone

import pytest

from anomaly_detector.data_ingestion.cloudwatch_client import fetch_cloudwatch_records
from anomaly_detector.data_ingestion.telemetry_deduplication import deduplicate_telemetry_records
from anomaly_detector.utils.exceptions import (
    CloudWatchClientError,
    CloudWatchPaginationError,
    CloudWatchRetryError,
    DataValidationError,
)


class FakeCloudWatchClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get_metric_data(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def page(value, *, hour=12, next_token=None, include_token=False, metric_id="cpu"):
    result = {
        "Id": metric_id,
        "Timestamps": [datetime(2026, 10, 25, hour, tzinfo=timezone.utc)],
        "Values": [value],
        "StatusCode": "Complete",
    }
    response = {"MetricDataResults": [result]}
    if include_token:
        response["NextToken"] = next_token
    return response


def empty_page(*, next_token=None, include_token=False):
    response = {"MetricDataResults": []}
    if include_token:
        response["NextToken"] = next_token
    return response


def metadata():
    return {
        "cpu": {
            "namespace": "AWS/EC2",
            "metric_name": "CPUUtilization",
            "dimensions": {"InstanceId": "i-123"},
            "account_id": "123456789012",
            "region": "us-east-1",
            "period": 300,
            "statistic": "Average",
        },
    }


def collect_and_deduplicate(client, request, *, max_pages=100, max_attempts=3, sleeps=None):
    records = fetch_cloudwatch_records(
        client,
        request,
        metadata(),
        max_pages=max_pages,
        max_attempts=max_attempts,
        base_delay_seconds=1,
        jitter_seconds=0,
        sleep=(sleeps if sleeps is not None else []).append,
        random_value=lambda: 0,
    )
    return deduplicate_telemetry_records(records)


def test_successful_multi_page_ingestion_propagates_tokens_and_preserves_order():
    client = FakeCloudWatchClient([
        page(1.0, hour=12, next_token="page-2", include_token=True),
        page(2.0, hour=13),
    ])
    records = collect_and_deduplicate(client, {"MetricDataQueries": [{"Id": "cpu"}]})
    assert [record.metric_value for record in records] == [1.0, 2.0]
    assert client.calls == [
        {"MetricDataQueries": [{"Id": "cpu"}]},
        {"MetricDataQueries": [{"Id": "cpu"}], "NextToken": "page-2"},
    ]


def test_later_page_transient_failure_retries_without_losing_prior_page():
    sleeps = []
    client = FakeCloudWatchClient([
        page(1.0, next_token="page-2", include_token=True),
        TimeoutError("temporary"),
        page(2.0, hour=13),
    ])
    records = collect_and_deduplicate(client, {}, sleeps=sleeps)
    assert [record.metric_value for record in records] == [1.0, 2.0]
    assert len(client.calls) == 3
    assert client.calls[1] == client.calls[2] == {"NextToken": "page-2"}
    assert sleeps == [1.0]


def test_later_permanent_failure_returns_no_partial_result():
    client = FakeCloudWatchClient([
        page(1.0, next_token="page-2", include_token=True),
        PermissionError("denied"),
    ])
    with pytest.raises(CloudWatchClientError) as error:
        collect_and_deduplicate(client, {})
    assert isinstance(error.value.__cause__, PermissionError)
    assert len(client.calls) == 2


def test_later_malformed_page_returns_no_partial_result():
    client = FakeCloudWatchClient([
        page(1.0, next_token="page-2", include_token=True),
        {"MetricDataResults": "malformed"},
    ])
    with pytest.raises(DataValidationError):
        collect_and_deduplicate(client, {})
    assert len(client.calls) == 2


def test_repeated_token_is_rejected():
    client = FakeCloudWatchClient([
        empty_page(next_token="same", include_token=True),
        empty_page(next_token="same", include_token=True),
    ])
    with pytest.raises(CloudWatchPaginationError, match="Repeated"):
        collect_and_deduplicate(client, {})
    assert len(client.calls) == 2


def test_maximum_page_count_is_enforced():
    client = FakeCloudWatchClient([page(1.0, next_token="page-2", include_token=True)])
    with pytest.raises(CloudWatchPaginationError, match="max_pages"):
        collect_and_deduplicate(client, {}, max_pages=1)
    assert len(client.calls) == 1


def test_empty_pages_and_empty_collection_are_supported():
    empty_client = FakeCloudWatchClient([empty_page()])
    assert collect_and_deduplicate(empty_client, {}) == []

    followed_client = FakeCloudWatchClient([
        empty_page(next_token="page-2", include_token=True),
        page(2.0),
    ])
    records = collect_and_deduplicate(followed_client, {})
    assert [record.metric_value for record in records] == [2.0]


def test_duplicate_datapoint_across_pages_keeps_first_value_and_order():
    client = FakeCloudWatchClient([
        page(10.0, next_token="page-2", include_token=True),
        page(99.0),
    ])
    records = collect_and_deduplicate(client, {})
    assert len(records) == 1
    assert records[0].metric_value == 10.0


def test_duplicate_datapoint_with_different_label_keeps_first_record():
    client = FakeCloudWatchClient([page(10.0)])
    first = fetch_cloudwatch_records(client, {}, metadata())[0]
    later = replace(first, metric_value=99.0, is_anomaly=True)
    result = deduplicate_telemetry_records([first, later])
    assert len(result) == 1
    assert result[0] is first
    assert result[0].is_anomaly is None


def test_distinct_datapoints_are_preserved():
    client = FakeCloudWatchClient([
        page(10.0, hour=12, next_token="page-2", include_token=True),
        page(20.0, hour=13),
    ])
    records = collect_and_deduplicate(client, {})
    assert len(records) == 2
    assert records[0].datapoint_identity != records[1].datapoint_identity


def test_request_parameters_nested_objects_remain_unchanged():
    request = {"MetricDataQueries": [{"Id": "cpu", "MetricStat": {"Period": 300}}]}
    original = {"MetricDataQueries": [{"Id": "cpu", "MetricStat": {"Period": 300}}]}
    client = FakeCloudWatchClient([page(1.0)])
    collect_and_deduplicate(client, request)
    assert request == original


def test_repeated_collection_is_deterministic():
    outcomes = [page(1.0, next_token="page-2", include_token=True), page(2.0, hour=13)]
    first = collect_and_deduplicate(FakeCloudWatchClient(outcomes), {})
    second = collect_and_deduplicate(FakeCloudWatchClient(outcomes), {})
    assert first == second
    assert [record.metric_value for record in first] == [1.0, 2.0]


def test_page_limit_one_allows_one_terminal_page():
    client = FakeCloudWatchClient([page(1.0)])
    records = collect_and_deduplicate(client, {}, max_pages=1)
    assert len(records) == 1


def test_client_and_adapter_failures_remain_classified():
    retry_client = FakeCloudWatchClient([
        page(1.0, next_token="page-2", include_token=True),
        ConnectionError("unavailable"),
    ])
    with pytest.raises(CloudWatchRetryError):
        collect_and_deduplicate(retry_client, {}, max_attempts=1)

    malformed_client = FakeCloudWatchClient([
        page(1.0, next_token="page-2", include_token=True),
        {"MetricDataResults": [{"Id": "cpu", "Timestamps": [], "Values": [1.0]}]},
    ])
    with pytest.raises(DataValidationError):
        collect_and_deduplicate(malformed_client, {})
