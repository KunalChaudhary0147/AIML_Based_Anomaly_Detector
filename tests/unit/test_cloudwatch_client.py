from datetime import datetime, timezone

import pytest

from anomaly_detector.data_ingestion.cloudwatch_client import fetch_cloudwatch_records
from anomaly_detector.utils.exceptions import (
    CloudWatchClientError,
    CloudWatchDeadlineError,
    CloudWatchPaginationError,
    CloudWatchRecordLimitError,
    CloudWatchRetryError,
    DataValidationError,
)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get_metric_data(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


_MISSING = object()


def response(value=42.0, *, next_token=_MISSING, result_id="cpu", hour=12):
    result = {
        "Id": result_id,
        "Timestamps": [datetime(2026, 10, 25, hour, tzinfo=timezone.utc)],
        "Values": [value],
        "StatusCode": "Complete",
    }
    response_value = {"MetricDataResults": [result]}
    if next_token is not _MISSING:
        response_value["NextToken"] = next_token
    return response_value


def empty_response(*, next_token=_MISSING):
    response_value = {"MetricDataResults": []}
    if next_token is not _MISSING:
        response_value["NextToken"] = next_token
    return response_value


def metadata():
    return {"cpu": {"metric_name": "CPUUtilization", "namespace": "AWS/EC2"}}


def test_immediate_success_calls_client_once_and_returns_records():
    client = FakeClient([response()])
    records = fetch_cloudwatch_records(client, {"MetricDataQueries": []}, metadata(), sleep=lambda _: None)
    assert len(records) == 1
    assert client.calls == [{"MetricDataQueries": []}]


def test_transient_failure_retries_then_succeeds():
    client = FakeClient([ConnectionError("temporary"), response()])
    sleeps = []
    records = fetch_cloudwatch_records(
        client,
        {"NextToken": "token"},
        metadata(),
        max_attempts=3,
        base_delay_seconds=2,
        jitter_seconds=0,
        sleep=sleeps.append,
    )
    assert len(records) == 1
    assert len(client.calls) == 2
    assert sleeps == [2.0]


def test_retryable_failures_stop_at_attempt_limit_without_final_sleep():
    client = FakeClient([TimeoutError("temporary")] * 4)
    sleeps = []
    with pytest.raises(CloudWatchRetryError) as error:
        fetch_cloudwatch_records(
            client,
            {},
            metadata(),
            max_attempts=3,
            base_delay_seconds=1,
            jitter_seconds=0,
            sleep=sleeps.append,
        )
    assert len(client.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert isinstance(error.value.__cause__, TimeoutError)


class AuthError(Exception):
    response = {"Error": {"Code": "AccessDeniedException"}}


class ThrottlingError(Exception):
    response = {"Error": {"Code": "ThrottlingException"}}


def test_authentication_and_authorization_errors_are_not_retried():
    client = FakeClient([AuthError("denied")])
    with pytest.raises(CloudWatchClientError) as error:
        fetch_cloudwatch_records(client, {}, metadata(), sleep=lambda _: None)
    assert len(client.calls) == 1
    assert isinstance(error.value.__cause__, AuthError)


@pytest.mark.parametrize("code", ["ExpiredTokenException", "NotAuthorized", "ValidationError", "InvalidNextToken"])
def test_official_permanent_error_codes_are_not_retried(code):
    class OfficialError(Exception):
        response = {"Error": {"Code": code}}

    client = FakeClient([OfficialError(code)])
    with pytest.raises(CloudWatchClientError):
        fetch_cloudwatch_records(client, {}, metadata())
    assert len(client.calls) == 1


def test_official_internal_failure_code_is_retryable():
    class OfficialInternalFailure(Exception):
        response = {"Error": {"Code": "InternalFailure"}}

    client = FakeClient([OfficialInternalFailure("temporary"), response()])
    records = fetch_cloudwatch_records(client, {}, metadata(), sleep=lambda _: None)
    assert len(records) == 1
    assert len(client.calls) == 2


def test_validation_and_malformed_response_errors_are_not_retried():
    validation_client = FakeClient([DataValidationError("bad metadata")])
    with pytest.raises(DataValidationError):
        fetch_cloudwatch_records(validation_client, {}, metadata(), sleep=lambda _: None)
    assert len(validation_client.calls) == 1

    malformed_client = FakeClient([{"MetricDataResults": "not-a-list"}])
    with pytest.raises(DataValidationError):
        fetch_cloudwatch_records(malformed_client, {}, metadata(), sleep=lambda _: None)
    assert len(malformed_client.calls) == 1


def test_throttling_retries_with_bounded_exponential_jitter():
    client = FakeClient([ThrottlingError("slow down"), ThrottlingError("slow down"), response()])
    sleeps = []
    records = fetch_cloudwatch_records(
        client,
        {},
        metadata(),
        max_attempts=3,
        base_delay_seconds=1,
        jitter_seconds=0.5,
        sleep=sleeps.append,
        random_value=lambda: 0.5,
    )
    assert len(records) == 1
    assert sleeps == [1.25, 2.25]
    assert all(1 <= delay <= 2.5 for delay in sleeps)


def test_invalid_retry_configuration_is_rejected_before_client_call():
    client = FakeClient([response()])
    for kwargs in (
        {"max_attempts": 0},
        {"max_attempts": True},
        {"base_delay_seconds": -1},
        {"jitter_seconds": float("inf")},
    ):
        with pytest.raises(CloudWatchClientError):
            fetch_cloudwatch_records(client, {}, metadata(), **kwargs)
    assert client.calls == []


def test_random_value_outside_unit_interval_is_rejected():
    client = FakeClient([ConnectionError("temporary"), response()])
    with pytest.raises(CloudWatchClientError):
        fetch_cloudwatch_records(
            client,
            {},
            metadata(),
            max_attempts=2,
            random_value=lambda: 2.0,
            sleep=lambda _: None,
        )
    assert len(client.calls) == 1


def test_empty_successful_response_remains_empty():
    client = FakeClient([{"MetricDataResults": []}])
    assert fetch_cloudwatch_records(client, {}, metadata()) == []


def test_failed_adapter_response_is_not_returned():
    client = FakeClient([{
        "MetricDataResults": [{
            "Id": "cpu",
            "Timestamps": [datetime(2026, 10, 25, 12, tzinfo=timezone.utc)],
            "Values": [1.0],
            "StatusCode": "InternalError",
        }],
    }])
    with pytest.raises(DataValidationError):
        fetch_cloudwatch_records(client, {}, metadata())
    assert len(client.calls) == 1


def test_partial_page_with_next_token_continues_collection():
    client = FakeClient([
        {
            "MetricDataResults": [{
                "Id": "cpu",
                "Timestamps": [1792929600],
                "Values": [1.0],
                "StatusCode": "PartialData",
            }],
            "NextToken": "page-2",
        },
        response(2.0),
    ])
    records = fetch_cloudwatch_records(client, {}, metadata())
    assert [record.metric_value for record in records] == [1.0, 2.0]
    assert client.calls == [{}, {"NextToken": "page-2"}]


def test_internal_error_page_uses_next_token_without_emitting_failed_page():
    client = FakeClient([
        {
            "MetricDataResults": [{"Id": "cpu", "StatusCode": "InternalError"}],
            "NextToken": "page-2",
        },
        response(2.0),
    ])
    records = fetch_cloudwatch_records(client, {}, metadata())
    assert [record.metric_value for record in records] == [2.0]


def test_partial_data_without_next_token_is_rejected():
    client = FakeClient([{
        "MetricDataResults": [{
            "Id": "cpu",
            "Timestamps": [1792929600],
            "Values": [1.0],
            "StatusCode": "PartialData",
        }],
    }])
    with pytest.raises(DataValidationError, match="requires NextToken"):
        fetch_cloudwatch_records(client, {}, metadata())


def test_request_parameters_are_not_mutated():
    request = {"MetricDataQueries": [{"Id": "cpu"}]}
    original = {"MetricDataQueries": [{"Id": "cpu"}]}
    client = FakeClient([response()])
    fetch_cloudwatch_records(client, request, metadata())
    assert request == original


def test_nested_request_parameters_are_isolated_from_client_mutation():
    class MutatingClient:
        def get_metric_data(self, **kwargs):
            kwargs["MetricDataQueries"].append({"Id": "mutated"})
            return response()

    request = {"MetricDataQueries": [{"Id": "cpu"}]}
    fetch_cloudwatch_records(MutatingClient(), request, metadata())
    assert request == {"MetricDataQueries": [{"Id": "cpu"}]}


def test_single_page_without_token_is_compatible():
    client = FakeClient([response(1.0)])
    records = fetch_cloudwatch_records(client, {}, metadata(), max_pages=1)
    assert [record.metric_value for record in records] == [1.0]
    assert client.calls == [{}]


def test_multiple_pages_propagate_tokens_and_preserve_order():
    client = FakeClient([response(1.0, next_token="page-2", hour=12), response(2.0, hour=13)])
    records = fetch_cloudwatch_records(client, {"MetricDataQueries": []}, metadata())
    assert [record.metric_value for record in records] == [1.0, 2.0]
    assert client.calls == [{"MetricDataQueries": []}, {"MetricDataQueries": [], "NextToken": "page-2"}]


def test_empty_token_terminates_pagination():
    client = FakeClient([response(1.0, next_token="")])
    records = fetch_cloudwatch_records(client, {}, metadata())
    assert [record.metric_value for record in records] == [1.0]
    assert len(client.calls) == 1


def test_empty_page_can_be_followed_by_nonempty_page():
    client = FakeClient([empty_response(next_token="page-2"), response(2.0)])
    records = fetch_cloudwatch_records(client, {}, metadata())
    assert [record.metric_value for record in records] == [2.0]


def test_repeated_token_is_rejected_without_looping():
    client = FakeClient([empty_response(next_token="same"), empty_response(next_token="same")])
    with pytest.raises(CloudWatchPaginationError, match="Repeated"):
        fetch_cloudwatch_records(client, {}, metadata())
    assert len(client.calls) == 2


def test_maximum_page_count_is_enforced():
    client = FakeClient([response(1.0, next_token="page-2")])
    with pytest.raises(CloudWatchPaginationError, match="max_pages"):
        fetch_cloudwatch_records(client, {}, metadata(), max_pages=1)
    assert len(client.calls) == 1


def test_invalid_page_limits_are_rejected_before_client_call():
    client = FakeClient([response()])
    for limit in (0, -1, True, 1.5, "2"):
        with pytest.raises(CloudWatchPaginationError):
            fetch_cloudwatch_records(client, {}, metadata(), max_pages=limit)
    assert client.calls == []


def test_record_budget_allows_success_within_and_at_exact_limit():
    client = FakeClient([response(1.0, next_token="page-2"), response(2.0)])
    records = fetch_cloudwatch_records(client, {}, metadata(), max_records=2)
    assert [record.metric_value for record in records] == [1.0, 2.0]


def test_record_budget_is_enforced_across_pages_without_partial_results():
    client = FakeClient([response(1.0, next_token="page-2"), response(2.0)])
    with pytest.raises(CloudWatchRecordLimitError, match="max_records"):
        fetch_cloudwatch_records(client, {}, metadata(), max_records=1)
    assert len(client.calls) == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_record_budgets_are_rejected_before_client_call(limit):
    client = FakeClient([response()])
    with pytest.raises(CloudWatchRecordLimitError):
        fetch_cloudwatch_records(client, {}, metadata(), max_records=limit)
    assert client.calls == []


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def test_deadline_allows_success_within_budget():
    clock = FakeClock()
    client = FakeClient([response()])
    records = fetch_cloudwatch_records(client, {}, metadata(), collection_timeout_seconds=1, monotonic=clock)
    assert len(records) == 1


def test_deadline_exceeded_before_first_call():
    clock_values = iter([0.0, 2.0])
    client = FakeClient([response()])
    with pytest.raises(CloudWatchDeadlineError):
        fetch_cloudwatch_records(client, {}, metadata(), collection_timeout_seconds=1, monotonic=lambda: next(clock_values))
    assert client.calls == []


def test_deadline_exceeded_between_pages():
    clock_values = iter([0.0, 0.0, 0.0, 2.0])
    client = FakeClient([response(1.0, next_token="page-2"), response(2.0)])

    with pytest.raises(CloudWatchDeadlineError):
        fetch_cloudwatch_records(client, {}, metadata(), collection_timeout_seconds=1, monotonic=lambda: next(clock_values))
    assert len(client.calls) == 1


def test_deadline_exceeded_after_retry_wait():
    clock = FakeClock()
    client = FakeClient([ConnectionError("temporary"), response()])

    def sleep_and_advance(_delay):
        clock.value = 2

    with pytest.raises(CloudWatchDeadlineError):
        fetch_cloudwatch_records(
            client,
            {},
            metadata(),
            collection_timeout_seconds=1,
            monotonic=clock,
            sleep=sleep_and_advance,
        )
    assert len(client.calls) == 1


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), "1"])
def test_invalid_deadlines_are_rejected_before_client_call(timeout):
    client = FakeClient([response()])
    with pytest.raises(CloudWatchDeadlineError):
        fetch_cloudwatch_records(client, {}, metadata(), collection_timeout_seconds=timeout)
    assert client.calls == []


def test_later_page_failure_does_not_return_partial_results():
    client = FakeClient([response(1.0, next_token="page-2"), ConnectionError("failed")])
    with pytest.raises(CloudWatchRetryError):
        fetch_cloudwatch_records(client, {}, metadata(), max_attempts=1)


def test_later_malformed_page_does_not_return_partial_results():
    client = FakeClient([response(1.0, next_token="page-2"), {"MetricDataResults": "bad"}])
    with pytest.raises(DataValidationError):
        fetch_cloudwatch_records(client, {}, metadata())


def test_retries_apply_to_the_failing_page_request():
    client = FakeClient([response(1.0, next_token="page-2"), TimeoutError("temporary"), response(2.0)])
    sleeps = []
    records = fetch_cloudwatch_records(
        client,
        {},
        metadata(),
        base_delay_seconds=2,
        jitter_seconds=0,
        sleep=sleeps.append,
    )
    assert [record.metric_value for record in records] == [1.0, 2.0]
    assert client.calls == [{}, {"NextToken": "page-2"}, {"NextToken": "page-2"}]
    assert sleeps == [2.0]


def test_caller_starting_token_is_preserved_as_explicit_resume_point():
    client = FakeClient([response(2.0)])
    records = fetch_cloudwatch_records(client, {"NextToken": "resume"}, metadata())
    assert [record.metric_value for record in records] == [2.0]
    assert client.calls == [{"NextToken": "resume"}]


def test_non_string_response_token_is_rejected():
    client = FakeClient([response(next_token=123)])
    with pytest.raises(CloudWatchPaginationError, match="NextToken"):
        fetch_cloudwatch_records(client, {}, metadata())
