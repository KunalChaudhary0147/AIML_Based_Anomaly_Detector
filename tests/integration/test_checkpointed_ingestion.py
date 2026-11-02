from datetime import datetime, timezone
import sqlite3

import pytest

from anomaly_detector.data_ingestion.checkpoint_store import get_checkpoint
from anomaly_detector.data_ingestion.checkpointed_ingestion import run_checkpointed_ingestion
from anomaly_detector.data_ingestion.telemetry_persistence import retrieve_telemetry_records
from anomaly_detector.utils.exceptions import CloudWatchClientError, DataValidationError


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


def page(value, hour=12, token=None):
    response = {
        "MetricDataResults": [{
            "Id": "cpu",
            "Timestamps": [datetime(2026, 10, 25, hour, tzinfo=timezone.utc)],
            "Values": [value],
            "StatusCode": "Complete",
        }],
    }
    if token is not None:
        response["NextToken"] = token
    return response


QUERY = {"Id": "cpu", "namespace": "AWS/EC2", "metric_name": "CPUUtilization", "dimensions": {"InstanceId": "i-1"}, "period": 300, "statistic": "Average"}
METADATA = {"cpu": {"namespace": "AWS/EC2", "metric_name": "CPUUtilization", "dimensions": {"InstanceId": "i-1"}, "period": 300, "statistic": "Average"}}
START = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
END = datetime(2026, 10, 25, 14, tzinfo=timezone.utc)


def run(path, client, **kwargs):
    return run_checkpointed_ingestion(
        path, client, {}, METADATA, query_configuration=QUERY,
        source_type="aws_cloudwatch", window_start=START, window_end=END,
        **kwargs,
    )


def test_successful_pages_persist_before_next_request_and_complete(tmp_path):
    path = tmp_path / "state.sqlite"
    client = FakeClient([page(1.0, token="p2"), page(2.0, hour=13)])
    checkpoint = run(path, client)
    assert checkpoint.run_status == "completed"
    assert checkpoint.committed_page_count == 2
    assert checkpoint.committed_record_count == 2
    assert [record.metric_value for record in retrieve_telemetry_records(path)] == [1.0, 2.0]
    assert client.calls == [{}, {"NextToken": "p2"}]


def test_repeated_completed_run_is_deterministic_and_does_not_duplicate(tmp_path):
    path = tmp_path / "state.sqlite"
    first_client = FakeClient([page(1.0)])
    first = run(path, first_client, run_id="run-1")
    second = run(path, FakeClient([]), run_id="run-1")
    assert first == second
    assert len(retrieve_telemetry_records(path)) == 1


def test_query_fingerprint_mismatch_is_rejected(tmp_path):
    path = tmp_path / "state.sqlite"
    run(path, FakeClient([page(1.0)]), run_id="run-1")
    with pytest.raises(DataValidationError):
        run_checkpointed_ingestion(
            path, FakeClient([]), {}, METADATA,
            query_configuration={**QUERY, "statistic": "Maximum"},
            source_type="aws_cloudwatch", window_start=START, window_end=END,
            run_id="run-1",
        )


def test_page_transaction_rolls_back_records_and_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite"
    client = FakeClient([page(1.0, token="p2"), page(2.0, hour=13)])
    module = __import__("anomaly_detector.data_ingestion.checkpointed_ingestion", fromlist=["dummy"])
    real_persist = module.persist_telemetry_records_in_connection

    def fail_on_second_page(connection, records):
        if records and records[0].metric_value == 2.0:
            raise sqlite3.OperationalError("simulated page write failure")
        return real_persist(connection, records)

    monkeypatch.setattr(module, "persist_telemetry_records_in_connection", fail_on_second_page)
    with pytest.raises(sqlite3.OperationalError):
        run(path, client, run_id="run-1")
    assert [record.metric_value for record in retrieve_telemetry_records(path)] == [1.0]
    checkpoint = get_checkpoint(path, "run-1")
    assert checkpoint.run_status == "failed"
    assert checkpoint.committed_page_count == 1
    assert checkpoint.committed_record_count == 1


def test_invalid_token_replays_window_and_deduplicates(tmp_path):
    class InvalidToken(Exception):
        response = {"Error": {"Code": "InvalidNextToken"}}

    path = tmp_path / "state.sqlite"
    client = FakeClient([page(1.0, token="p2"), InvalidToken("expired"), page(1.0)])
    checkpoint = run(path, client)
    assert checkpoint.run_status == "completed"
    assert len(retrieve_telemetry_records(path)) == 1
    assert client.calls == [{}, {"NextToken": "p2"}, {}]


def test_failure_does_not_complete_run(tmp_path):
    path = tmp_path / "state.sqlite"
    client = FakeClient([CloudWatchClientError("permanent")])
    with pytest.raises(CloudWatchClientError):
        run(path, client, run_id="run-1")
    assert get_checkpoint(path, "run-1").run_status == "failed"
