from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from anomaly_detector.data_ingestion.checkpoint_store import (
    create_checkpoint,
    get_checkpoint,
    list_incomplete_checkpoints,
    query_identity_fingerprint,
    update_checkpoint,
)
from anomaly_detector.data_ingestion.telemetry_persistence import TelemetryPersistenceError, initialize_database
from anomaly_detector.utils.exceptions import DataValidationError


WINDOW_START = datetime(2026, 10, 25, 12, tzinfo=timezone(timedelta(hours=2)))
WINDOW_END = datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
QUERY = {
    "account_id": "123",
    "region": "us-east-1",
    "namespace": "AWS/EC2",
    "metric_name": "CPUUtilization",
    "dimensions": {"B": "2", "A": "1"},
    "period": 300,
    "statistic": "Average",
    "options": {"ScanBy": "TimestampDescending"},
}


def create(path, **overrides):
    values = {
        "database_path": path,
        "run_id": "run-1",
        "query_configuration": QUERY,
        "source_type": "aws_cloudwatch",
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "account_id": "123",
        "region": "us-east-1",
        "namespace": "AWS/EC2",
        "metric_name": "CPUUtilization",
        "dimensions_json": '{"A":"1","B":"2"}',
        "period": 300,
        "statistic": "Average",
        "now": datetime(2026, 10, 25, 14, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return create_checkpoint(**values)


def test_fingerprint_is_deterministic_and_normalizes_utc():
    equivalent = dict(QUERY)
    equivalent["dimensions"] = {"A": "1", "B": "2"}
    assert query_identity_fingerprint(QUERY, WINDOW_START, WINDOW_END) == query_identity_fingerprint(equivalent, WINDOW_START.astimezone(timezone.utc), WINDOW_END)
    changed = dict(QUERY, statistic="Maximum")
    assert query_identity_fingerprint(QUERY, WINDOW_START, WINDOW_END) != query_identity_fingerprint(changed, WINDOW_START, WINDOW_END)


@pytest.mark.parametrize("bad", [None, "bad", [], {"value": object()}])
def test_invalid_fingerprint_inputs_are_rejected(bad):
    with pytest.raises(DataValidationError):
        query_identity_fingerprint(bad, WINDOW_START, WINDOW_END)
    with pytest.raises(DataValidationError):
        query_identity_fingerprint(QUERY, WINDOW_END, WINDOW_START)


def test_checkpoint_creation_retrieval_and_reopening(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    checkpoint = create(path)
    restored = get_checkpoint(path, "run-1")
    assert checkpoint == restored
    assert restored.run_status == "pending"
    assert restored.window_start_utc == "2026-10-25T10:00:00+00:00"


def test_state_transitions_and_progress_updates(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    checkpoint = create(path)
    fingerprint = checkpoint.query_fingerprint
    running = update_checkpoint(path, "run-1", expected_query_fingerprint=fingerprint, new_status="running")
    committed = update_checkpoint(path, "run-1", expected_query_fingerprint=fingerprint, new_status="page_committed", pagination_token="token", committed_page_count=1, committed_record_count=4, last_committed_datapoint_identity="identity")
    completed = update_checkpoint(path, "run-1", expected_query_fingerprint=fingerprint, new_status="completed", committed_page_count=2, committed_record_count=8)
    assert running.run_status == "running"
    assert committed.pagination_token == "token"
    assert completed.run_status == "completed"
    assert completed.completed_at_utc is not None


def test_invalid_transition_and_query_mismatch_are_rejected(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    checkpoint = create(path)
    with pytest.raises(DataValidationError):
        update_checkpoint(path, "run-1", expected_query_fingerprint="wrong", new_status="running")
    update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, new_status="running")
    with pytest.raises(DataValidationError):
        update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, new_status="pending")


def test_incomplete_listing_and_sanitized_failure(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    checkpoint = create(path)
    failure = ValueError("secret-like detail " + "x" * 2000)
    failed = update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, new_status="failed", error=failure)
    assert failed.last_error_class == "ValueError"
    assert len(failed.last_error_message) == 1000
    assert list_incomplete_checkpoints(path)[0].run_id == "run-1"
    update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, new_status="needs_recovery")
    update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, new_status="running")
    update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, new_status="completed")
    assert list_incomplete_checkpoints(path) == []


def test_unsupported_checkpoint_schema_is_rejected(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    create(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE checkpoint_schema SET schema_version = 999")
    with pytest.raises(TelemetryPersistenceError, match="Unsupported checkpoint schema"):
        get_checkpoint(path, "run-1")


def test_initialization_remains_safe_and_existing_database_is_not_overwritten(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    initialize_database(path)
    initialize_database(path)
    create(path)
    initialize_database(path)
    assert get_checkpoint(path, "run-1").run_status == "pending"


def test_progress_counters_cannot_decrease(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    checkpoint = create(path)
    with pytest.raises(DataValidationError):
        update_checkpoint(path, "run-1", expected_query_fingerprint=checkpoint.query_fingerprint, committed_page_count=-1)
