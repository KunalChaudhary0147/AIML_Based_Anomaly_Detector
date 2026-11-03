from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import (
    PersistenceResult,
    TelemetryPersistenceError,
    initialize_database,
    persist_telemetry_records,
    retrieve_conflict_events,
    retrieve_telemetry_records,
)
import anomaly_detector.data_ingestion.telemetry_persistence as persistence_module
from anomaly_detector.utils.exceptions import DataValidationError


def make_record(**overrides):
    values = {
        "timestamp": datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
        "source_type": "aws_cloudwatch",
        "metric_name": "CPUUtilization",
        "metric_value": 42.0,
        "metric_unit": "Percent",
        "entity_id": "i-123",
        "is_anomaly": None,
        "namespace": "AWS/EC2",
        "dimensions": {"InstanceId": "i-123"},
        "resource_arn": "arn:aws:ec2:us-east-1:123:instance/i-123",
        "account_id": "123",
        "region": "us-east-1",
        "period": 300,
        "statistic": "Average",
    }
    values.update(overrides)
    return TelemetryRecord(**values)


def test_schema_initialization_and_reopening(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    initialize_database(path)
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"telemetry_records", "telemetry_conflicts"}.issubset(tables)


def test_insert_and_retrieve_round_trip_with_nulls(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record(metric_unit=None, entity_id=None, is_anomaly=None, namespace=None, dimensions={}, resource_arn=None, account_id=None, region=None, period=None, statistic=None)
    result = persist_telemetry_records(path, [original])
    restored = retrieve_telemetry_records(path)
    assert result == PersistenceResult(inserted=1, already_existing=0, conflicts=0)
    assert restored == [original]


def test_identity_serialization_is_deterministic(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    first = make_record(dimensions={"B": "2", "A": "1"})
    second = make_record(dimensions={"A": "1", "B": "2"})
    assert persist_telemetry_records(path, [first]) == PersistenceResult(1, 0, 0)
    assert persist_telemetry_records(path, [second]) == PersistenceResult(0, 1, 0)


def test_duplicate_identical_record_is_already_existing(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    record = make_record()
    assert persist_telemetry_records(path, [record]) == PersistenceResult(1, 0, 0)
    assert persist_telemetry_records(path, [record]) == PersistenceResult(0, 1, 0)
    assert retrieve_telemetry_records(path) == [record]


def test_conflicting_value_preserves_original_and_records_event(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record(metric_value=42.0)
    conflict = make_record(metric_value=99.0)
    persist_telemetry_records(path, [original])
    result = persist_telemetry_records(path, [conflict])
    assert result == PersistenceResult(0, 0, 1)
    assert retrieve_telemetry_records(path) == [original]
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM telemetry_conflicts").fetchone()[0] == 1


def test_conflicting_label_preserves_original_and_records_event(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record(is_anomaly=False)
    conflict = make_record(is_anomaly=True)
    persist_telemetry_records(path, [original])
    result = persist_telemetry_records(path, [conflict])
    assert result == PersistenceResult(0, 0, 1)
    assert retrieve_telemetry_records(path)[0].is_anomaly is False


def test_mixed_batch_reports_inserted_existing_and_conflict(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record()
    other = make_record(timestamp=datetime(2026, 10, 25, 13, tzinfo=timezone.utc))
    persist_telemetry_records(path, [original])
    result = persist_telemetry_records(path, [original, other, make_record(metric_value=5.0)])
    assert result == PersistenceResult(inserted=1, already_existing=1, conflicts=1)
    assert len(retrieve_telemetry_records(path)) == 2


def test_transaction_rolls_back_entire_batch_on_database_failure(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_bad_metric BEFORE INSERT ON telemetry_records "
            "WHEN NEW.metric_name = 'FAIL' BEGIN SELECT RAISE(ABORT, 'forced failure'); END;",
        )
    with pytest.raises(TelemetryPersistenceError):
        persist_telemetry_records(path, [make_record(), make_record(metric_name="FAIL", timestamp=datetime(2026, 10, 25, 13, tzinfo=timezone.utc))])
    assert retrieve_telemetry_records(path) == []


def test_invalid_input_is_rejected_without_database_side_effects(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    with pytest.raises(DataValidationError):
        persist_telemetry_records(path, [make_record(), object()])
    assert not path.exists()


def test_database_reopening_preserves_records(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    record = make_record()
    persist_telemetry_records(path, [record])
    assert retrieve_telemetry_records(path) == [record]
    assert persist_telemetry_records(path, [record]).already_existing == 1


def test_concurrent_reads_and_writes_complete_without_lock_errors(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    records = [
        make_record(timestamp=datetime(2026, 10, 25, hour, tzinfo=timezone.utc), entity_id=f"i-{hour}")
        for hour in range(8)
    ]

    def write_and_read(record):
        result = persist_telemetry_records(path, [record])
        return result.inserted, len(retrieve_telemetry_records(path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_and_read, records))

    assert sum(inserted for inserted, _ in results) == len(records)
    assert len(retrieve_telemetry_records(path)) == len(records)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_conflict_event_retrieval_is_ordered_and_complete(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record()
    persist_telemetry_records(path, [original])
    persist_telemetry_records(path, [make_record(metric_value=43.0)])
    persist_telemetry_records(path, [make_record(metric_value=44.0, is_anomaly=True)])
    events = retrieve_conflict_events(path)
    assert [event.conflict_id for event in events] == sorted(event.conflict_id for event in events)
    assert [event.incoming_metric_value for event in events] == [43.0, 44.0]
    assert events[1].incoming_is_anomaly is True
    assert events[0].datapoint_identity == events[1].datapoint_identity


def test_missing_required_table_is_rejected(tmp_path):
    path = tmp_path / "incompatible.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE telemetry_records (datapoint_identity TEXT PRIMARY KEY)")
    with pytest.raises(TelemetryPersistenceError, match="missing tables"):
        initialize_database(path)


def test_missing_required_column_is_rejected(tmp_path):
    path = tmp_path / "incompatible.sqlite"
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE telemetry_conflicts RENAME TO telemetry_conflicts_old")
        connection.execute("CREATE TABLE telemetry_conflicts (conflict_id INTEGER PRIMARY KEY)")
    with pytest.raises(TelemetryPersistenceError, match="missing columns"):
        initialize_database(path)


def test_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "incompatible.sqlite"
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE telemetry_schema SET schema_version = 999")
    with pytest.raises(TelemetryPersistenceError, match="Unsupported telemetry schema version"):
        initialize_database(path)


def test_missing_identity_uniqueness_is_rejected(tmp_path):
    path = tmp_path / "incompatible.sqlite"
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE telemetry_records RENAME TO telemetry_records_old")
        connection.execute("""CREATE TABLE telemetry_records (
            datapoint_identity TEXT, time_series_identity TEXT NOT NULL, schema_version INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL, source_type TEXT NOT NULL, metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL, metric_unit TEXT, entity_id TEXT, is_anomaly INTEGER,
            namespace TEXT, dimensions_json TEXT NOT NULL, resource_arn TEXT, account_id TEXT,
            region TEXT, period INTEGER, statistic TEXT
        )""")
    with pytest.raises(TelemetryPersistenceError, match="not unique"):
        initialize_database(path)


def test_conflict_event_failure_rolls_back_event_and_new_rows(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record()
    persist_telemetry_records(path, [original])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_conflicts BEFORE INSERT ON telemetry_conflicts "
            "BEGIN SELECT RAISE(ABORT, 'forced conflict failure'); END;",
        )
    with pytest.raises(TelemetryPersistenceError):
        persist_telemetry_records(path, [make_record(metric_value=99.0), make_record(timestamp=datetime(2026, 10, 25, 13, tzinfo=timezone.utc))])
    assert retrieve_telemetry_records(path) == [original]
    assert retrieve_conflict_events(path) == []


def test_later_record_failure_rolls_back_prior_conflict_and_insert(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    original = make_record()
    persist_telemetry_records(path, [original])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_bad_metric BEFORE INSERT ON telemetry_records "
            "WHEN NEW.metric_name = 'FAIL' BEGIN SELECT RAISE(ABORT, 'forced failure'); END;",
        )
    with pytest.raises(TelemetryPersistenceError):
        persist_telemetry_records(path, [make_record(metric_value=99.0), make_record(metric_name="FAIL", timestamp=datetime(2026, 10, 25, 13, tzinfo=timezone.utc))])
    assert retrieve_telemetry_records(path) == [original]
    assert retrieve_conflict_events(path) == []


def test_commit_failure_rolls_back_and_closes_connection(tmp_path, monkeypatch):
    path = tmp_path / "telemetry.sqlite"
    initialize_database(path)
    real_connect = persistence_module._connect
    wrappers = []

    class CommitFailingConnection:
        def __init__(self, connection):
            self.connection = connection
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            raise sqlite3.OperationalError("forced commit failure")

        def close(self):
            self.closed = True
            self.connection.close()

    def connect_with_commit_failure(database_path):
        wrapper = CommitFailingConnection(real_connect(database_path))
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(persistence_module, "_connect", connect_with_commit_failure)
    with pytest.raises(TelemetryPersistenceError, match="rolled back"):
        persist_telemetry_records(path, [make_record()])
    assert all(wrapper.closed for wrapper in wrappers)
    monkeypatch.undo()
    assert retrieve_telemetry_records(path) == []


def test_sqlite_writer_timeout_policy_is_explicit():
    assert persistence_module.SQLITE_TIMEOUT_SECONDS == 5.0
