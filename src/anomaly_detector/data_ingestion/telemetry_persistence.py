"""SQLite persistence for normalized telemetry records.

This module is intentionally independent from CloudWatch collection and the
weather pipeline. It provides local durable storage only; checkpointing,
retention, migrations, and service integration are outside this phase.
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.detection.generic_detector import GenericDetectionResult
from anomaly_detector.detection.advanced_detector import AdvancedDetectionResult
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError

SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
SQLITE_TIMEOUT_SECONDS = 5.0
CHECKPOINT_STATES = frozenset({"pending", "running", "page_committed", "completed", "failed", "needs_recovery"})
_VALID_STATE_TRANSITIONS = {
    "pending": {"running", "failed", "needs_recovery"},
    "running": {"running", "page_committed", "completed", "failed", "needs_recovery"},
    "page_committed": {"page_committed", "running", "completed", "failed", "needs_recovery"},
    "needs_recovery": {"running", "failed", "needs_recovery"},
    "failed": {"needs_recovery", "running"},
    "completed": set(),
}


class TelemetryPersistenceError(DataIngestionError):
    """Raised when SQLite telemetry persistence cannot complete safely."""


@dataclass(frozen=True)
class PersistenceResult:
    """Counts for one transactional persistence batch."""

    inserted: int
    already_existing: int
    conflicts: int


@dataclass(frozen=True)
class ConflictEvent:
    """A persisted conflict between an existing and incoming observation."""

    conflict_id: int
    datapoint_identity: str
    existing_metric_value: float
    incoming_metric_value: float
    existing_is_anomaly: bool | None
    incoming_is_anomaly: bool | None
    detected_at_utc: str


@dataclass(frozen=True)
class Checkpoint:
    """Persisted ingestion progress bound to one exact query fingerprint."""

    run_id: str
    checkpoint_schema_version: int
    query_fingerprint: str
    source_type: str
    run_status: str
    window_start_utc: str
    window_end_utc: str
    account_id: str | None
    region: str | None
    namespace: str | None
    metric_name: str | None
    dimensions_json: str | None
    resource_identity: str | None
    metric_unit: str | None
    period: int | None
    statistic: str | None
    pagination_token: str | None
    committed_page_count: int
    committed_record_count: int
    last_committed_datapoint_identity: str | None
    created_at_utc: str
    started_at_utc: str | None
    updated_at_utc: str
    completed_at_utc: str | None
    last_error_class: str | None
    last_error_message: str | None


@dataclass(frozen=True)
class DetectionPersistenceResult:
    """Counts for a generic detection-result persistence batch."""

    inserted: int
    already_existing: int


_SCHEMA_STATEMENTS = (
"""
CREATE TABLE IF NOT EXISTS telemetry_schema (
    schema_name TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS telemetry_records (
    datapoint_identity TEXT PRIMARY KEY,
    time_series_identity TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    timestamp_utc TEXT NOT NULL,
    source_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    entity_id TEXT,
    is_anomaly INTEGER CHECK (is_anomaly IS NULL OR is_anomaly IN (0, 1)),
    namespace TEXT,
    dimensions_json TEXT NOT NULL,
    resource_arn TEXT,
    account_id TEXT,
    region TEXT,
    period INTEGER,
    statistic TEXT
)
""",
"""
CREATE TABLE IF NOT EXISTS telemetry_conflicts (
    conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    datapoint_identity TEXT NOT NULL,
    existing_metric_value REAL NOT NULL,
    incoming_metric_value REAL NOT NULL,
    existing_is_anomaly INTEGER,
    incoming_is_anomaly INTEGER,
    detected_at_utc TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS checkpoint_schema (
    schema_name TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
    run_id TEXT PRIMARY KEY,
    checkpoint_schema_version INTEGER NOT NULL,
    query_fingerprint TEXT NOT NULL,
    source_type TEXT NOT NULL,
    run_status TEXT NOT NULL,
    window_start_utc TEXT NOT NULL,
    window_end_utc TEXT NOT NULL,
    account_id TEXT,
    region TEXT,
    namespace TEXT,
    metric_name TEXT,
    dimensions_json TEXT,
    resource_identity TEXT,
    metric_unit TEXT,
    period INTEGER,
    statistic TEXT,
    pagination_token TEXT,
    committed_page_count INTEGER NOT NULL DEFAULT 0,
    committed_record_count INTEGER NOT NULL DEFAULT 0,
    last_committed_datapoint_identity TEXT,
    created_at_utc TEXT NOT NULL,
    started_at_utc TEXT,
    updated_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    last_error_class TEXT,
    last_error_message TEXT,
    UNIQUE(query_fingerprint, window_start_utc, window_end_utc)
)
""",
"""
CREATE TABLE IF NOT EXISTS generic_detection_results (
    datapoint_identity TEXT PRIMARY KEY,
    metric_name TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    metric_value REAL NOT NULL,
    predicted_is_anomaly INTEGER NOT NULL CHECK (predicted_is_anomaly IN (0, 1)),
    explanation TEXT NOT NULL,
    source_type TEXT NOT NULL,
    entity_id TEXT,
    namespace TEXT,
    dimensions_json TEXT NOT NULL,
    account_id TEXT,
    region TEXT,
    resource_arn TEXT,
    metric_unit TEXT,
    period INTEGER,
    statistic TEXT,
    created_at_utc TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS advanced_detection_results (
    result_identity TEXT PRIMARY KEY,
    datapoint_identity TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    metric_value REAL NOT NULL,
    predicted_is_anomaly INTEGER NOT NULL CHECK (predicted_is_anomaly IN (0, 1)),
    explanation TEXT NOT NULL,
    detection_method TEXT NOT NULL,
    reference_value REAL,
    statistical_score REAL,
    source_type TEXT NOT NULL,
    entity_id TEXT,
    namespace TEXT,
    dimensions_json TEXT NOT NULL,
    account_id TEXT,
    region TEXT,
    resource_arn TEXT,
    metric_unit TEXT,
    period INTEGER,
    statistic TEXT,
    created_at_utc TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    alert_id TEXT PRIMARY KEY,
    detection_identity TEXT NOT NULL UNIQUE,
    detection_method TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    entity_id TEXT,
    source_type TEXT NOT NULL,
    observed_value REAL NOT NULL,
    severity TEXT NOT NULL,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS detection_configurations (
    name TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
)
""",
)

_REQUIRED_COLUMNS = {
    "telemetry_schema": {"schema_name", "schema_version"},
    "telemetry_records": {
        "datapoint_identity", "time_series_identity", "schema_version", "timestamp_utc",
        "source_type", "metric_name", "metric_value", "metric_unit", "entity_id", "is_anomaly",
        "namespace", "dimensions_json", "resource_arn", "account_id", "region", "period", "statistic",
    },
    "telemetry_conflicts": {
        "conflict_id", "datapoint_identity", "existing_metric_value", "incoming_metric_value",
        "existing_is_anomaly", "incoming_is_anomaly", "detected_at_utc",
    },
    "checkpoint_schema": {"schema_name", "schema_version"},
    "ingestion_checkpoints": {
        "run_id", "checkpoint_schema_version", "query_fingerprint", "source_type", "run_status",
        "window_start_utc", "window_end_utc", "account_id", "region", "namespace", "metric_name",
        "dimensions_json", "resource_identity", "metric_unit", "period", "statistic", "pagination_token",
        "committed_page_count", "committed_record_count", "last_committed_datapoint_identity",
        "created_at_utc", "started_at_utc", "updated_at_utc", "completed_at_utc",
        "last_error_class", "last_error_message",
    },
    "generic_detection_results": {
        "datapoint_identity", "metric_name", "timestamp_utc", "metric_value",
        "predicted_is_anomaly", "explanation", "source_type", "entity_id", "namespace",
        "dimensions_json",
        "account_id", "region", "resource_arn", "metric_unit", "period", "statistic",
        "created_at_utc",
    },
    "advanced_detection_results": {
        "result_identity", "datapoint_identity", "metric_name", "timestamp_utc", "metric_value",
        "predicted_is_anomaly", "explanation", "detection_method", "reference_value", "statistical_score",
        "source_type", "entity_id", "namespace", "dimensions_json", "account_id", "region",
        "resource_arn", "metric_unit", "period", "statistic", "created_at_utc",
    },
    "anomaly_alerts": {
        "alert_id", "detection_identity", "detection_method", "metric_name", "entity_id",
        "source_type", "observed_value", "severity", "explanation", "status",
        "created_at_utc", "updated_at_utc",
    },
    "detection_configurations": {"name", "payload_json", "updated_at_utc"},
}
_CORE_TABLES = {"telemetry_schema", "telemetry_records", "telemetry_conflicts"}
_CHECKPOINT_TABLES = {"checkpoint_schema", "ingestion_checkpoints"}
_RESULT_TABLES = {"generic_detection_results"}
_ADVANCED_RESULT_TABLES = {"advanced_detection_results"}
_ALERT_TABLES = {"anomaly_alerts", "detection_configurations"}


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _identity_json(identity: tuple[object, ...]) -> str:
    return json.dumps(_json_value(identity), separators=(",", ":"), ensure_ascii=True)


def _label_value(label: bool | None) -> int | None:
    return None if label is None else int(label)


def _timestamp_text(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat()


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
    connection.execute(f"PRAGMA busy_timeout = {int(SQLITE_TIMEOUT_SECONDS * 1000)}")
    connection.row_factory = sqlite3.Row
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        )
    }


def _validate_schema(connection: sqlite3.Connection, *, require_checkpoints: bool = True) -> None:
    tables = _table_names(connection)
    required_tables = set(_REQUIRED_COLUMNS) if require_checkpoints else _CORE_TABLES
    missing_tables = required_tables - tables
    if missing_tables:
        raise TelemetryPersistenceError(f"Incompatible telemetry database; missing tables: {sorted(missing_tables)}")
    for table_name, required_columns in _REQUIRED_COLUMNS.items():
        if table_name not in required_tables:
            continue
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})")
        }
        missing_columns = required_columns - columns
        if missing_columns:
            raise TelemetryPersistenceError(
                f"Incompatible telemetry database; {table_name} missing columns: {sorted(missing_columns)}",
            )
    schema_row = connection.execute(
        "SELECT schema_version FROM telemetry_schema WHERE schema_name = 'telemetry'",
    ).fetchone()
    if schema_row is None or schema_row[0] != SCHEMA_VERSION:
        raise TelemetryPersistenceError(
            f"Unsupported telemetry schema version: {None if schema_row is None else schema_row[0]!r}",
        )
    record_pk = connection.execute(
        "SELECT pk FROM pragma_table_info('telemetry_records') WHERE name = 'datapoint_identity'",
    ).fetchone()
    unique_indexes = connection.execute("PRAGMA index_list(telemetry_records)").fetchall()
    has_unique_identity = bool(record_pk and record_pk[0] == 1)
    for index in unique_indexes:
        if index[2]:
            indexed_columns = [row[2] for row in connection.execute(f"PRAGMA index_info({index[1]})")]
            if indexed_columns == ["datapoint_identity"]:
                has_unique_identity = True
    if not has_unique_identity:
        raise TelemetryPersistenceError("Incompatible telemetry database; datapoint identity is not unique.")
    if require_checkpoints:
        checkpoint_row = connection.execute(
            "SELECT schema_version FROM checkpoint_schema WHERE schema_name = 'checkpoints'",
        ).fetchone()
        if checkpoint_row is None or checkpoint_row[0] != CHECKPOINT_SCHEMA_VERSION:
            raise TelemetryPersistenceError(
                f"Unsupported checkpoint schema version: {None if checkpoint_row is None else checkpoint_row[0]!r}",
            )


def initialize_database(database_path: str | Path) -> None:
    """Create the telemetry and conflict tables if they do not exist."""
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        existing_tables = _table_names(connection)
        if existing_tables:
            _validate_schema(connection, require_checkpoints=False)
            if not _CHECKPOINT_TABLES.issubset(existing_tables) or not _RESULT_TABLES.issubset(existing_tables) or not _ADVANCED_RESULT_TABLES.issubset(existing_tables) or not _ALERT_TABLES.issubset(existing_tables):
                connection.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA_STATEMENTS[3:]:
                    connection.execute(statement)
                if not _CHECKPOINT_TABLES.issubset(existing_tables):
                    connection.execute(
                        "INSERT INTO checkpoint_schema (schema_name, schema_version) VALUES (?, ?)",
                        ("checkpoints", CHECKPOINT_SCHEMA_VERSION),
                    )
                connection.commit()
            _validate_schema(connection)
            return
        connection.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO telemetry_schema (schema_name, schema_version) VALUES (?, ?)",
            ("telemetry", SCHEMA_VERSION),
        )
        connection.execute(
            "INSERT INTO checkpoint_schema (schema_name, schema_version) VALUES (?, ?)",
            ("checkpoints", CHECKPOINT_SCHEMA_VERSION),
        )
        connection.commit()
        _validate_schema(connection)
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise TelemetryPersistenceError(f"Unable to initialize telemetry database: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()


def persist_telemetry_records(
    database_path: str | Path,
    records: Iterable[TelemetryRecord],
) -> PersistenceResult:
    """Persist a batch transactionally using ``datapoint_identity`` uniqueness.

    Existing identical rows are counted as ``already_existing``. An existing
    identity with a different metric value or label is counted as a conflict;
    the original row is preserved and a conflict event is stored, then the
    batch continues. SQLite failures roll back the entire batch.
    """
    if isinstance(records, (str, bytes)):
        raise DataValidationError("records must be an iterable of TelemetryRecord objects.")
    try:
        record_list = list(records)
    except TypeError as exc:
        raise DataValidationError("records must be an iterable of TelemetryRecord objects.") from exc
    if any(not isinstance(record, TelemetryRecord) for record in record_list):
        raise DataValidationError("records must contain only TelemetryRecord objects.")

    initialize_database(database_path)

    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = persist_telemetry_records_in_connection(connection, record_list)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise TelemetryPersistenceError(f"Unable to persist telemetry batch; transaction rolled back: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return result


def persist_telemetry_records_in_connection(
    connection: sqlite3.Connection,
    records: Iterable[TelemetryRecord],
) -> PersistenceResult:
    """Apply telemetry writes to an active transaction without committing."""
    record_list = list(records)
    if any(not isinstance(record, TelemetryRecord) for record in record_list):
        raise DataValidationError("records must contain only TelemetryRecord objects.")
    inserted = already_existing = conflicts = 0
    for record in record_list:
        identity = _identity_json(record.datapoint_identity)
        existing = connection.execute(
            "SELECT metric_value, is_anomaly FROM telemetry_records WHERE datapoint_identity = ?",
            (identity,),
        ).fetchone()
        if existing is not None:
            incoming_label = _label_value(record.is_anomaly)
            if existing["metric_value"] == record.metric_value and existing["is_anomaly"] == incoming_label:
                already_existing += 1
            else:
                connection.execute(
                    """INSERT INTO telemetry_conflicts (
                        datapoint_identity, existing_metric_value, incoming_metric_value,
                        existing_is_anomaly, incoming_is_anomaly, detected_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (identity, existing["metric_value"], record.metric_value,
                     existing["is_anomaly"], incoming_label, datetime.now(timezone.utc).isoformat()),
                )
                conflicts += 1
            continue
        connection.execute(
            """INSERT INTO telemetry_records (
                datapoint_identity, time_series_identity, schema_version,
                timestamp_utc, source_type, metric_name, metric_value, metric_unit,
                entity_id, is_anomaly, namespace, dimensions_json, resource_arn,
                account_id, region, period, statistic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (identity, _identity_json(record.time_series_identity), SCHEMA_VERSION,
             _timestamp_text(record.timestamp), record.source_type, record.metric_name,
             record.metric_value, record.metric_unit, record.entity_id, _label_value(record.is_anomaly),
             record.namespace, json.dumps(dict(record.dimensions), sort_keys=True, separators=(",", ":")),
             record.resource_arn, record.account_id, record.region, record.period, record.statistic),
        )
        inserted += 1
    return PersistenceResult(inserted, already_existing, conflicts)


def retrieve_telemetry_records(database_path: str | Path) -> list[TelemetryRecord]:
    """Retrieve persisted telemetry records in insertion order."""
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        with connection:
            rows = connection.execute(
                "SELECT * FROM telemetry_records ORDER BY rowid",
            ).fetchall()
    except sqlite3.Error as exc:
        raise TelemetryPersistenceError(f"Unable to retrieve telemetry records: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [
        TelemetryRecord(
            timestamp=datetime.fromisoformat(row["timestamp_utc"]),
            source_type=row["source_type"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            metric_unit=row["metric_unit"],
            entity_id=row["entity_id"],
            is_anomaly=None if row["is_anomaly"] is None else bool(row["is_anomaly"]),
            namespace=row["namespace"],
            dimensions=json.loads(row["dimensions_json"]),
            resource_arn=row["resource_arn"],
            account_id=row["account_id"],
            region=row["region"],
            period=row["period"],
            statistic=row["statistic"],
        )
        for row in rows
    ]


def retrieve_conflict_events(database_path: str | Path) -> list[ConflictEvent]:
    """Retrieve conflict events in deterministic conflict-id order."""
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        rows = connection.execute(
            """SELECT conflict_id, datapoint_identity, existing_metric_value,
                      incoming_metric_value, existing_is_anomaly,
                      incoming_is_anomaly, detected_at_utc
               FROM telemetry_conflicts ORDER BY conflict_id""",
        ).fetchall()
    except sqlite3.Error as exc:
        raise TelemetryPersistenceError(f"Unable to retrieve telemetry conflicts: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [
        ConflictEvent(
            conflict_id=row["conflict_id"],
            datapoint_identity=row["datapoint_identity"],
            existing_metric_value=row["existing_metric_value"],
            incoming_metric_value=row["incoming_metric_value"],
            existing_is_anomaly=None if row["existing_is_anomaly"] is None else bool(row["existing_is_anomaly"]),
            incoming_is_anomaly=None if row["incoming_is_anomaly"] is None else bool(row["incoming_is_anomaly"]),
            detected_at_utc=row["detected_at_utc"],
        )
        for row in rows
    ]


def persist_generic_detection_results(
    database_path: str | Path,
    results: Iterable[GenericDetectionResult],
) -> DetectionPersistenceResult:
    """Persist generic detector results transactionally and idempotently."""
    if isinstance(results, (str, bytes)):
        raise DataValidationError("results must be an iterable of GenericDetectionResult objects.")
    try:
        result_list = list(results)
    except TypeError as exc:
        raise DataValidationError("results must be an iterable of GenericDetectionResult objects.") from exc
    if any(not isinstance(result, GenericDetectionResult) for result in result_list):
        raise DataValidationError("results must contain only GenericDetectionResult objects.")
    initialize_database(database_path)
    inserted = already_existing = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for result in result_list:
                record = result.record
                identity = _identity_json(record.datapoint_identity)
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO generic_detection_results (
                        datapoint_identity, metric_name, timestamp_utc, metric_value,
                        predicted_is_anomaly, explanation, source_type, entity_id, namespace,
                        dimensions_json, account_id, region, resource_arn, metric_unit,
                        period, statistic, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (identity, record.metric_name, _timestamp_text(record.timestamp), record.metric_value,
                     int(result.predicted_is_anomaly), result.explanation, record.source_type,
                     record.entity_id, record.namespace,
                     json.dumps(dict(record.dimensions), sort_keys=True, separators=(",", ":")),
                     record.account_id, record.region, record.resource_arn, record.metric_unit,
                     record.period, record.statistic, datetime.now(timezone.utc).isoformat()),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    already_existing += 1
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise TelemetryPersistenceError(f"Unable to persist generic detection results; transaction rolled back: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return DetectionPersistenceResult(inserted, already_existing)


def retrieve_generic_detection_results(
    database_path: str | Path,
    *,
    anomaly: bool | None = None,
    metric_name: str | None = None,
) -> list[GenericDetectionResult]:
    """Retrieve generic results, optionally filtering by anomaly and metric."""
    if anomaly is not None and not isinstance(anomaly, bool):
        raise DataValidationError("anomaly filter must be Boolean or None.")
    if metric_name is not None and (not isinstance(metric_name, str) or not metric_name.strip()):
        raise DataValidationError("metric_name filter must be a non-empty string or None.")
    initialize_database(database_path)
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        clauses: list[str] = []
        parameters: list[object] = []
        if anomaly is not None:
            clauses.append("predicted_is_anomaly = ?")
            parameters.append(int(anomaly))
        if metric_name is not None:
            clauses.append("metric_name = ?")
            parameters.append(metric_name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            "SELECT * FROM generic_detection_results" + where + " ORDER BY rowid",
            parameters,
        ).fetchall()
    except sqlite3.Error as exc:
        raise TelemetryPersistenceError(f"Unable to retrieve generic detection results: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [
        GenericDetectionResult(
            record=TelemetryRecord(
                timestamp=datetime.fromisoformat(row["timestamp_utc"]),
                source_type=row["source_type"], metric_name=row["metric_name"],
                metric_value=row["metric_value"], entity_id=row["entity_id"],
                namespace=row["namespace"], dimensions=json.loads(row["dimensions_json"]),
                account_id=row["account_id"], region=row["region"], resource_arn=row["resource_arn"],
                metric_unit=row["metric_unit"], period=row["period"], statistic=row["statistic"],
            ),
            predicted_is_anomaly=bool(row["predicted_is_anomaly"]),
            explanation=row["explanation"],
        )
        for row in rows
    ]


def persist_advanced_detection_results(
    database_path: str | Path,
    results: Iterable[AdvancedDetectionResult],
) -> DetectionPersistenceResult:
    """Persist advanced results with detection-method-specific uniqueness."""
    if isinstance(results, (str, bytes)):
        raise DataValidationError("results must be an iterable of AdvancedDetectionResult objects.")
    result_list = list(results)
    if any(not isinstance(result, AdvancedDetectionResult) for result in result_list):
        raise DataValidationError("results must contain only AdvancedDetectionResult objects.")
    initialize_database(database_path)
    inserted = already_existing = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for result in result_list:
                record = result.record
                datapoint_identity = _identity_json(record.datapoint_identity)
                result_identity = _identity_json(record.datapoint_identity + (result.detection_method,))
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO advanced_detection_results (
                        result_identity, datapoint_identity, metric_name, timestamp_utc,
                        metric_value, predicted_is_anomaly, explanation, detection_method,
                        reference_value, statistical_score, source_type, entity_id, namespace,
                        dimensions_json, account_id, region, resource_arn, metric_unit,
                        period, statistic, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (result_identity, datapoint_identity, record.metric_name, _timestamp_text(record.timestamp),
                     record.metric_value, int(result.predicted_is_anomaly), result.explanation,
                     result.detection_method, result.reference_value, result.statistical_score,
                     record.source_type, record.entity_id, record.namespace,
                     json.dumps(dict(record.dimensions), sort_keys=True, separators=(",", ":")),
                     record.account_id, record.region, record.resource_arn, record.metric_unit,
                     record.period, record.statistic, datetime.now(timezone.utc).isoformat()),
                )
                inserted += int(cursor.rowcount == 1)
                already_existing += int(cursor.rowcount != 1)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise TelemetryPersistenceError(f"Unable to persist advanced detection results; transaction rolled back: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return DetectionPersistenceResult(inserted, already_existing)


def retrieve_advanced_detection_results(
    database_path: str | Path,
    *,
    anomaly: bool | None = None,
    metric_name: str | None = None,
    detection_method: str | None = None,
) -> list[AdvancedDetectionResult]:
    """Retrieve advanced results with optional classification/metric/method filters."""
    if anomaly is not None and not isinstance(anomaly, bool):
        raise DataValidationError("anomaly filter must be Boolean or None.")
    for name, value in (("metric_name", metric_name), ("detection_method", detection_method)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DataValidationError(f"{name} filter must be a non-empty string or None.")
    initialize_database(database_path)
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        clauses: list[str] = []
        parameters: list[object] = []
        if anomaly is not None:
            clauses.append("predicted_is_anomaly = ?")
            parameters.append(int(anomaly))
        if metric_name is not None:
            clauses.append("metric_name = ?")
            parameters.append(metric_name)
        if detection_method is not None:
            clauses.append("detection_method = ?")
            parameters.append(detection_method)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute("SELECT * FROM advanced_detection_results" + where + " ORDER BY rowid", parameters).fetchall()
    except sqlite3.Error as exc:
        raise TelemetryPersistenceError(f"Unable to retrieve advanced detection results: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [
        AdvancedDetectionResult(
            record=TelemetryRecord(
                timestamp=datetime.fromisoformat(row["timestamp_utc"]), source_type=row["source_type"],
                metric_name=row["metric_name"], metric_value=row["metric_value"], entity_id=row["entity_id"],
                namespace=row["namespace"], dimensions=json.loads(row["dimensions_json"]),
                account_id=row["account_id"], region=row["region"], resource_arn=row["resource_arn"],
                metric_unit=row["metric_unit"], period=row["period"], statistic=row["statistic"],
            ),
            predicted_is_anomaly=bool(row["predicted_is_anomaly"]), explanation=row["explanation"],
            detection_method=row["detection_method"], observed_value=row["metric_value"],
            reference_value=row["reference_value"], statistical_score=row["statistical_score"],
        )
        for row in rows
    ]
