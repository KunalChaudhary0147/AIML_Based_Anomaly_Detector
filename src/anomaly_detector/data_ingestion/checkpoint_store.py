"""Checkpoint API for SQLite-backed ingestion progress.

This module stores run/checkpoint state only. It does not fetch AWS data,
write telemetry observations, or resume CloudWatch collection.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from anomaly_detector.data_ingestion.telemetry_persistence import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    _connect,
    _validate_schema,
    initialize_database,
)
from anomaly_detector.utils.exceptions import DataValidationError
from anomaly_detector.data_ingestion.telemetry_persistence import TelemetryPersistenceError

CHECKPOINT_STATES = frozenset({"pending", "running", "page_committed", "completed", "failed", "needs_recovery"})
_VALID_STATE_TRANSITIONS = {
    "pending": {"running", "failed", "needs_recovery"},
    "running": {"page_committed", "completed", "failed", "needs_recovery"},
    "page_committed": {"running", "completed", "failed", "needs_recovery"},
    "needs_recovery": {"running", "failed", "needs_recovery"},
    "failed": {"needs_recovery", "running"},
    "completed": set(),
}


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError("Checkpoint timestamps must be timezone-aware datetimes.")
    return value.astimezone(timezone.utc).isoformat()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc_text(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataValidationError("Query configuration floats must be finite.")
        return value
    if isinstance(value, Mapping):
        normalized = {str(key): _canonical_value(item) for key, item in value.items()}
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise DataValidationError(f"Unsupported query configuration value: {type(value).__name__}")


def query_identity_fingerprint(query_configuration: Mapping[str, Any], window_start: datetime, window_end: datetime) -> str:
    """Return SHA-256 for deterministic query configuration and UTC window."""
    if not isinstance(query_configuration, Mapping):
        raise DataValidationError("query_configuration must be a mapping.")
    start = _utc_text(window_start)
    end = _utc_text(window_end)
    if start >= end:
        raise DataValidationError("Checkpoint window_start must precede window_end.")
    payload = {"query": _canonical_value(query_configuration), "window_start_utc": start, "window_end_utc": end}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint(**dict(row))


def create_checkpoint(
    database_path: str,
    *,
    run_id: str | None = None,
    query_configuration: Mapping[str, Any],
    source_type: str,
    window_start: datetime,
    window_end: datetime,
    account_id: str | None = None,
    region: str | None = None,
    namespace: str | None = None,
    metric_name: str | None = None,
    dimensions_json: str | None = None,
    resource_identity: str | None = None,
    metric_unit: str | None = None,
    period: int | None = None,
    statistic: str | None = None,
    now: datetime | None = None,
) -> Checkpoint:
    """Create a pending checkpoint bound to one query fingerprint."""
    if not isinstance(source_type, str) or not source_type.strip():
        raise DataValidationError("source_type must be a non-empty string.")
    fingerprint = query_identity_fingerprint(query_configuration, window_start, window_end)
    run_id = run_id or str(uuid4())
    if not isinstance(run_id, str) or not run_id.strip():
        raise DataValidationError("run_id must be a non-empty string.")
    now_text = _utc_text(now or datetime.now(timezone.utc))
    checkpoint = Checkpoint(
        run_id=run_id, checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        query_fingerprint=fingerprint, source_type=source_type,
        run_status="pending", window_start_utc=_utc_text(window_start), window_end_utc=_utc_text(window_end),
        account_id=account_id, region=region, namespace=namespace, metric_name=metric_name,
        dimensions_json=dimensions_json, resource_identity=resource_identity, metric_unit=metric_unit,
        period=period, statistic=statistic, pagination_token=None, committed_page_count=0,
        committed_record_count=0, last_committed_datapoint_identity=None, created_at_utc=now_text,
        started_at_utc=None, updated_at_utc=now_text, completed_at_utc=None,
        last_error_class=None, last_error_message=None,
    )
    initialize_database(database_path)
    connection = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO ingestion_checkpoints (
                run_id, checkpoint_schema_version, query_fingerprint, source_type, run_status,
                window_start_utc, window_end_utc, account_id, region, namespace, metric_name,
                dimensions_json, resource_identity, metric_unit, period, statistic, pagination_token,
                committed_page_count, committed_record_count, last_committed_datapoint_identity,
                created_at_utc, started_at_utc, updated_at_utc, completed_at_utc,
                last_error_class, last_error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(checkpoint.__dict__.values()),
        )
        connection.commit()
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise TelemetryPersistenceError(f"Unable to create checkpoint: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return checkpoint


def get_checkpoint(database_path: str, run_id: str) -> Checkpoint:
    """Retrieve one checkpoint by run ID."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise DataValidationError("run_id must be a non-empty string.")
    initialize_database(database_path)
    connection = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        row = connection.execute("SELECT * FROM ingestion_checkpoints WHERE run_id = ?", (run_id,)).fetchone()
    except sqlite3.Error as exc:
        raise TelemetryPersistenceError(f"Unable to retrieve checkpoint: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    if row is None:
        raise TelemetryPersistenceError(f"Checkpoint not found: {run_id}")
    return _row_to_checkpoint(row)


def update_checkpoint(
    database_path: str,
    run_id: str,
    *,
    expected_query_fingerprint: str,
    new_status: str | None = None,
    pagination_token: str | None = None,
    clear_pagination_token: bool = False,
    committed_page_count: int | None = None,
    committed_record_count: int | None = None,
    last_committed_datapoint_identity: str | None = None,
    error: BaseException | None = None,
    now: datetime | None = None,
) -> Checkpoint:
    """Update checkpoint progress or state with query binding validation."""
    current = get_checkpoint(database_path, run_id)
    if current.query_fingerprint != expected_query_fingerprint:
        raise DataValidationError("Checkpoint query fingerprint does not match the requested query.")
    target = new_status or current.run_status
    if target not in CHECKPOINT_STATES or (new_status is not None and target not in _VALID_STATE_TRANSITIONS[current.run_status]):
        raise DataValidationError(f"Invalid checkpoint state transition: {current.run_status} -> {target}")
    pages = current.committed_page_count if committed_page_count is None else committed_page_count
    records = current.committed_record_count if committed_record_count is None else committed_record_count
    if isinstance(pages, bool) or not isinstance(pages, int) or pages < current.committed_page_count:
        raise DataValidationError("committed_page_count must be a non-decreasing non-negative integer.")
    if isinstance(records, bool) or not isinstance(records, int) or records < current.committed_record_count:
        raise DataValidationError("committed_record_count must be a non-decreasing non-negative integer.")
    now_text = _utc_text(now or datetime.now(timezone.utc))
    started = current.started_at_utc or (now_text if target == "running" else None)
    completed = now_text if target == "completed" else current.completed_at_utc
    error_class = type(error).__name__ if error is not None else current.last_error_class
    error_message = str(error)[:1000] if error is not None else current.last_error_message
    connection = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """UPDATE ingestion_checkpoints SET run_status=?, pagination_token=?,
                committed_page_count=?, committed_record_count=?,
                last_committed_datapoint_identity=?, started_at_utc=?, updated_at_utc=?,
                completed_at_utc=?, last_error_class=?, last_error_message=? WHERE run_id=?""",
            (target, None if clear_pagination_token else (pagination_token if pagination_token is not None else current.pagination_token),
             pages, records, last_committed_datapoint_identity or current.last_committed_datapoint_identity,
             started, now_text, completed, error_class, error_message, run_id),
        )
        connection.commit()
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise TelemetryPersistenceError(f"Unable to update checkpoint: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return get_checkpoint(database_path, run_id)


def update_checkpoint_in_connection(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    expected_query_fingerprint: str,
    new_status: str,
    pagination_token: str | None,
    committed_page_count: int,
    committed_record_count: int,
    last_committed_datapoint_identity: str | None,
    now: datetime,
) -> Checkpoint:
    """Update checkpoint progress inside an already-open transaction."""
    row = connection.execute("SELECT * FROM ingestion_checkpoints WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise TelemetryPersistenceError(f"Checkpoint not found: {run_id}")
    current = _row_to_checkpoint(row)
    if current.query_fingerprint != expected_query_fingerprint:
        raise DataValidationError("Checkpoint query fingerprint does not match the requested query.")
    if new_status not in CHECKPOINT_STATES or new_status not in _VALID_STATE_TRANSITIONS[current.run_status]:
        raise DataValidationError(f"Invalid checkpoint state transition: {current.run_status} -> {new_status}")
    if committed_page_count < current.committed_page_count or committed_record_count < current.committed_record_count:
        raise DataValidationError("Checkpoint progress counters must not decrease.")
    now_text = _utc_text(now)
    started = current.started_at_utc or (now_text if new_status == "running" else None)
    completed = now_text if new_status == "completed" else current.completed_at_utc
    connection.execute(
        """UPDATE ingestion_checkpoints SET run_status=?, pagination_token=?,
            committed_page_count=?, committed_record_count=?,
            last_committed_datapoint_identity=?, started_at_utc=?, updated_at_utc=?,
            completed_at_utc=? WHERE run_id=?""",
        (new_status, pagination_token, committed_page_count, committed_record_count,
         last_committed_datapoint_identity, started, now_text, completed, run_id),
    )
    row = connection.execute("SELECT * FROM ingestion_checkpoints WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_checkpoint(row)


def list_incomplete_checkpoints(database_path: str) -> list[Checkpoint]:
    """List checkpoints not in terminal completed state."""
    initialize_database(database_path)
    connection = None
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        rows = connection.execute(
            "SELECT * FROM ingestion_checkpoints WHERE run_status <> 'completed' ORDER BY updated_at_utc, run_id",
        ).fetchall()
    except sqlite3.Error as exc:
        raise TelemetryPersistenceError(f"Unable to list incomplete checkpoints: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [_row_to_checkpoint(row) for row in rows]
