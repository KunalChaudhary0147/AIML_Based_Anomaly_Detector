"""Dataset catalog and lifecycle operations for uploaded telemetry."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import (
    _connect,
    _validate_schema,
    initialize_database,
    persist_telemetry_records_in_connection,
)
from anomaly_detector.utils.exceptions import DataValidationError

LEGACY_DATASET_ID = "legacy-unassigned"
LEGACY_DATASET_FILENAME = "Legacy unassigned telemetry"


def _backfill_legacy_dataset_in_connection(connection: sqlite3.Connection) -> bool:
    """Assign unowned telemetry to one explicit dataset without inventing provenance."""
    telemetry = connection.execute(
        "SELECT datapoint_identity FROM telemetry_records WHERE dataset_id IS NULL"
    ).fetchall()
    if not telemetry:
        return False
    counts = connection.execute(
        """SELECT COUNT(*), COUNT(DISTINCT entity_id), COUNT(DISTINCT metric_name),
                  MIN(timestamp_utc), MAX(timestamp_utc)
           FROM telemetry_records WHERE dataset_id IS NULL"""
    ).fetchone()
    connection.execute(
        """INSERT OR IGNORE INTO datasets (
            dataset_id, filename, source_type, upload_timestamp_utc, record_count,
            entity_count, metric_count, start_timestamp_utc, end_timestamp_utc,
            content_hash, status
        ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?)""",
        (
            LEGACY_DATASET_ID,
            LEGACY_DATASET_FILENAME,
            "legacy_unassigned",
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            "legacy_unassigned",
        ),
    )
    connection.execute(
        "UPDATE telemetry_records SET dataset_id = ? WHERE dataset_id IS NULL",
        (LEGACY_DATASET_ID,),
    )
    for table in ("generic_detection_results", "advanced_detection_results"):
        connection.execute(
            f"""UPDATE {table}
                SET dataset_id = (
                    SELECT dataset_id FROM telemetry_records
                    WHERE telemetry_records.datapoint_identity = {table}.datapoint_identity
                )
                WHERE dataset_id IS NULL"""
        )
    identity_to_dataset = {
        row["datapoint_identity"]: row["dataset_id"]
        for row in connection.execute(
            "SELECT datapoint_identity, dataset_id FROM telemetry_records WHERE dataset_id = ?",
            (LEGACY_DATASET_ID,),
        )
    }
    for row in connection.execute(
        "SELECT alert_id, detection_identity, detection_method FROM anomaly_alerts WHERE dataset_id IS NULL"
    ).fetchall():
        for datapoint_identity, dataset_id in identity_to_dataset.items():
            detection_identity = json.dumps(
                json.loads(datapoint_identity) + [row["detection_method"]],
                separators=(",", ":"),
                ensure_ascii=True,
            )
            if detection_identity == row["detection_identity"]:
                connection.execute(
                    "UPDATE anomaly_alerts SET dataset_id = ? WHERE alert_id = ?",
                    (dataset_id, row["alert_id"]),
                )
                break
    connection.execute(
        """UPDATE datasets SET record_count = ?, entity_count = ?, metric_count = ?,
               start_timestamp_utc = ?, end_timestamp_utc = ?
           WHERE dataset_id = ?""",
        (*counts, LEGACY_DATASET_ID),
    )
    return True


def backfill_legacy_dataset(database_path: str | Path) -> bool:
    """Idempotently recover unassigned telemetry as one explicit legacy dataset."""
    initialize_database(database_path)
    connection = _connect(database_path)
    try:
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        changed = _backfill_legacy_dataset_in_connection(connection)
        connection.commit()
        return changed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _validate_ids(dataset_ids):
    if not isinstance(dataset_ids, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in dataset_ids):
        raise DataValidationError("dataset_ids must be a list of non-empty strings.")
    return list(dict.fromkeys(dataset_ids))


def ingest_dataset(database_path: str | Path, filename: str, content: bytes, records: list[TelemetryRecord]) -> dict:
    if not filename or not isinstance(filename, str):
        raise DataValidationError("filename must be a non-empty string.")
    dataset_id = str(uuid4())
    upload_time = datetime.now(timezone.utc).isoformat()
    timestamps = [record.timestamp for record in records]
    metadata = (
        dataset_id, filename, "csv", upload_time, len(records),
        len({record.entity_id for record in records}), len({record.metric_name for record in records}),
        min(timestamps).isoformat() if timestamps else None, max(timestamps).isoformat() if timestamps else None,
        hashlib.sha256(content).hexdigest(), "completed",
    )
    initialize_database(database_path)
    connection = _connect(database_path)
    try:
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO datasets (
                dataset_id, filename, source_type, upload_timestamp_utc, record_count,
                entity_count, metric_count, start_timestamp_utc, end_timestamp_utc,
                content_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            metadata,
        )
        result = persist_telemetry_records_in_connection(
            connection, [replace(record, dataset_id=dataset_id) for record in records],
        )
        if result.inserted != len(records):
            raise DataValidationError("Dataset upload could not persist every telemetry record.")
        connection.commit()
        return get_dataset(database_path, dataset_id)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def list_datasets(database_path, *, search=None, source_type=None):
    initialize_database(database_path)
    connection = _connect(database_path)
    try:
        clauses, params = [], []
        if search is not None:
            clauses.append("filename LIKE ?")
            params.append(f"%{search}%")
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            "SELECT d.*, (SELECT COUNT(*) FROM datasets same WHERE same.content_hash = d.content_hash) > 1 AS is_duplicate FROM datasets d"
            + where + " ORDER BY upload_timestamp_utc DESC", params,
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = [
                metric[0] for metric in connection.execute(
                    "SELECT DISTINCT metric_name FROM telemetry_records WHERE dataset_id = ? ORDER BY metric_name",
                    (item["dataset_id"],),
                ).fetchall()
            ]
            result.append(item)
        return result
    finally:
        connection.close()


def get_dataset(database_path, dataset_id):
    initialize_database(database_path)
    connection = _connect(database_path)
    try:
        row = connection.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return dict(row)
    finally:
        connection.close()


def preview_dataset(database_path, dataset_id):
    get_dataset(database_path, dataset_id)
    connection = _connect(database_path)
    try:
        rows = connection.execute(
            "SELECT * FROM telemetry_records WHERE dataset_id = ? ORDER BY timestamp_utc, rowid LIMIT 10",
            (dataset_id,),
        ).fetchall()
        from anomaly_detector.historical import _telemetry_from_row
        return [_telemetry_from_row(row) for row in rows]
    finally:
        connection.close()


def quality_dataset(database_path, dataset_id):
    """Return stored-data quality metrics; invalid-record history is unavailable.

    Duplicate records are defined as rows sharing a telemetry datapoint identity.
    The primary key prevents such duplicates in the current store, so the value
    is zero for persisted data.
    """
    get_dataset(database_path, dataset_id)
    connection = _connect(database_path)
    try:
        row = connection.execute(
            """SELECT COUNT(*) total_records, COUNT(DISTINCT entity_id) entity_count,
                      COUNT(DISTINCT metric_name) metric_count, MIN(timestamp_utc) start_timestamp_utc,
                      MAX(timestamp_utc) end_timestamp_utc,
                      SUM(timestamp_utc IS NULL) missing_timestamps,
                      SUM(metric_value IS NULL) missing_metric_values
               FROM telemetry_records WHERE dataset_id = ?""",
            (dataset_id,),
        ).fetchone()
        result = dict(row)
        result["invalid_records"] = None
        result["duplicate_records"] = 0
        result["duplicate_definition"] = "Rows sharing the same telemetry datapoint identity."
        result["unique_entities"] = result["entity_count"]
        result["available_metrics"] = [
            metric[0] for metric in connection.execute(
                "SELECT DISTINCT metric_name FROM telemetry_records WHERE dataset_id = ? ORDER BY metric_name",
                (dataset_id,),
            ).fetchall()
        ]
        result["invalid_timestamps"] = 0
        result["invalid_numerical_values"] = 0
        return result
    finally:
        connection.close()


def delete_datasets(database_path, dataset_ids):
    ids = _validate_ids(dataset_ids)
    initialize_database(database_path)
    connection = _connect(database_path)
    try:
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in ids)
        existing = [row[0] for row in connection.execute(f"SELECT dataset_id FROM datasets WHERE dataset_id IN ({placeholders})", ids)]
        if set(existing) != set(ids):
            missing = sorted(set(ids) - set(existing))
            raise KeyError(", ".join(missing))
        counts = {}
        for table in ("telemetry_records", "generic_detection_results", "advanced_detection_results", "anomaly_alerts"):
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE dataset_id IN ({placeholders})", ids).fetchone()[0]
            connection.execute(f"DELETE FROM {table} WHERE dataset_id IN ({placeholders})", ids)
        connection.execute(f"DELETE FROM datasets WHERE dataset_id IN ({placeholders})", ids)
        counts["datasets"] = len(ids)
        connection.commit()
        return counts
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
