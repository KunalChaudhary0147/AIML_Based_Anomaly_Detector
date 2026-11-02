"""Historical telemetry and detection-result querying and CSV export."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import _connect, _validate_schema, initialize_database
from anomaly_detector.detection.advanced_detector import AdvancedDetectionResult
from anomaly_detector.detection.generic_detector import GenericDetectionResult
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError


def _utc_filter(value: datetime | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError(f"{name} must be a timezone-aware datetime.")
    return value.astimezone(timezone.utc).isoformat()


def _query_filters(start, end, metric_name, entity_id, source_type, anomaly):
    start_text = _utc_filter(start, "start")
    end_text = _utc_filter(end, "end")
    if start_text is not None and end_text is not None and start_text > end_text:
        raise DataValidationError("start must not be later than end.")
    for value, name in ((metric_name, "metric_name"), (entity_id, "entity_id"), (source_type, "source_type")):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DataValidationError(f"{name} must be a non-empty string or None.")
    if anomaly is not None and not isinstance(anomaly, bool):
        raise DataValidationError("anomaly must be Boolean or None.")
    clauses: list[str] = []
    parameters: list[Any] = []
    if start_text is not None:
        clauses.append("timestamp_utc >= ?")
        parameters.append(start_text)
    if end_text is not None:
        clauses.append("timestamp_utc <= ?")
        parameters.append(end_text)
    for column, value in (("metric_name", metric_name), ("entity_id", entity_id), ("source_type", source_type)):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    if anomaly is not None:
        clauses.append("is_anomaly = ?")
        parameters.append(int(anomaly))
    return clauses, parameters


def _connection(database_path):
    initialize_database(database_path)
    connection = _connect(database_path)
    _validate_schema(connection)
    return connection


def _telemetry_from_row(row) -> TelemetryRecord:
    import json
    return TelemetryRecord(
        timestamp=datetime.fromisoformat(row["timestamp_utc"]), source_type=row["source_type"],
        metric_name=row["metric_name"], metric_value=row["metric_value"], metric_unit=row["metric_unit"],
        entity_id=row["entity_id"], is_anomaly=None if row["is_anomaly"] is None else bool(row["is_anomaly"]),
        namespace=row["namespace"], dimensions=json.loads(row["dimensions_json"]),
        resource_arn=row["resource_arn"], account_id=row["account_id"], region=row["region"],
        period=row["period"], statistic=row["statistic"],
    )


def retrieve_telemetry_history(database_path, *, start=None, end=None, metric_name=None, entity_id=None, source_type=None, anomaly=None):
    """Retrieve telemetry records in deterministic UTC timestamp order."""
    clauses, parameters = _query_filters(start, end, metric_name, entity_id, source_type, anomaly)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    connection = None
    try:
        connection = _connection(database_path)
        rows = connection.execute("SELECT * FROM telemetry_records" + where + " ORDER BY timestamp_utc, rowid", parameters).fetchall()
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataIngestionError(f"Unable to retrieve telemetry history: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [_telemetry_from_row(row) for row in rows]


def _retrieve_detection_rows(database_path, table, where, parameters, advanced):
    import json
    connection = None
    try:
        connection = _connection(database_path)
        rows = connection.execute("SELECT * FROM " + table + where + " ORDER BY timestamp_utc, rowid", parameters).fetchall()
    except Exception as exc:
        raise DataIngestionError(f"Unable to retrieve detection history: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    results = []
    for row in rows:
        record = TelemetryRecord(
            timestamp=datetime.fromisoformat(row["timestamp_utc"]), source_type=row["source_type"],
            metric_name=row["metric_name"], metric_value=row["metric_value"], entity_id=row["entity_id"],
            namespace=row["namespace"], dimensions=json.loads(row["dimensions_json"]),
            account_id=row["account_id"], region=row["region"], resource_arn=row["resource_arn"],
            metric_unit=row["metric_unit"], period=row["period"], statistic=row["statistic"],
        )
        if advanced:
            results.append(AdvancedDetectionResult(record, bool(row["predicted_is_anomaly"]), row["explanation"], row["detection_method"], row["metric_value"], row["reference_value"], row["statistical_score"]))
        else:
            results.append(GenericDetectionResult(record, bool(row["predicted_is_anomaly"]), row["explanation"]))
    return results


def _detection_filters(filters, advanced):
    start, end = filters.pop("start", None), filters.pop("end", None)
    metric_name, entity_id = filters.pop("metric_name", None), filters.pop("entity_id", None)
    source_type, anomaly = filters.pop("source_type", None), filters.pop("anomaly", None)
    method = filters.pop("detection_method", None) if advanced else None
    if filters:
        raise DataValidationError(f"Unsupported filters: {sorted(filters)}")
    clauses, parameters = _query_filters(start, end, metric_name, entity_id, source_type, None)
    if anomaly is not None:
        if not isinstance(anomaly, bool):
            raise DataValidationError("anomaly must be Boolean or None.")
        clauses.append("predicted_is_anomaly = ?")
        parameters.append(int(anomaly))
    if method is not None:
        if not isinstance(method, str) or not method.strip():
            raise DataValidationError("detection_method must be a non-empty string or None.")
        clauses.append("detection_method = ?")
        parameters.append(method)
    return clauses, parameters


def retrieve_generic_detection_history(database_path, **filters):
    """Retrieve generic threshold results with telemetry filters."""
    clauses, parameters = _detection_filters(filters, False)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return _retrieve_detection_rows(database_path, "generic_detection_results", where, parameters, False)


def retrieve_advanced_detection_history(database_path, **filters):
    """Retrieve advanced results with telemetry and method filters."""
    clauses, parameters = _detection_filters(filters, True)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return _retrieve_detection_rows(database_path, "advanced_detection_results", where, parameters, True)


def _export_rows(destination, headers, rows):
    path = Path(destination)
    if path.exists() and path.is_dir():
        raise DataIngestionError(f"Export destination is a directory: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent, delete=False, encoding="utf-8", newline="")
        temporary = Path(handle.name)
        handle.close()
        try:
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(headers)
                writer.writerows(rows)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    except OSError as exc:
        raise DataIngestionError(f"Unable to export CSV: {path}") from exc


def export_telemetry_csv(records: Iterable[TelemetryRecord], destination) -> None:
    """Export telemetry records as UTC ISO-8601 CSV."""
    _export_rows(destination, ["timestamp", "source_type", "metric_name", "metric_value", "entity_id", "is_anomaly"], [[r.timestamp.isoformat(), r.source_type, r.metric_name, r.metric_value, r.entity_id or "", "" if r.is_anomaly is None else r.is_anomaly] for r in records])


def export_generic_detection_csv(results: Iterable[GenericDetectionResult], destination) -> None:
    _export_rows(destination, ["timestamp", "source_type", "metric_name", "metric_value", "predicted_is_anomaly", "explanation"], [[r.record.timestamp.isoformat(), r.record.source_type, r.record.metric_name, r.record.metric_value, r.predicted_is_anomaly, r.explanation] for r in results])


def export_advanced_detection_csv(results: Iterable[AdvancedDetectionResult], destination) -> None:
    _export_rows(destination, ["timestamp", "source_type", "metric_name", "metric_value", "predicted_is_anomaly", "detection_method", "reference_value", "statistical_score", "explanation"], [[r.record.timestamp.isoformat(), r.record.source_type, r.record.metric_name, r.record.metric_value, r.predicted_is_anomaly, r.detection_method, r.reference_value, r.statistical_score, r.explanation] for r in results])
