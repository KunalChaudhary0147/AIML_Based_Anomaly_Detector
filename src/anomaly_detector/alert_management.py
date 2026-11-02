"""Local alert generation, persistence, and lifecycle management."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from anomaly_detector.data_ingestion.telemetry_persistence import _connect, _validate_schema, _identity_json, initialize_database
from anomaly_detector.detection.advanced_detector import AdvancedDetectionResult
from anomaly_detector.detection.generic_detector import GenericDetectionResult
from anomaly_detector.utils.exceptions import DataValidationError

SEVERITIES = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
ALERT_STATUSES = frozenset({"NEW", "ACKNOWLEDGED", "RESOLVED"})


@dataclass(frozen=True)
class SeverityPolicy:
    """Score thresholds for advanced results; generic anomalies default LOW."""

    medium_score: float = 2.0
    high_score: float = 3.0
    critical_score: float = 5.0

    def validate(self) -> None:
        if not (0 < self.medium_score < self.high_score < self.critical_score):
            raise DataValidationError("Severity scores must satisfy 0 < medium < high < critical.")


@dataclass(frozen=True)
class Alert:
    """Anomaly alert associated with one detection result."""

    alert_id: str
    detection_identity: str
    detection_method: str
    metric_name: str
    entity_id: str | None
    source_type: str
    observed_value: float
    severity: str
    explanation: str
    status: str
    created_at_utc: str
    updated_at_utc: str


def _method(result) -> str:
    return getattr(result, "detection_method", "generic_threshold")


def _severity(result, policy: SeverityPolicy) -> str:
    score = getattr(result, "statistical_score", None)
    if score is None:
        return "LOW"
    if score >= policy.critical_score:
        return "CRITICAL"
    if score >= policy.high_score:
        return "HIGH"
    if score >= policy.medium_score:
        return "MEDIUM"
    return "LOW"


def generate_alerts(results: Iterable[GenericDetectionResult | AdvancedDetectionResult], policy: SeverityPolicy = SeverityPolicy()) -> list[Alert]:
    """Generate alerts only for anomalous detection results."""
    policy.validate()
    alerts: list[Alert] = []
    now = datetime.now(timezone.utc).isoformat()
    for result in results:
        if not isinstance(result, (GenericDetectionResult, AdvancedDetectionResult)):
            raise DataValidationError("results must contain supported detection results.")
        if not result.predicted_is_anomaly:
            continue
        record = result.record
        method = _method(result)
        alerts.append(Alert(
            alert_id=str(uuid4()),
            detection_identity=_identity_json(record.datapoint_identity + (method,)),
            detection_method=method,
            metric_name=record.metric_name,
            entity_id=record.entity_id,
            source_type=record.source_type,
            observed_value=float(record.metric_value),
            severity=_severity(result, policy),
            explanation=result.explanation,
            status="NEW",
            created_at_utc=now,
            updated_at_utc=now,
        ))
    return alerts


def persist_alerts(database_path, alerts: Iterable[Alert]) -> int:
    """Persist alerts transactionally; duplicate identities are ignored."""
    alert_list = list(alerts)
    if any(not isinstance(alert, Alert) for alert in alert_list):
        raise DataValidationError("alerts must contain only Alert objects.")
    initialize_database(database_path)
    connection = None
    inserted = 0
    try:
        connection = _connect(database_path)
        _validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        for alert in alert_list:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO anomaly_alerts (
                    alert_id, detection_identity, detection_method, metric_name, entity_id,
                    source_type, observed_value, severity, explanation, status,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (alert.alert_id, alert.detection_identity, alert.detection_method, alert.metric_name,
                 alert.entity_id, alert.source_type, alert.observed_value, alert.severity,
                 alert.explanation, alert.status, alert.created_at_utc, alert.updated_at_utc),
            )
            inserted += int(cursor.rowcount == 1)
        connection.commit()
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        raise DataValidationError(f"Unable to persist alerts; transaction rolled back: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return inserted


def retrieve_alerts(database_path, *, severity=None, status=None, metric_name=None, entity_id=None) -> list[Alert]:
    """Retrieve alerts with optional filters in creation order."""
    for value, name in ((severity, "severity"), (status, "status"), (metric_name, "metric_name"), (entity_id, "entity_id")):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DataValidationError(f"{name} must be a non-empty string or None.")
    if severity is not None and severity not in SEVERITIES:
        raise DataValidationError("Invalid alert severity.")
    if status is not None and status not in ALERT_STATUSES:
        raise DataValidationError("Invalid alert status.")
    clauses, params = [], []
    for column, value in (("severity", severity), ("status", status), ("metric_name", metric_name), ("entity_id", entity_id)):
        if value is not None:
            clauses.append(f"{column} = ?"); params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    connection = None
    try:
        connection = _connect(database_path); _validate_schema(connection)
        rows = connection.execute("SELECT * FROM anomaly_alerts" + where + " ORDER BY created_at_utc, rowid", params).fetchall()
    except sqlite3.Error as exc:
        raise DataValidationError(f"Unable to retrieve alerts: {exc}") from exc
    finally:
        if connection is not None: connection.close()
    return [Alert(**dict(row)) for row in rows]


def _transition(database_path, alert_id: str, expected: str, target: str) -> Alert:
    if target not in ALERT_STATUSES or expected == "RESOLVED" or (expected == "NEW" and target != "ACKNOWLEDGED") or (expected == "ACKNOWLEDGED" and target != "RESOLVED"):
        raise DataValidationError(f"Invalid alert transition: {expected} -> {target}")
    connection = None
    try:
        connection = _connect(database_path); _validate_schema(connection); connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM anomaly_alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        if row is None: raise DataValidationError(f"Alert not found: {alert_id}")
        if row["status"] != expected: raise DataValidationError(f"Invalid alert transition: {row['status']} -> {target}")
        now = datetime.now(timezone.utc).isoformat()
        connection.execute("UPDATE anomaly_alerts SET status = ?, updated_at_utc = ? WHERE alert_id = ?", (target, now, alert_id)); connection.commit()
        row = connection.execute("SELECT * FROM anomaly_alerts WHERE alert_id = ?", (alert_id,)).fetchone()
    except sqlite3.Error as exc:
        if connection is not None: connection.rollback()
        raise DataValidationError(f"Unable to update alert; transaction rolled back: {exc}") from exc
    finally:
        if connection is not None: connection.close()
    return Alert(**dict(row))


def acknowledge_alert(database_path, alert_id: str) -> Alert:
    return _transition(database_path, alert_id, "NEW", "ACKNOWLEDGED")


def resolve_alert(database_path, alert_id: str) -> Alert:
    return _transition(database_path, alert_id, "ACKNOWLEDGED", "RESOLVED")
