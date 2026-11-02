from datetime import datetime, timezone

import pytest

from anomaly_detector.alert_management import (
    Alert,
    SeverityPolicy,
    acknowledge_alert,
    generate_alerts,
    persist_alerts,
    resolve_alert,
    retrieve_alerts,
)
from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.detection.advanced_detector import AdvancedDetectionResult
from anomaly_detector.detection.generic_detector import GenericDetectionResult
from anomaly_detector.utils.exceptions import DataValidationError


def record(value=1.0, metric="cpu", entity="host-a"):
    return TelemetryRecord(datetime(2026, 11, 2, 12, tzinfo=timezone.utc), "csv", metric, value, entity_id=entity)


def test_normal_results_create_no_alerts_and_anomalous_results_create_alerts():
    normal = GenericDetectionResult(record(), False, "normal")
    anomaly = GenericDetectionResult(record(100), True, "above")
    alerts = generate_alerts([normal, anomaly])
    assert len(alerts) == 1
    assert alerts[0].metric_name == "cpu"
    assert alerts[0].severity == "LOW"
    assert alerts[0].status == "NEW"


def test_advanced_severity_levels_are_configurable():
    policy = SeverityPolicy(1, 2, 3)
    results = [AdvancedDetectionResult(record(i), True, "x", "z_score", i, statistical_score=i) for i in [0.5, 1, 2, 3]]
    alerts = generate_alerts(results, policy)
    assert [alert.severity for alert in alerts] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_alert_persistence_filters_duplicates_and_lifecycle(tmp_path):
    path = tmp_path / "alerts.sqlite"
    alert = generate_alerts([GenericDetectionResult(record(100), True, "above")])[0]
    assert persist_alerts(path, [alert]) == 1
    assert persist_alerts(path, [alert]) == 0
    assert retrieve_alerts(path, severity="LOW", status="NEW", metric_name="cpu", entity_id="host-a")[0].alert_id == alert.alert_id
    acknowledged = acknowledge_alert(path, alert.alert_id)
    resolved = resolve_alert(path, acknowledged.alert_id)
    assert resolved.status == "RESOLVED"
    with pytest.raises(DataValidationError):
        acknowledge_alert(path, alert.alert_id)


def test_invalid_policy_and_filters_are_rejected():
    with pytest.raises(DataValidationError):
        SeverityPolicy(3, 2, 1).validate()
    with pytest.raises(DataValidationError):
        retrieve_alerts("does-not-matter.sqlite", severity="INVALID")
