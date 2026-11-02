from datetime import datetime, timezone
import sqlite3

import pytest

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import (
    DetectionPersistenceResult,
    TelemetryPersistenceError,
    persist_generic_detection_results,
    retrieve_generic_detection_results,
)
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry
from anomaly_detector.utils.exceptions import DataValidationError


def record(metric_name="CPUUtilization", value=50.0, hour=12):
    return TelemetryRecord(
        timestamp=datetime(2026, 11, 2, hour, tzinfo=timezone.utc),
        source_type="aws_cloudwatch", metric_name=metric_name, metric_value=value,
        namespace="AWS/EC2", dimensions={"InstanceId": "i-test"},
        account_id="123", region="us-east-1", metric_unit="Percent",
    )


def test_save_and_retrieve_normal_and_anomalous_results(tmp_path):
    path = tmp_path / "results.sqlite"
    results = detect_numeric_telemetry(
        [record(value=50, hour=12), record(value=150, hour=13)],
        {"CPUUtilization": NumericThreshold(0, 100)},
    )
    assert persist_generic_detection_results(path, results) == DetectionPersistenceResult(2, 0)
    restored = retrieve_generic_detection_results(path)
    assert [item.predicted_is_anomaly for item in restored] == [False, True]
    assert [item.record.metric_value for item in restored] == [50.0, 150.0]
    assert restored[1].explanation.startswith("above_upper_threshold")


def test_filters_by_status_and_metric(tmp_path):
    path = tmp_path / "results.sqlite"
    results = detect_numeric_telemetry(
        [record(value=50), record(metric_name="Latency", value=200)],
        {"CPUUtilization": NumericThreshold(0, 100), "Latency": NumericThreshold(0, 100)},
    )
    persist_generic_detection_results(path, results)
    assert len(retrieve_generic_detection_results(path, anomaly=False)) == 1
    assert len(retrieve_generic_detection_results(path, anomaly=True)) == 1
    assert len(retrieve_generic_detection_results(path, metric_name="Latency")) == 1


def test_duplicate_result_is_not_inserted_twice(tmp_path):
    path = tmp_path / "results.sqlite"
    result = detect_numeric_telemetry([record()], {"CPUUtilization": NumericThreshold(0, 100)})[0]
    assert persist_generic_detection_results(path, [result]) == DetectionPersistenceResult(1, 0)
    assert persist_generic_detection_results(path, [result]) == DetectionPersistenceResult(0, 1)
    assert len(retrieve_generic_detection_results(path)) == 1


def test_empty_and_invalid_inputs(tmp_path):
    path = tmp_path / "results.sqlite"
    assert persist_generic_detection_results(path, []) == DetectionPersistenceResult(0, 0)
    assert retrieve_generic_detection_results(path) == []
    with pytest.raises(DataValidationError):
        persist_generic_detection_results(path, [object()])
    with pytest.raises(DataValidationError):
        retrieve_generic_detection_results(path, anomaly="false")
    with pytest.raises(DataValidationError):
        retrieve_generic_detection_results(path, metric_name=" ")


def test_transaction_rolls_back_on_database_failure(tmp_path):
    path = tmp_path / "results.sqlite"
    first = detect_numeric_telemetry([record(value=50)], {"CPUUtilization": NumericThreshold(0, 100)})[0]
    second = detect_numeric_telemetry([record(metric_name="FAIL", value=50)], {"FAIL": NumericThreshold(0, 100)})[0]
    persist_generic_detection_results(path, [first])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_generic BEFORE INSERT ON generic_detection_results "
            "WHEN NEW.metric_name = 'FAIL' BEGIN SELECT RAISE(ABORT, 'forced failure'); END;",
        )
    with pytest.raises(TelemetryPersistenceError):
        persist_generic_detection_results(path, [first, second])
    assert len(retrieve_generic_detection_results(path)) == 1
