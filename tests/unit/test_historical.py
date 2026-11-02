from datetime import datetime, timedelta, timezone

import pytest

from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.data_ingestion.telemetry_persistence import (
    persist_advanced_detection_results,
    persist_generic_detection_results,
    persist_telemetry_records,
)
from anomaly_detector.detection.advanced_detector import detect_z_scores
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry
from anomaly_detector.historical import (
    export_advanced_detection_csv,
    export_generic_detection_csv,
    export_telemetry_csv,
    retrieve_advanced_detection_history,
    retrieve_generic_detection_history,
    retrieve_telemetry_history,
)
from anomaly_detector.utils.exceptions import DataValidationError


def make_csv(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text(
        "timestamp,host,cpu,latency,label\n"
        "2026-11-02T12:00:00+02:00,a,10,100,false\n"
        "2026-11-02T12:01:00+02:00,b,90,250,true\n",
        encoding="utf-8",
    )
    return path


def seed(tmp_path):
    records = load_infrastructure_csv(make_csv(tmp_path), metric_columns=["cpu", "latency"], entity_column="host", label_column="label")
    database = tmp_path / "history.sqlite"
    persist_telemetry_records(database, records)
    generic = detect_numeric_telemetry(records, {"cpu": NumericThreshold(0, 80), "latency": NumericThreshold(0, 200)})
    persist_generic_detection_results(database, generic)
    advanced = detect_z_scores(records, min_samples=2, z_threshold=1)
    persist_advanced_detection_results(database, advanced)
    return database, records, generic, advanced


def test_telemetry_filters_and_utc_order(tmp_path):
    database, records, _, _ = seed(tmp_path)
    history = retrieve_telemetry_history(database, start=datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc), end=datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc))
    assert len(history) == 2
    assert history[0].timestamp.tzinfo == timezone.utc
    assert len(retrieve_telemetry_history(database, metric_name="cpu", entity_id="a", source_type="csv")) == 1
    assert len(retrieve_telemetry_history(database, anomaly=True)) == 2


def test_detection_filters(tmp_path):
    database, _, _, _ = seed(tmp_path)
    assert len(retrieve_generic_detection_history(database, anomaly=True, metric_name="cpu")) == 1
    assert len(retrieve_advanced_detection_history(database, detection_method="z_score", metric_name="latency")) == 2
    assert len(retrieve_advanced_detection_history(database, anomaly=False)) >= 1


def test_empty_and_invalid_filters(tmp_path):
    database, _, _, _ = seed(tmp_path)
    assert retrieve_telemetry_history(database, metric_name="missing") == []
    with pytest.raises(DataValidationError):
        retrieve_telemetry_history(database, start=datetime(2026, 11, 2, 10), end=datetime(2026, 11, 2, 9))
    with pytest.raises(DataValidationError):
        retrieve_generic_detection_history(database, anomaly="false")
    with pytest.raises(DataValidationError):
        retrieve_advanced_detection_history(database, detection_method=" ")
    assert retrieve_telemetry_history(database, metric_name="' OR 1=1 --") == []


def test_exports_headers_timestamps_and_empty_results(tmp_path):
    database, records, generic, advanced = seed(tmp_path)
    telemetry_path = tmp_path / "telemetry.csv"
    generic_path = tmp_path / "generic.csv"
    advanced_path = tmp_path / "advanced.csv"
    empty_path = tmp_path / "empty.csv"
    export_telemetry_csv(records, telemetry_path)
    export_generic_detection_csv(generic, generic_path)
    export_advanced_detection_csv(advanced, advanced_path)
    export_telemetry_csv([], empty_path)
    assert telemetry_path.read_text(encoding="utf-8").splitlines()[0].startswith("timestamp,source_type")
    assert "2026-11-02T10:00:00+00:00" in telemetry_path.read_text(encoding="utf-8")
    assert generic_path.read_text(encoding="utf-8").splitlines()[0].startswith("timestamp,source_type")
    assert "detection_method" in advanced_path.read_text(encoding="utf-8").splitlines()[0]
    assert empty_path.read_text(encoding="utf-8").startswith("timestamp,source_type")


def test_export_rejects_directory(tmp_path):
    with pytest.raises(Exception):
        export_telemetry_csv([], tmp_path)
