from datetime import datetime, timezone
from importlib import import_module

from fastapi import FastAPI
from fastapi.testclient import TestClient

from anomaly_detector.api import create_app
from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import persist_telemetry_records


def test_documented_uvicorn_target_exposes_asgi_app():
    module = import_module("anomaly_detector.api.app")

    assert isinstance(module.app, FastAPI)


def seed_database(path):
    persist_telemetry_records(path, [
        TelemetryRecord(datetime(2026, 11, 2, 12, tzinfo=timezone.utc), "csv", "cpu", 95.0, entity_id="host-a"),
        TelemetryRecord(datetime(2026, 11, 2, 12, tzinfo=timezone.utc), "csv", "cpu", 20.0, entity_id="host-b"),
    ])


def test_health_csv_ingestion_and_telemetry(tmp_path):
    database = tmp_path / "api.sqlite"
    client = TestClient(create_app(database))
    health = client.get("/health")
    assert health.status_code == 200
    csv_data = b"timestamp,host,cpu\n2026-11-02T12:00:00Z,a,95\n"
    uploaded = client.post("/api/v1/ingestion/csv", files={"file": ("infra.csv", csv_data, "text/csv")}, data={"metric_columns": "cpu", "entity_column": "host"})
    assert uploaded.status_code == 200
    seed_database(database)
    telemetry = client.get("/api/v1/telemetry", params={"metric_name": "cpu"})
    assert telemetry.status_code == 200
    assert len(telemetry.json()) == 2


def test_detection_alert_configuration_and_exports(tmp_path):
    database = tmp_path / "api.sqlite"
    seed_database(database)
    client = TestClient(create_app(database))
    detection = client.post("/api/v1/detection/generic", json={"thresholds": {"cpu": {"lower": 0, "upper": 80}}})
    assert detection.status_code == 200
    assert detection.json()["alerts"] == 1
    history = client.get("/api/v1/detection/generic/results", params={"anomaly": "true"})
    assert history.status_code == 200
    alerts = client.get("/api/v1/alerts", params={"status": "NEW"})
    assert alerts.status_code == 200
    alert_id = alerts.json()[0]["alert_id"]
    assert client.post(f"/api/v1/alerts/{alert_id}/acknowledge").status_code == 200
    assert client.post(f"/api/v1/alerts/{alert_id}/resolve").status_code == 200
    configuration = client.post("/api/v1/configurations", json={"name": "default", "thresholds": {"cpu": {"lower": 0, "upper": 80}}})
    assert configuration.status_code == 200
    assert client.get("/api/v1/configurations/default").status_code == 200
    export = client.get("/api/v1/export/detection/generic")
    assert export.status_code == 200
    assert "predicted_is_anomaly" in export.text


def test_invalid_api_input_returns_client_error(tmp_path):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    response = client.post("/api/v1/detection/generic", json={"thresholds": {"cpu": {"lower": 10, "upper": 0}}})
    assert response.status_code in {400, 422}
