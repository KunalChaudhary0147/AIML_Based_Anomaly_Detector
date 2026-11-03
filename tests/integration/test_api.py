from datetime import datetime, timezone
from importlib import import_module

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from anomaly_detector.api import create_app
from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import persist_telemetry_records


FRONTEND_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


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
    assert len(telemetry.json()) == 3


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


def test_empty_overview_and_configuration_lifecycle(tmp_path):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    overview = client.get("/api/v1/overview")
    assert overview.status_code == 200
    assert overview.json()["totals"] == {
        "telemetry_records": 0,
        "anomalies": 0,
        "normal_observations": 0,
        "anomalous_observations": 0,
        "total_metrics": 0,
        "active_alerts": 0,
        "critical_alerts": 0,
        "monitored_entities": 0,
    }
    assert overview.json()["observation_split"] == []
    saved = client.post(
        "/api/v1/configurations",
        json={"name": "temporary", "thresholds": {"cpu": {"lower": 0, "upper": 80}}},
    )
    assert saved.status_code == 200
    assert client.get("/api/v1/configurations").json()[0]["name"] == "temporary"
    assert client.delete("/api/v1/configurations/temporary").status_code == 200
    assert client.get("/api/v1/configurations").json() == []


def test_overview_classification_and_explicit_empty_dataset_filter(tmp_path):
    database = tmp_path / "api.sqlite"
    persist_telemetry_records(database, [
        TelemetryRecord(datetime(2026, 11, 2, 12, tzinfo=timezone.utc), "csv", "cpu", 20.0, entity_id="host-a", is_anomaly=False),
        TelemetryRecord(datetime(2026, 11, 2, 13, tzinfo=timezone.utc), "csv", "cpu", 95.0, entity_id="host-a", is_anomaly=True),
    ])
    client = TestClient(create_app(database))
    overview = client.get("/api/v1/overview").json()
    assert overview["totals"]["normal_observations"] == 1
    assert overview["totals"]["anomalous_observations"] == 1
    assert overview["observation_split"][0]["normal"] == 1
    assert overview["observation_split"][0]["anomalous"] == 1
    filtered = client.get("/api/v1/telemetry", params=[("dataset_ids", "__none__")])
    assert filtered.status_code == 200
    assert filtered.json() == []


def test_invalid_api_input_returns_client_error(tmp_path):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    response = client.post("/api/v1/detection/generic", json={"thresholds": {"cpu": {"lower": 10, "upper": 0}}})
    assert response.status_code in {400, 422}


@pytest.mark.parametrize("origin", FRONTEND_ORIGINS)
def test_frontend_origins_receive_cors_header(tmp_path, origin):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
