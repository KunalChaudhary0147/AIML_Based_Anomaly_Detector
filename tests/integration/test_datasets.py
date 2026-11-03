from fastapi.testclient import TestClient

from anomaly_detector.api import create_app
from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.data_ingestion.telemetry_persistence import persist_telemetry_records
from datetime import datetime, timezone


CSV = b"timestamp,host,cpu,latency\n2026-11-02T12:00:00Z,a,95,10\n2026-11-02T12:01:00Z,b,20,20\n"


def upload(client, filename="metrics.csv", content=CSV):
    response = client.post(
        "/api/v1/ingestion/csv",
        files={"file": (filename, content, "text/csv")},
        data={"metric_columns": "cpu,latency", "entity_column": "host"},
    )
    assert response.status_code == 200
    return response.json()["dataset_id"]


def test_upload_catalog_preview_quality_and_duplicate_hash(tmp_path):
    client = TestClient(create_app(tmp_path / "datasets.sqlite"))
    first = upload(client)
    second = upload(client)
    datasets = client.get("/api/v1/datasets").json()
    assert first != second
    assert len(datasets) == 2
    assert all(item["is_duplicate"] for item in datasets)
    details = client.get(f"/api/v1/datasets/{first}").json()
    assert details["record_count"] == 4
    assert details["entity_count"] == 2
    assert details["metric_count"] == 2
    assert len(client.get(f"/api/v1/datasets/{first}/preview").json()) == 4
    quality = client.get(f"/api/v1/datasets/{first}/quality").json()
    assert quality["total_records"] == 4
    assert quality["missing_timestamps"] == 0
    assert quality["invalid_records"] is None


def test_dataset_filtering_detection_and_transactional_delete(tmp_path):
    client = TestClient(create_app(tmp_path / "datasets.sqlite"))
    first = upload(client)
    second = upload(client, content=CSV.replace(b"95", b"75"))
    response = client.post(
        "/api/v1/detection/generic",
        json={"thresholds": {"cpu": {"lower": 0, "upper": 80}}, "dataset_ids": [first]},
    )
    assert response.status_code == 200
    assert response.json()["records"] == 4
    assert all(item["dataset_id"] == first for item in client.get("/api/v1/alerts", params=[("dataset_ids", first)]).json())
    assert len(client.get("/api/v1/telemetry", params=[("dataset_ids", first)]).json()) == 4
    assert client.delete(f"/api/v1/datasets/{first}").status_code == 200
    assert client.get(f"/api/v1/datasets/{first}").status_code == 404
    assert len(client.get("/api/v1/telemetry", params=[("dataset_ids", second)]).json()) == 4


def test_invalid_dataset_filter_is_rejected(tmp_path):
    client = TestClient(create_app(tmp_path / "datasets.sqlite"))
    response = client.get("/api/v1/telemetry", params=[("dataset_ids", "missing")])
    assert response.status_code == 400


def test_legacy_records_are_not_recovered_automatically(tmp_path):
    database = tmp_path / "legacy.sqlite"
    records = [
        TelemetryRecord(
            datetime(2026, 11, 2, 12, tzinfo=timezone.utc),
            "csv",
            "cpu",
            20.0,
            entity_id="host-a",
        ),
        TelemetryRecord(
            datetime(2026, 11, 2, 13, tzinfo=timezone.utc),
            "csv",
            "cpu",
            95.0,
            entity_id="host-b",
        ),
    ]
    persist_telemetry_records(database, records)
    client = TestClient(create_app(database))
    datasets = client.get("/api/v1/datasets").json()
    assert datasets == []
    assert client.get("/api/v1/datasets/stats").json()["total_records"] == 0
    assert len(client.get("/api/v1/telemetry").json()) == 2
    new_dataset = upload(client, filename="new.csv", content=CSV.replace(b"95", b"75"))
    assert new_dataset
    assert client.get("/api/v1/datasets/stats").json()["total_records"] == 4
    assert len(client.get("/api/v1/telemetry").json()) == 6
