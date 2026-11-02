from anomaly_detector.alert_management import generate_alerts, persist_alerts, retrieve_alerts
from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.data_ingestion.telemetry_persistence import persist_generic_detection_results
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry


def test_csv_detection_persistence_alert_flow(tmp_path):
    csv_path = tmp_path / "infra.csv"
    csv_path.write_text("timestamp,host,cpu\n2026-11-02T12:00:00Z,a,95\n", encoding="utf-8")
    database = tmp_path / "alerts.sqlite"
    records = load_infrastructure_csv(csv_path, metric_columns=["cpu"], entity_column="host")
    results = detect_numeric_telemetry(records, {"cpu": NumericThreshold(0, 80)})
    persist_generic_detection_results(database, results)
    alerts = generate_alerts(results)
    assert persist_alerts(database, alerts) == 1
    stored = retrieve_alerts(database, status="NEW", metric_name="cpu", entity_id="a")
    assert len(stored) == 1
    assert stored[0].severity == "LOW"
