from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.data_ingestion.telemetry_persistence import (
    persist_generic_detection_results,
    persist_telemetry_records,
)
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry
from anomaly_detector.historical import (
    export_generic_detection_csv,
    retrieve_generic_detection_history,
    retrieve_telemetry_history,
)


def test_csv_ingestion_detection_persistence_filter_and_export(tmp_path):
    csv_path = tmp_path / "infra.csv"
    csv_path.write_text(
        "timestamp,host,cpu\n"
        "2026-11-02T12:00:00Z,host-a,10\n"
        "2026-11-02T12:01:00Z,host-a,90\n",
        encoding="utf-8",
    )
    database_path = tmp_path / "history.sqlite"
    output_path = tmp_path / "filtered_results.csv"
    records = load_infrastructure_csv(csv_path, metric_columns=["cpu"], entity_column="host")
    persist_telemetry_records(database_path, records)
    results = detect_numeric_telemetry(records, {"cpu": NumericThreshold(0, 80)})
    persist_generic_detection_results(database_path, results)
    filtered = retrieve_generic_detection_history(database_path, anomaly=True, entity_id="host-a")
    telemetry = retrieve_telemetry_history(database_path, metric_name="cpu", entity_id="host-a")
    export_generic_detection_csv(filtered, output_path)
    assert len(telemetry) == 2
    assert len(filtered) == 1
    assert filtered[0].record.metric_value == 90.0
    assert "predicted_is_anomaly" in output_path.read_text(encoding="utf-8").splitlines()[0]
