from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.data_ingestion.telemetry_persistence import (
    persist_generic_detection_results,
    retrieve_generic_detection_results,
)
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry


def test_csv_telemetry_flows_through_generic_detector_and_sqlite(tmp_path):
    csv_path = tmp_path / "infra.csv"
    csv_path.write_text(
        "timestamp,host,cpu,latency\n"
        "2026-11-02T12:00:00Z,host-a,25,100\n"
        "2026-11-02T12:01:00Z,host-a,95,250\n",
        encoding="utf-8",
    )
    records = load_infrastructure_csv(
        csv_path,
        metric_columns=["cpu", "latency"],
        entity_column="host",
    )
    results = detect_numeric_telemetry(
        records,
        {"cpu": NumericThreshold(0, 80), "latency": NumericThreshold(0, 200)},
    )
    database_path = tmp_path / "results.sqlite"
    persistence = persist_generic_detection_results(database_path, results)
    restored = retrieve_generic_detection_results(database_path, anomaly=True)
    assert persistence.inserted == 4
    assert len(restored) == 2
    assert {item.record.metric_name for item in restored} == {"cpu", "latency"}
    assert all(item.record.source_type == "csv" for item in restored)
