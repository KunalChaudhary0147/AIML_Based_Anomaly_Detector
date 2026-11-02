from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.data_ingestion.telemetry_persistence import (
    DetectionPersistenceResult,
    persist_advanced_detection_results,
    retrieve_advanced_detection_results,
)
from anomaly_detector.detection.advanced_detector import detect_z_scores


def test_csv_to_advanced_detection_to_sqlite(tmp_path):
    csv_path = tmp_path / "infra.csv"
    csv_path.write_text(
        "timestamp,host,cpu\n"
        "2026-11-02T12:00:00Z,host-a,10\n"
        "2026-11-02T12:01:00Z,host-a,10\n"
        "2026-11-02T12:02:00Z,host-a,10\n"
        "2026-11-02T12:03:00Z,host-a,100\n",
        encoding="utf-8",
    )
    records = load_infrastructure_csv(csv_path, metric_columns=["cpu"], entity_column="host")
    results = detect_z_scores(records, z_threshold=1.5, min_samples=3)
    database_path = tmp_path / "advanced.sqlite"
    saved = persist_advanced_detection_results(database_path, results)
    restored = retrieve_advanced_detection_results(database_path, detection_method="z_score")
    assert saved == DetectionPersistenceResult(4, 0)
    assert len(restored) == 4
    assert restored[-1].predicted_is_anomaly is True
    assert restored[-1].record.entity_id == "host-a"
    assert persist_advanced_detection_results(database_path, results) == DetectionPersistenceResult(0, 4)
