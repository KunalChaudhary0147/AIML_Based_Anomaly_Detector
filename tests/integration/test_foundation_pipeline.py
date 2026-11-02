from anomaly_detector.data_ingestion.csv_loader import load_weather_csv
from anomaly_detector.data_ingestion.weather_generator import generate_weather_data
from anomaly_detector.pipeline import run_pipeline


def test_complete_pipeline_preserves_data(tmp_path):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    generated = generate_weather_data(3, 5, 42, 0.2)
    serializable = generated.copy()
    serializable["timestamp"] = serializable["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serializable.to_csv(raw_path, index=False)
    original_raw = raw_path.read_bytes()

    summary = run_pipeline(raw_path, processed_path)
    loaded = load_weather_csv(processed_path)
    assert summary.input_rows == summary.output_rows == 15
    assert summary.station_count == 3
    assert summary.anomaly_count == 3
    assert raw_path.read_bytes() == original_raw
    assert len(loaded) == len(generated)
    assert loaded.is_anomaly.sum() == generated.is_anomaly.sum()
    assert not list(tmp_path.glob(".*processed.csv.*"))


def test_failed_processing_does_not_replace_existing_output(tmp_path):
    raw_path = tmp_path / "bad.csv"
    output_path = tmp_path / "processed.csv"
    output_path.write_text("sentinel", encoding="utf-8")
    raw_path.write_text("timestamp,station_id\nnot-a-time,STATION_001\n", encoding="utf-8")
    try:
        run_pipeline(raw_path, output_path)
    except Exception:
        pass
    assert output_path.read_text(encoding="utf-8") == "sentinel"
