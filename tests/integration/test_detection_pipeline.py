from pathlib import Path

import pandas as pd
import pytest

from anomaly_detector.data_ingestion.csv_loader import load_weather_csv
import anomaly_detector.pipeline as pipeline_module
from anomaly_detector.data_ingestion.weather_generator import generate_weather_data
from anomaly_detector.pipeline import run_pipeline
from anomaly_detector.utils.exceptions import DataIngestionError


def test_detection_pipeline_writes_separate_predictions(tmp_path):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    prediction_path = tmp_path / "predictions.csv"
    generated = generate_weather_data(2, 6, 19, 0.5)
    serializable = generated.copy()
    serializable["timestamp"] = serializable["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serializable.to_csv(raw_path, index=False)
    raw_bytes = raw_path.read_bytes()

    summary = run_pipeline(raw_path, processed_path, prediction_path)
    processed = load_weather_csv(processed_path)
    predictions = pd.read_csv(prediction_path)

    assert summary.prediction_path == Path(prediction_path)
    assert summary.evaluation is not None
    assert list(processed.columns) == list(generated.columns)
    assert "predicted_is_anomaly" in predictions.columns
    assert "is_anomaly" in predictions.columns
    assert len(predictions) == len(generated)
    assert predictions["predicted_is_anomaly"].dtype == bool
    assert raw_path.read_bytes() == raw_bytes
    assert predictions["is_anomaly"].tolist() == generated["is_anomaly"].tolist()
    assert predictions["predicted_is_anomaly"].any()
    assert not list(tmp_path.glob(".*predictions.csv.*"))


def test_prediction_preparation_failure_preserves_both_existing_outputs(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    prediction_path = tmp_path / "predictions.csv"
    generated = generate_weather_data(1, 2, 19, 0)
    generated.to_csv(raw_path, index=False)
    processed_path.write_bytes(b"old processed")
    prediction_path.write_bytes(b"old predictions")
    original_prepare = pipeline_module._prepare_output

    def fail_prediction(data, output_path):
        if output_path == prediction_path:
            raise DataIngestionError("simulated prediction preparation failure")
        return original_prepare(data, output_path)

    monkeypatch.setattr(pipeline_module, "_prepare_output", fail_prediction)
    with pytest.raises(DataIngestionError):
        run_pipeline(raw_path, processed_path, prediction_path)
    assert processed_path.read_bytes() == b"old processed"
    assert prediction_path.read_bytes() == b"old predictions"
    assert not list(tmp_path.glob(".*"))


def test_replacement_failure_rolls_back_both_outputs(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    prediction_path = tmp_path / "predictions.csv"
    generated = generate_weather_data(1, 2, 19, 0)
    generated.to_csv(raw_path, index=False)
    processed_path.write_bytes(b"old processed")
    prediction_path.write_bytes(b"old predictions")
    real_replace = pipeline_module.os.replace
    replace_calls = 0

    def fail_prediction_publish(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_prediction_publish)
    with pytest.raises(DataIngestionError):
        run_pipeline(raw_path, processed_path, prediction_path)
    assert processed_path.read_bytes() == b"old processed"
    assert prediction_path.read_bytes() == b"old predictions"
    assert not list(tmp_path.glob(".*"))
