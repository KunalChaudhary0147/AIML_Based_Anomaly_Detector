import pandas as pd

from anomaly_detector.config import settings
from anomaly_detector.data_ingestion.csv_loader import load_weather_csv
from anomaly_detector.pipeline import run_pipeline


def test_opt_in_cross_feature_pipeline_preserves_phase1_output(tmp_path):
    raw_path = tmp_path / "reference.csv"
    processed_path = tmp_path / "processed.csv"
    prediction_path = tmp_path / "predictions.csv"
    data = pd.DataFrame([
        {
            "timestamp": "2026-09-01T12:00:00Z",
            "station_id": "PHASE22_A",
            "temperature": settings.HEAT_STRESS_TEMPERATURE_MIN + 1,
            "humidity": settings.HEAT_STRESS_HUMIDITY_MIN + 1,
            "wind_speed": 5.0,
            "atmospheric_pressure": 1013.0,
            "is_anomaly": True,
        },
        {
            "timestamp": "2026-09-01T13:00:00Z",
            "station_id": "PHASE22_A",
            "temperature": 20.0,
            "humidity": 50.0,
            "wind_speed": 5.0,
            "atmospheric_pressure": 1013.0,
            "is_anomaly": False,
        },
    ])
    data.to_csv(raw_path, index=False)
    raw_bytes = raw_path.read_bytes()

    summary = run_pipeline(
        raw_path,
        processed_path,
        prediction_path,
        enable_cross_feature_rules=True,
    )
    processed = load_weather_csv(processed_path)
    predictions = pd.read_csv(prediction_path)
    assert summary.evaluation is not None
    assert predictions["predicted_is_anomaly"].tolist() == [True, False]
    assert predictions["anomaly_reasons"].iloc[0] == "heat_stress_condition"
    assert list(processed.columns) == list(data.columns)
    assert raw_path.read_bytes() == raw_bytes
