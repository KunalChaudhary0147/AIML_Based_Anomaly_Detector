import pandas as pd
import pytest

from anomaly_detector.detection.detector import Rule, detect_anomalies
from anomaly_detector.utils.exceptions import DataValidationError


def weather_frame(**overrides):
    values = {
        "timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
        "station_id": "STATION_001",
        "temperature": 20.0,
        "humidity": 50.0,
        "wind_speed": 5.0,
        "atmospheric_pressure": 1013.0,
        "is_anomaly": False,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_normal_observation_and_threshold_edges_are_not_anomalies():
    data = pd.concat(
        [
            weather_frame(),
            weather_frame(temperature=-40.0, station_id="STATION_002"),
            weather_frame(humidity=100.0, station_id="STATION_003"),
        ],
        ignore_index=True,
    )
    result = detect_anomalies(data)
    assert result.predicted_is_anomaly.tolist() == [False, False, False]
    assert result.anomaly_reasons.tolist() == ["", "", ""]


def test_upper_lower_and_multiple_rule_violations_are_reported():
    data = pd.concat(
        [
            weather_frame(temperature=51.0, station_id="STATION_002"),
            weather_frame(temperature=-41.0, humidity=101.0, wind_speed=36.0, station_id="STATION_003"),
        ],
        ignore_index=True,
    )
    result = detect_anomalies(data)
    assert result.predicted_is_anomaly.tolist() == [True, True]
    assert result.anomaly_reasons.iloc[0] == "temperature_range"
    assert result.anomaly_reasons.iloc[1] == "temperature_range;humidity_range;wind_speed_range"


def test_detector_preserves_input_and_does_not_use_ground_truth():
    normal = weather_frame(is_anomaly=False)
    labeled_anomaly = weather_frame(is_anomaly=True)
    original = normal.copy(deep=True)
    first = detect_anomalies(normal)
    second = detect_anomalies(labeled_anomaly)
    without_label = detect_anomalies(normal.drop(columns=["is_anomaly"]))
    assert normal.equals(original)
    assert first.predicted_is_anomaly.tolist() == second.predicted_is_anomaly.tolist()
    assert without_label.predicted_is_anomaly.tolist() == first.predicted_is_anomaly.tolist()
    assert "is_anomaly" in first.columns


def test_detector_rejects_missing_nonfinite_and_invalid_rules():
    with pytest.raises(DataValidationError):
        detect_anomalies(weather_frame().drop(columns=["humidity"]))
    with pytest.raises(DataValidationError):
        detect_anomalies(weather_frame(temperature=float("nan")))
    with pytest.raises(DataValidationError):
        detect_anomalies(weather_frame(), (Rule("temperature", 5.0, 1.0, "bad"),))
    with pytest.raises(DataValidationError):
        detect_anomalies(weather_frame(), (Rule("not_weather", 0.0, 1.0, "bad"),))


def test_detector_handles_empty_input():
    empty = weather_frame().iloc[0:0].copy()
    result = detect_anomalies(empty)
    assert result.empty
    assert result.predicted_is_anomaly.dtype == bool
    assert result.anomaly_reasons.dtype.name == "string"
