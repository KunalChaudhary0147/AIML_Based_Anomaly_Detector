import pandas as pd
import pytest

from anomaly_detector.preprocessing.processor import preprocess_weather_data
from anomaly_detector.utils.exceptions import DataValidationError


def test_processor_normalizes_and_preserves_input(valid_frame):
    source = valid_frame.copy(deep=True)
    processed = preprocess_weather_data(source)
    assert processed is not source
    assert source.equals(valid_frame)
    assert str(processed.timestamp.dt.tz) == "UTC"
    assert processed.is_anomaly.equals(valid_frame.is_anomaly)
    assert processed.station_id.astype(str).equals(valid_frame.station_id.astype(str))


def test_processor_rejects_missing_and_duplicates(valid_frame):
    missing = valid_frame.copy()
    missing.loc[0, "temperature"] = None
    with pytest.raises(DataValidationError):
        preprocess_weather_data(missing)
    duplicate = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        preprocess_weather_data(duplicate)


def test_processor_preserves_extreme_anomaly(valid_frame):
    processed = preprocess_weather_data(valid_frame)
    assert (processed.loc[processed.is_anomaly, "humidity"] > 100).any()
