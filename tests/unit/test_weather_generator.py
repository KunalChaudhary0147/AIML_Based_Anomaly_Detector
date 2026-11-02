import numpy as np
import pandas as pd
import pytest

from anomaly_detector.config.settings import EXPECTED_COLUMNS
from anomaly_detector.data_ingestion.weather_generator import generate_weather_data
from anomaly_detector.utils.exceptions import DataValidationError


def test_generator_schema_and_reproducibility():
    first = generate_weather_data(2, 4, 7, 0.5)
    second = generate_weather_data(2, 4, 7, 0.5)
    assert list(first.columns) == list(EXPECTED_COLUMNS)
    assert first.equals(second)
    assert pd.api.types.is_string_dtype(first.station_id)
    assert pd.api.types.is_bool_dtype(first.is_anomaly)
    assert first.is_anomaly.sum() == 4


def test_generator_does_not_change_global_numpy_state():
    np.random.seed(123)
    before = np.random.get_state()
    generate_weather_data(2, 3, 99, 0.2)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])


def test_generator_boundaries_and_validation():
    assert not generate_weather_data(1, 4, 1, 0).is_anomaly.any()
    assert generate_weather_data(1, 4, 1, 1).is_anomaly.all()
    with pytest.raises(DataValidationError):
        generate_weather_data(1, 4, 1, 1.1)
