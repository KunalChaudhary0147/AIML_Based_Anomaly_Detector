from pathlib import Path

import pytest

from anomaly_detector.data_ingestion.weather_generator import generate_weather_data


@pytest.fixture
def valid_frame():
    return generate_weather_data(2, 3, 11, 1 / 6)


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"
