import pytest

from anomaly_detector.data_ingestion.csv_loader import load_weather_csv
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError


def test_load_valid_fixture_preserves_leading_zero_station(fixture_dir):
    frame = load_weather_csv(fixture_dir / "valid_weather.csv")
    assert frame.station_id.iloc[0] == "001"
    assert frame.is_anomaly.tolist() == [False, True]
    assert str(frame.timestamp.dt.tz) == "UTC"


@pytest.mark.parametrize("name", ["missing_columns.csv", "malformed_weather.csv", "invalid_types.csv", "duplicate_records.csv"])
def test_invalid_fixtures_fail(fixture_dir, name):
    with pytest.raises(DataValidationError):
        load_weather_csv(fixture_dir / name)


def test_missing_file_fails(tmp_path):
    with pytest.raises(DataIngestionError):
        load_weather_csv(tmp_path / "missing.csv")


def test_empty_file_fails(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_weather_csv(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("temperature", "not-a-number"),
        ("timestamp", "not-a-time"),
        ("is_anomaly", "maybe"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, column, value):
    path = tmp_path / "invalid.csv"
    row = {
        "timestamp": "2025-01-01T00:00:00Z",
        "station_id": "001",
        "temperature": "10.0",
        "humidity": "50.0",
        "wind_speed": "3.0",
        "atmospheric_pressure": "1012.0",
        "is_anomaly": "false",
    }
    row[column] = value
    path.write_text(",".join(row) + "\n" + ",".join(row.values()) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_weather_csv(path)


def test_missing_value_and_extra_column_are_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        "timestamp,station_id,temperature,humidity,wind_speed,atmospheric_pressure,is_anomaly,extra\n"
        "2025-01-01T00:00:00Z,001,,50.0,3.0,1012.0,false,ignored\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError):
        load_weather_csv(path)


def test_non_finite_numeric_value_is_rejected(tmp_path):
    path = tmp_path / "nonfinite.csv"
    path.write_text(
        "timestamp,station_id,temperature,humidity,wind_speed,atmospheric_pressure,is_anomaly\n"
        "2025-01-01T00:00:00Z,001,inf,50.0,3.0,1012.0,false\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError):
        load_weather_csv(path)
