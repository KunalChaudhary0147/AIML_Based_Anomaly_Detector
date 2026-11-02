"""Deterministic synthetic weather observation generation."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from anomaly_detector.config.settings import EXPECTED_COLUMNS
from anomaly_detector.utils.exceptions import DataValidationError


def generate_weather_data(
    station_count: int = 3,
    observations_per_station: int = 24,
    seed: int = 42,
    anomaly_rate: float = 0.05,
) -> pd.DataFrame:
    """Generate a UTC-indexed synthetic dataset without writing to disk.

    The anomaly rate applies to the complete dataset. The target count is
    ``floor(rate * rows + 0.5)`` (nearest integer, with halves rounded up).
    """
    if not isinstance(station_count, Integral) or isinstance(station_count, bool) or station_count < 1:
        raise DataValidationError("station_count must be a positive integer.")
    if not isinstance(observations_per_station, Integral) or isinstance(observations_per_station, bool) or observations_per_station < 1:
        raise DataValidationError("observations_per_station must be a positive integer.")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise DataValidationError("seed must be an integer.")
    if not isinstance(anomaly_rate, Real) or isinstance(anomaly_rate, bool) or not 0 <= anomaly_rate <= 1:
        raise DataValidationError("anomaly_rate must be between 0 and 1 inclusive.")

    rng = np.random.default_rng(int(seed))
    total_rows = int(station_count) * int(observations_per_station)
    timestamps = pd.date_range("2025-01-01T00:00:00Z", periods=observations_per_station, freq="h")
    records: list[dict[str, object]] = []
    for station_number in range(1, int(station_count) + 1):
        station_id = f"STATION_{station_number:03d}"
        for observation_number, timestamp in enumerate(timestamps):
            hour = observation_number % 24
            temperature = 15 + 8 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 1.5)
            humidity = np.clip(65 - 15 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 4), 15, 95)
            wind_speed = np.clip(rng.normal(5, 1.5), 0, 15)
            pressure = rng.normal(1013, 7)
            records.append({
                "timestamp": timestamp,
                "station_id": station_id,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "atmospheric_pressure": pressure,
                "is_anomaly": False,
            })

    anomaly_count = int(np.floor(float(anomaly_rate) * total_rows + 0.5))
    if anomaly_count:
        anomaly_indices = rng.choice(total_rows, size=anomaly_count, replace=False)
        for index in np.asarray(anomaly_indices):
            record = records[int(index)]
            anomaly_kind = int(rng.integers(0, 4))
            if anomaly_kind == 0:
                record["temperature"] = float(rng.choice([-45.0, 58.0]))
            elif anomaly_kind == 1:
                record["humidity"] = float(rng.choice([-20.0, 135.0]))
            elif anomaly_kind == 2:
                record["wind_speed"] = float(rng.uniform(45.0, 90.0))
            else:
                record["atmospheric_pressure"] = float(rng.choice([850.0, 1150.0]))
            record["is_anomaly"] = True

    return pd.DataFrame.from_records(records, columns=EXPECTED_COLUMNS)
