"""Centralized configuration for the anomaly detector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from anomaly_detector.utils.exceptions import ConfigurationError

load_dotenv()

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
RAW_DATA_DIR: Final[Path] = Path(os.getenv("ANOMALY_RAW_DATA_DIR", str(PROJECT_ROOT / "data" / "raw")))
PROCESSED_DATA_DIR: Final[Path] = Path(os.getenv("ANOMALY_PROCESSED_DATA_DIR", str(PROJECT_ROOT / "data" / "processed")))
DEFAULT_RAW_FILENAME: Final[str] = os.getenv("ANOMALY_RAW_FILENAME", "weather_raw.csv")
DEFAULT_PROCESSED_FILENAME: Final[str] = os.getenv("ANOMALY_PROCESSED_FILENAME", "weather_processed.csv")
DEFAULT_PREDICTIONS_FILENAME: Final[str] = os.getenv("ANOMALY_PREDICTIONS_FILENAME", "weather_predictions.csv")
LOG_LEVEL: Final[str] = os.getenv("ANOMALY_LOG_LEVEL", "INFO").upper()

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "station_id",
    "temperature",
    "humidity",
    "wind_speed",
    "atmospheric_pressure",
    "is_anomaly",
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = EXPECTED_COLUMNS
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "temperature",
    "humidity",
    "wind_speed",
    "atmospheric_pressure",
)
PHYSICAL_LIMITS: Final[dict[str, tuple[float, float]]] = {
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 200.0),
}


def _environment_int(name: str, default: str) -> int:
    value = os.getenv(name, default)
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; received {value!r}.") from exc


def _environment_float(name: str, default: str) -> float:
    value = os.getenv(name, default)
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number; received {value!r}.") from exc


def _threshold(name: str, default: str) -> float:
    value = _environment_float(name, default)
    if not value == value or value in {float("inf"), float("-inf")}:
        raise ConfigurationError(f"{name} must be finite; received {value!r}.")
    return value


DEFAULT_RANDOM_SEED: Final[int] = _environment_int("ANOMALY_RANDOM_SEED", "42")
DEFAULT_STATION_COUNT: Final[int] = _environment_int("ANOMALY_STATION_COUNT", "3")
DEFAULT_OBSERVATIONS_PER_STATION: Final[int] = _environment_int("ANOMALY_OBSERVATIONS_PER_STATION", "24")
DEFAULT_ANOMALY_RATE: Final[float] = _environment_float("ANOMALY_RATE", "0.05")

# Explicit operational rules; these are not learned from the data.
TEMPERATURE_MIN: Final[float] = _threshold("ANOMALY_TEMPERATURE_MIN", "-40")
TEMPERATURE_MAX: Final[float] = _threshold("ANOMALY_TEMPERATURE_MAX", "50")
HUMIDITY_MIN: Final[float] = _threshold("ANOMALY_HUMIDITY_MIN", "0")
HUMIDITY_MAX: Final[float] = _threshold("ANOMALY_HUMIDITY_MAX", "100")
WIND_SPEED_MIN: Final[float] = _threshold("ANOMALY_WIND_SPEED_MIN", "0")
WIND_SPEED_MAX: Final[float] = _threshold("ANOMALY_WIND_SPEED_MAX", "35")
PRESSURE_MIN: Final[float] = _threshold("ANOMALY_PRESSURE_MIN", "900")
PRESSURE_MAX: Final[float] = _threshold("ANOMALY_PRESSURE_MAX", "1100")
HEAT_STRESS_TEMPERATURE_MIN: Final[float] = _threshold("ANOMALY_HEAT_STRESS_TEMPERATURE_MIN", "35")
HEAT_STRESS_HUMIDITY_MIN: Final[float] = _threshold("ANOMALY_HEAT_STRESS_HUMIDITY_MIN", "80")


def _validate() -> None:
    """Validate scalar configuration values at import time."""
    if not DEFAULT_RAW_FILENAME.strip() or not DEFAULT_PROCESSED_FILENAME.strip() or not DEFAULT_PREDICTIONS_FILENAME.strip():
        raise ConfigurationError("CSV filenames must not be empty.")
    if DEFAULT_STATION_COUNT < 1:
        raise ConfigurationError("ANOMALY_STATION_COUNT must be at least 1.")
    if DEFAULT_OBSERVATIONS_PER_STATION < 1:
        raise ConfigurationError("ANOMALY_OBSERVATIONS_PER_STATION must be at least 1.")
    if not 0 <= DEFAULT_ANOMALY_RATE <= 1:
        raise ConfigurationError("ANOMALY_RATE must be between 0 and 1 inclusive.")
    if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("ANOMALY_LOG_LEVEL must be a standard logging level.")
    thresholds = {
        "temperature": (TEMPERATURE_MIN, TEMPERATURE_MAX),
        "humidity": (HUMIDITY_MIN, HUMIDITY_MAX),
        "wind_speed": (WIND_SPEED_MIN, WIND_SPEED_MAX),
        "atmospheric_pressure": (PRESSURE_MIN, PRESSURE_MAX),
    }
    for name, (lower, upper) in thresholds.items():
        if lower > upper:
            raise ConfigurationError(f"Invalid {name} thresholds: lower bound exceeds upper bound.")
    if HEAT_STRESS_TEMPERATURE_MIN > TEMPERATURE_MAX:
        raise ConfigurationError("Heat-stress temperature threshold must not exceed temperature maximum.")
    if HEAT_STRESS_HUMIDITY_MIN > HUMIDITY_MAX:
        raise ConfigurationError("Heat-stress humidity threshold must not exceed humidity maximum.")


_validate()
