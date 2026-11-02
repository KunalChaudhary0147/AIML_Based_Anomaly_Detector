"""Strict CSV loading and structural validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detector.config.settings import EXPECTED_COLUMNS, NUMERIC_COLUMNS
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError


def _parse_bool_series(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise DataValidationError(f"Invalid Boolean values in {column}: {series[invalid].tolist()}.")
    return normalized.map(mapping).astype(bool)


def load_weather_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a weather CSV, preserving all valid observations."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataIngestionError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise DataIngestionError(f"CSV path is not a file: {csv_path}")
    try:
        if csv_path.stat().st_size == 0:
            raise DataValidationError(f"CSV file is empty: {csv_path}")
        frame = pd.read_csv(csv_path, dtype={"station_id": "string", "is_anomaly": "string"})
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"CSV file contains no columns: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Malformed CSV structure in {csv_path}: {exc}") from exc
    except OSError as exc:
        raise DataIngestionError(f"Unable to read CSV file {csv_path}: {exc}") from exc

    if frame.empty:
        raise DataValidationError(f"CSV dataset contains no rows: {csv_path}")
    columns = list(frame.columns)
    missing = [column for column in EXPECTED_COLUMNS if column not in columns]
    extra = [column for column in columns if column not in EXPECTED_COLUMNS]
    if missing:
        raise DataValidationError(f"Missing required columns in {csv_path}: {missing}")
    if extra:
        raise DataValidationError(f"Unexpected columns in {csv_path}: {extra}")
    frame = frame.loc[:, EXPECTED_COLUMNS].copy()

    try:
        parsed_timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Invalid timestamps in {csv_path}.") from exc
    if parsed_timestamps.isna().any():
        raise DataValidationError(f"Invalid or missing timestamps in {csv_path}.")
    frame["timestamp"] = parsed_timestamps

    station_ids = frame["station_id"].astype("string")
    if station_ids.isna().any() or station_ids.str.strip().eq("").any():
        raise DataValidationError(f"Empty station_id values in {csv_path}.")
    frame["station_id"] = station_ids

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise DataValidationError(f"Invalid or non-finite numeric values in {column} ({csv_path}).")
        frame[column] = converted.astype(float)
    frame["is_anomaly"] = _parse_bool_series(frame["is_anomaly"], "is_anomaly")

    duplicates = frame.duplicated(subset=["timestamp", "station_id"], keep=False)
    if duplicates.any():
        raise DataValidationError(f"Duplicate timestamp/station_id observations in {csv_path}.")
    return frame
