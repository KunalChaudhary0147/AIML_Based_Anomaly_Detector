"""Safe Phase 1 preprocessing for weather observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_detector.config.settings import EXPECTED_COLUMNS, NUMERIC_COLUMNS
from anomaly_detector.utils.exceptions import DataValidationError, PreprocessingError


def preprocess_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a copy of weather data without filtering anomalies."""
    if not isinstance(data, pd.DataFrame):
        raise PreprocessingError("data must be a pandas DataFrame.")
    if data.empty:
        raise DataValidationError("Weather dataset contains no rows.")
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    extra_columns = [column for column in data.columns if column not in EXPECTED_COLUMNS]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if extra_columns:
        raise DataValidationError(f"Unexpected columns: {extra_columns}")

    processed = data.loc[:, EXPECTED_COLUMNS].copy()
    missing = processed.isna().any()
    if missing.any():
        raise DataValidationError(f"Missing values in required columns: {list(missing[missing].index)}")

    try:
        processed["timestamp"] = pd.to_datetime(processed["timestamp"], errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Timestamps could not be normalized to UTC.") from exc
    if processed["timestamp"].isna().any():
        raise DataValidationError("Invalid timestamps detected.")

    processed["station_id"] = processed["station_id"].astype("string")
    if processed["station_id"].str.strip().eq("").any():
        raise DataValidationError("station_id must not be empty.")
    for column in NUMERIC_COLUMNS:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        if processed[column].isna().any() or not np.isfinite(processed[column].to_numpy(dtype=float)).all():
            raise DataValidationError(f"Invalid or non-finite values in {column}.")
        processed[column] = processed[column].astype(float)

    if processed["is_anomaly"].dtype != bool:
        normalized = processed["is_anomaly"].astype("string").str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False}
        if not normalized.isin(mapping).all():
            raise DataValidationError("is_anomaly must contain Boolean values.")
        processed["is_anomaly"] = normalized.map(mapping).astype(bool)
    else:
        processed["is_anomaly"] = processed["is_anomaly"].astype(bool)

    if processed.duplicated(subset=["timestamp", "station_id"], keep=False).any():
        raise DataValidationError("Duplicate timestamp/station_id observations detected.")
    return processed
