"""Validated CSV ingestion for generic infrastructure telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError


def _parse_optional_labels(series: pd.Series, column: str) -> list[bool | None]:
    if series.isna().any():
        raise DataValidationError(f"Missing ground-truth labels in column {column!r}.")
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise DataValidationError(f"Invalid Boolean labels in column {column!r}.")
    return [mapping[value] for value in normalized]


def load_infrastructure_csv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    metric_columns: Sequence[str] | None = None,
    entity_column: str | None = None,
    label_column: str | None = None,
) -> list[TelemetryRecord]:
    """Load infrastructure metrics from CSV as validated telemetry records.

    Naive timestamps are interpreted as UTC by pandas; all timestamps are
    stored as timezone-aware UTC values. If ``metric_columns`` is omitted, all
    columns except timestamp, entity, and label columns are treated as metrics.
    Missing metric values are rejected. Duplicate timestamp/entity/metric
    observations are rejected rather than silently discarded.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataIngestionError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise DataIngestionError(f"CSV path is not a file: {csv_path}")
    try:
        if csv_path.stat().st_size == 0:
            raise DataValidationError(f"CSV file is empty: {csv_path}")
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"CSV file has no header: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Malformed CSV file: {csv_path}") from exc
    except OSError as exc:
        raise DataIngestionError(f"Unable to read CSV file: {csv_path}") from exc

    if frame.empty:
        raise DataValidationError(f"CSV file contains no observations: {csv_path}")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip() or timestamp_column not in frame.columns:
        raise DataValidationError(f"Missing timestamp column {timestamp_column!r}.")
    for optional_column in (entity_column, label_column):
        if optional_column is not None and optional_column not in frame.columns:
            raise DataValidationError(f"Configured column {optional_column!r} is missing.")

    excluded = {timestamp_column}
    if entity_column is not None:
        excluded.add(entity_column)
    if label_column is not None:
        excluded.add(label_column)
    selected_metrics = list(metric_columns) if metric_columns is not None else [column for column in frame.columns if column not in excluded]
    if not selected_metrics:
        raise DataValidationError("At least one metric column is required.")
    if len(set(selected_metrics)) != len(selected_metrics):
        raise DataValidationError("Metric columns must be unique.")
    missing_metrics = [column for column in selected_metrics if column not in frame.columns]
    if missing_metrics:
        raise DataValidationError(f"Missing metric columns: {missing_metrics}")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True, format="mixed")
    if timestamps.isna().any():
        raise DataValidationError(f"Invalid or missing timestamps in column {timestamp_column!r}.")

    entities: list[str | None]
    if entity_column is None:
        entities = [None] * len(frame)
    else:
        entity_series = frame[entity_column].astype("string")
        if entity_series.isna().any() or entity_series.str.strip().eq("").any():
            raise DataValidationError(f"Entity column {entity_column!r} contains missing or empty values.")
        entities = [str(value) for value in entity_series]

    labels = [None] * len(frame) if label_column is None else _parse_optional_labels(frame[label_column], label_column)
    records: list[TelemetryRecord] = []
    seen: set[tuple[object, ...]] = set()
    for row_index, timestamp in enumerate(timestamps):
        for metric_name in selected_metrics:
            value = pd.to_numeric(frame.iloc[row_index][metric_name], errors="coerce")
            if pd.isna(value) or not np.isfinite(float(value)):
                raise DataValidationError(f"Invalid or missing numeric value in {metric_name!r} at row {row_index}.")
            record = TelemetryRecord(
                timestamp=timestamp.to_pydatetime(),
                source_type="csv",
                metric_name=metric_name,
                metric_value=float(value),
                entity_id=entities[row_index],
                is_anomaly=labels[row_index],
            )
            if record.datapoint_identity in seen:
                raise DataValidationError(f"Duplicate observation at row {row_index} for metric {metric_name!r}.")
            seen.add(record.datapoint_identity)
            records.append(record)
    return records
