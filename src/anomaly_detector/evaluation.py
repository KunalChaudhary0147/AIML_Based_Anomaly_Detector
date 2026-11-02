"""Evaluation metrics for comparing predictions with ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from anomaly_detector.detection.detector import PREDICTION_COLUMN
from anomaly_detector.utils.exceptions import DataValidationError


@dataclass(frozen=True)
class EvaluationResult:
    """Binary classification counts and metrics."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1_score: float
    ground_truth_available: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return metrics as a plain dictionary."""
        return asdict(self)


def _validate_boolean_series(series: pd.Series, column: str) -> pd.Series:
    """Accept actual Boolean values and reject coercible non-Booleans."""
    if series.isna().any():
        raise DataValidationError(f"{column} must contain no missing values.")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataValidationError(f"{column} must contain Boolean values, not strings or integers.")
    return series.astype(bool)


def evaluate_predictions(data: pd.DataFrame) -> EvaluationResult | None:
    """Evaluate predictions when a valid ``is_anomaly`` label is available."""
    if PREDICTION_COLUMN not in data.columns:
        raise DataValidationError(f"Missing prediction column: {PREDICTION_COLUMN}")
    predicted = _validate_boolean_series(data[PREDICTION_COLUMN], PREDICTION_COLUMN)
    if "is_anomaly" not in data.columns:
        return None
    actual = _validate_boolean_series(data["is_anomaly"], "is_anomaly")
    true_positive = int((predicted & actual).sum())
    false_positive = int((predicted & ~actual).sum())
    true_negative = int((~predicted & ~actual).sum())
    false_negative = int((~predicted & actual).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
    return EvaluationResult(
        true_positive, false_positive, true_negative, false_negative,
        precision, recall, f1_score,
    )
