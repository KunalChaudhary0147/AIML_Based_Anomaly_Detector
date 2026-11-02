"""Configurable rule-based weather anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from anomaly_detector.config import settings
from anomaly_detector.utils.exceptions import DataValidationError

PREDICTION_COLUMN: Final[str] = "predicted_is_anomaly"
REASON_COLUMN: Final[str] = "anomaly_reasons"


@dataclass(frozen=True)
class Rule:
    """Inclusive lower and upper bounds for one numeric weather field."""

    column: str
    lower: float
    upper: float
    identifier: str

    def validate(self) -> None:
        """Reject contradictory or non-finite rule values."""
        if not np.isfinite([self.lower, self.upper]).all():
            raise DataValidationError(f"Rule {self.identifier} must use finite thresholds.")
        if self.lower > self.upper:
            raise DataValidationError(f"Rule {self.identifier} has lower bound above upper bound.")


@dataclass(frozen=True)
class CrossFeatureRule:
    """A deterministic conjunction of two defensible minimum thresholds."""

    first_column: str
    first_minimum: float
    second_column: str
    second_minimum: float
    identifier: str

    def validate(self) -> None:
        """Validate columns and finite thresholds."""
        if self.first_column not in settings.NUMERIC_COLUMNS or self.second_column not in settings.NUMERIC_COLUMNS:
            raise DataValidationError(f"Unsupported cross-feature columns for rule {self.identifier}.")
        if not np.isfinite([self.first_minimum, self.second_minimum]).all():
            raise DataValidationError(f"Cross-feature rule {self.identifier} must use finite thresholds.")


DEFAULT_CROSS_FEATURE_RULES: Final[tuple[CrossFeatureRule, ...]] = ()
HEAT_STRESS_RULE: Final[CrossFeatureRule] = CrossFeatureRule(
    "temperature", settings.HEAT_STRESS_TEMPERATURE_MIN,
    "humidity", settings.HEAT_STRESS_HUMIDITY_MIN,
    "heat_stress_condition",
)


DEFAULT_RULES: Final[tuple[Rule, ...]] = (
    Rule("temperature", settings.TEMPERATURE_MIN, settings.TEMPERATURE_MAX, "temperature_range"),
    Rule("humidity", settings.HUMIDITY_MIN, settings.HUMIDITY_MAX, "humidity_range"),
    Rule("wind_speed", settings.WIND_SPEED_MIN, settings.WIND_SPEED_MAX, "wind_speed_range"),
    Rule("atmospheric_pressure", settings.PRESSURE_MIN, settings.PRESSURE_MAX, "pressure_range"),
)


def detect_anomalies(
    data: pd.DataFrame,
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    cross_feature_rules: tuple[CrossFeatureRule, ...] = DEFAULT_CROSS_FEATURE_RULES,
) -> pd.DataFrame:
    """Return a copy with rule predictions and semicolon-separated reasons.

    Bounds are inclusive: values exactly on either threshold are normal. The
    detector only reads configured numeric fields and never reads ``is_anomaly``.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Detector input must be a pandas DataFrame.")
    if not isinstance(rules, tuple) or not rules:
        raise DataValidationError("At least one detection rule is required.")
    for rule in rules:
        if not isinstance(rule, Rule):
            raise DataValidationError("Detection rules must be Rule instances.")
        rule.validate()
        if rule.column not in settings.NUMERIC_COLUMNS:
            raise DataValidationError(f"Unsupported detection column: {rule.column}")
        if rule.column not in data.columns:
            raise DataValidationError(f"Missing required detection column: {rule.column}")
    if not isinstance(cross_feature_rules, tuple):
        raise DataValidationError("cross_feature_rules must be a tuple.")
    for rule in cross_feature_rules:
        if not isinstance(rule, CrossFeatureRule):
            raise DataValidationError("Cross-feature rules must be CrossFeatureRule instances.")
        rule.validate()
        for column in (rule.first_column, rule.second_column):
            if column not in data.columns:
                raise DataValidationError(f"Missing required detection column: {column}")

    result = data.copy(deep=True)
    if result.empty:
        result[PREDICTION_COLUMN] = pd.Series(dtype=bool, index=result.index)
        result[REASON_COLUMN] = pd.Series(dtype="string", index=result.index)
        return result

    violations: dict[str, pd.Series] = {}
    for rule in rules:
        numeric_values = pd.to_numeric(result[rule.column], errors="coerce")
        if numeric_values.isna().any() or not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
            raise DataValidationError(f"Missing or non-finite numeric values in detection column: {rule.column}")
        violations[rule.identifier] = (numeric_values < rule.lower) | (numeric_values > rule.upper)
    for rule in cross_feature_rules:
        first_values = pd.to_numeric(result[rule.first_column], errors="coerce")
        second_values = pd.to_numeric(result[rule.second_column], errors="coerce")
        if first_values.isna().any() or second_values.isna().any():
            raise DataValidationError(f"Missing or invalid numeric values for {rule.identifier}")
        if not np.isfinite(first_values.to_numpy(dtype=float)).all() or not np.isfinite(second_values.to_numpy(dtype=float)).all():
            raise DataValidationError(f"Non-finite numeric values for {rule.identifier}")
        violations[rule.identifier] = (first_values >= rule.first_minimum) & (second_values >= rule.second_minimum)

    reason_frame = pd.DataFrame(violations, index=result.index)
    result[PREDICTION_COLUMN] = reason_frame.any(axis=1).astype(bool)
    result[REASON_COLUMN] = reason_frame.apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1,
    ).astype("string")
    return result
