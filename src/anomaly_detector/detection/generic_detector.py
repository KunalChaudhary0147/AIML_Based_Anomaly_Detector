"""Minimal configurable rule detector for generic numeric telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import DataValidationError


@dataclass(frozen=True)
class NumericThreshold:
    """Inclusive lower and upper bounds for one metric name."""

    lower: float
    upper: float

    def validate(self, metric_name: str) -> None:
        """Reject non-finite or contradictory thresholds."""
        if not isfinite(float(self.lower)) or not isfinite(float(self.upper)):
            raise DataValidationError(f"Thresholds for {metric_name!r} must be finite.")
        if self.lower > self.upper:
            raise DataValidationError(f"Lower threshold exceeds upper threshold for {metric_name!r}.")


@dataclass(frozen=True)
class GenericDetectionResult:
    """Prediction associated with the original telemetry record."""

    record: TelemetryRecord
    predicted_is_anomaly: bool
    explanation: str


def detect_numeric_telemetry(
    records: list[TelemetryRecord] | tuple[TelemetryRecord, ...],
    thresholds: Mapping[str, NumericThreshold],
) -> list[GenericDetectionResult]:
    """Apply configured inclusive thresholds to generic telemetry records.

    Values below the lower bound or above the upper bound are anomalous.
    Metrics without a configured threshold are returned as unconfigured, not
    normal. The ground-truth ``is_anomaly`` field is never inspected.
    """
    if not isinstance(thresholds, Mapping):
        raise DataValidationError("thresholds must be a mapping by metric name.")
    for metric_name, threshold in thresholds.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise DataValidationError("Threshold metric names must be non-empty strings.")
        if not isinstance(threshold, NumericThreshold):
            raise DataValidationError(f"Threshold for {metric_name!r} must be NumericThreshold.")
        threshold.validate(metric_name)

    results: list[GenericDetectionResult] = []
    for record in records:
        if not isinstance(record, TelemetryRecord):
            raise DataValidationError("records must contain only TelemetryRecord objects.")
        threshold = thresholds.get(record.metric_name)
        if threshold is None:
            results.append(GenericDetectionResult(record, False, "unconfigured_metric"))
        elif record.metric_value < threshold.lower:
            results.append(GenericDetectionResult(record, True, f"below_lower_threshold:{threshold.lower}"))
        elif record.metric_value > threshold.upper:
            results.append(GenericDetectionResult(record, True, f"above_upper_threshold:{threshold.upper}"))
        else:
            results.append(GenericDetectionResult(record, False, "within_configured_thresholds"))
    return results
