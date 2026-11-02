"""Statistical and multivariate detectors for generic telemetry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import Iterable, Mapping, Sequence

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import DataValidationError


@dataclass(frozen=True)
class AdvancedDetectionResult:
    """An advanced prediction associated with one original telemetry record."""

    record: TelemetryRecord
    predicted_is_anomaly: bool
    explanation: str
    detection_method: str
    observed_value: float
    reference_value: float | None = None
    statistical_score: float | None = None


def _validate_common(records: Sequence[TelemetryRecord], threshold: float, min_samples: int) -> None:
    if not isfinite(float(threshold)) or threshold <= 0:
        raise DataValidationError("sensitivity threshold must be a finite positive number.")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise DataValidationError("min_samples must be an integer of at least 2.")
    if any(not isinstance(record, TelemetryRecord) for record in records):
        raise DataValidationError("records must contain only TelemetryRecord objects.")


def _group_key(record: TelemetryRecord) -> tuple[str | None, str]:
    return record.entity_id, record.metric_name


def detect_z_scores(
    records: Sequence[TelemetryRecord],
    *,
    z_threshold: float = 3.0,
    min_samples: int = 3,
) -> list[AdvancedDetectionResult]:
    """Detect values outside a per-entity/metric mean and population deviation."""
    _validate_common(records, z_threshold, min_samples)
    groups: dict[tuple[str | None, str], list[TelemetryRecord]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)
    results: list[AdvancedDetectionResult] = []
    for record in records:
        group = groups[_group_key(record)]
        values = [float(item.metric_value) for item in group]
        if len(values) < min_samples:
            results.append(AdvancedDetectionResult(record, False, "insufficient_samples", "z_score", float(record.metric_value)))
            continue
        mean = sum(values) / len(values)
        deviation = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        score = (0.0 if float(record.metric_value) == mean else float("inf")) if deviation == 0 else abs((float(record.metric_value) - mean) / deviation)
        explanation = "z_score_exceeds_threshold" if score > z_threshold else "z_score_within_threshold"
        results.append(AdvancedDetectionResult(record, score > z_threshold, explanation, "z_score", float(record.metric_value), mean, score))
    return results


def detect_rolling_z_scores(
    records: Sequence[TelemetryRecord],
    *,
    window_size: int = 10,
    z_threshold: float = 3.0,
    min_samples: int = 3,
) -> list[AdvancedDetectionResult]:
    """Detect each value against preceding observations for its entity/metric."""
    _validate_common(records, z_threshold, min_samples)
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise DataValidationError("window_size must be a positive integer.")
    groups: dict[tuple[str | None, str], list[TelemetryRecord]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)
    positions = {id(record): index for group in groups.values() for index, record in enumerate(sorted(group, key=lambda item: item.timestamp))}
    sorted_groups = {key: sorted(group, key=lambda item: item.timestamp) for key, group in groups.items()}
    results: list[AdvancedDetectionResult] = []
    for record in records:
        ordered = sorted_groups[_group_key(record)]
        index = positions[id(record)]
        reference = ordered[max(0, index - window_size):index]
        if len(reference) < min_samples:
            results.append(AdvancedDetectionResult(record, False, "insufficient_prior_samples", "rolling_z_score", float(record.metric_value)))
            continue
        values = [float(item.metric_value) for item in reference]
        mean = sum(values) / len(values)
        deviation = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        score = (0.0 if float(record.metric_value) == mean else float("inf")) if deviation == 0 else abs((float(record.metric_value) - mean) / deviation)
        explanation = "rolling_z_score_exceeds_threshold" if score > z_threshold else "rolling_z_score_within_threshold"
        results.append(AdvancedDetectionResult(record, score > z_threshold, explanation, "rolling_z_score", float(record.metric_value), mean, score))
    return results


def detect_multivariate_anomalies(
    records: Sequence[TelemetryRecord],
    metric_names: Sequence[str],
    *,
    sensitivity: float = 3.0,
    min_samples: int = 3,
) -> list[AdvancedDetectionResult]:
    """Detect unusual standardized combinations by entity and timestamp.

    Complete entity/timestamp metric vectors form reference observations. A
    vector with a standardized Euclidean score above ``sensitivity`` is flagged.
    Incomplete vectors are explicitly reported as ``missing_metric_combination``.
    """
    _validate_common(records, sensitivity, min_samples)
    if not metric_names or len(set(metric_names)) != len(metric_names) or any(not isinstance(name, str) or not name.strip() for name in metric_names):
        raise DataValidationError("metric_names must contain unique non-empty strings.")
    grouped: dict[tuple[str | None, object], dict[str, TelemetryRecord]] = defaultdict(dict)
    for record in records:
        if record.metric_name in metric_names:
            grouped[(record.entity_id, record.timestamp)][record.metric_name] = record
    complete = [group for group in grouped.values() if all(name in group for name in metric_names)]
    results: list[AdvancedDetectionResult] = []
    for record in records:
        group = grouped[(_group_key(record)[0], record.timestamp)]
        if not all(name in group for name in metric_names):
            results.append(AdvancedDetectionResult(record, False, "missing_metric_combination", "multivariate", float(record.metric_value)))
            continue
        if len(complete) < min_samples:
            results.append(AdvancedDetectionResult(record, False, "insufficient_complete_vectors", "multivariate", float(record.metric_value)))
            continue
        means = {name: sum(float(item[name].metric_value) for item in complete) / len(complete) for name in metric_names}
        deviations = {name: sqrt(sum((float(item[name].metric_value) - means[name]) ** 2 for item in complete) / len(complete)) for name in metric_names}
        score = sqrt(sum(0.0 if deviations[name] == 0 else ((float(group[name].metric_value) - means[name]) / deviations[name]) ** 2 for name in metric_names))
        flagged = score > sensitivity
        explanation = "multivariate_score_exceeds_threshold" if flagged else "multivariate_score_within_threshold"
        results.append(AdvancedDetectionResult(record, flagged, explanation, "multivariate", float(record.metric_value), means[record.metric_name], score))
    return results
