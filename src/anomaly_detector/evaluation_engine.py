"""Evaluation metrics for AdvancedDetectionResult collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from anomaly_detector.detection.advanced_detector import AdvancedDetectionResult
from anomaly_detector.utils.exceptions import DataValidationError


@dataclass(frozen=True)
class BinaryEvaluation:
    """Binary classification counts and metrics."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float


def evaluate_advanced_results(results: Iterable[AdvancedDetectionResult]) -> BinaryEvaluation | None:
    """Evaluate results using optional ``TelemetryRecord.is_anomaly`` labels."""
    result_list = list(results)
    if any(not isinstance(result, AdvancedDetectionResult) for result in result_list):
        raise DataValidationError("results must contain only AdvancedDetectionResult objects.")
    if not result_list or any(result.record.is_anomaly is None for result in result_list):
        return None
    if any(not isinstance(result.record.is_anomaly, bool) for result in result_list):
        raise DataValidationError("Ground-truth labels must be Boolean when present.")
    true_positive = sum(result.predicted_is_anomaly and result.record.is_anomaly for result in result_list)
    false_positive = sum(result.predicted_is_anomaly and not result.record.is_anomaly for result in result_list)
    true_negative = sum(not result.predicted_is_anomaly and not result.record.is_anomaly for result in result_list)
    false_negative = sum(not result.predicted_is_anomaly and result.record.is_anomaly for result in result_list)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(result_list)
    return BinaryEvaluation(true_positive, false_positive, true_negative, false_negative, precision, recall, f1, accuracy)
