"""Validation of rule behavior on independent synthetic observations."""

from __future__ import annotations

import pandas as pd
import pytest

from anomaly_detector.config import settings
from anomaly_detector.detection.detector import PREDICTION_COLUMN, REASON_COLUMN, detect_anomalies
from anomaly_detector.utils.exceptions import DataValidationError


def unseen_observation(**overrides: object) -> dict[str, object]:
    """Build a new observation that is independent of the stored datasets."""
    observation: dict[str, object] = {
        "timestamp": pd.Timestamp("2026-07-15T12:00:00Z"),
        "station_id": "UNSEEN_STATION_001",
        "temperature": (settings.TEMPERATURE_MIN + settings.TEMPERATURE_MAX) / 2,
        "humidity": (settings.HUMIDITY_MIN + settings.HUMIDITY_MAX) / 2,
        "wind_speed": (settings.WIND_SPEED_MIN + settings.WIND_SPEED_MAX) / 2,
        "atmospheric_pressure": (settings.PRESSURE_MIN + settings.PRESSURE_MAX) / 2,
        "is_anomaly": False,
    }
    observation.update(overrides)
    return observation


def frame(*observations: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(observations)


def prediction(data: pd.DataFrame) -> pd.Series:
    return detect_anomalies(data)[PREDICTION_COLUMN]


def test_clearly_normal_unseen_observation_is_normal():
    result = detect_anomalies(frame(unseen_observation()))
    assert result[PREDICTION_COLUMN].tolist() == [False]
    assert result[REASON_COLUMN].tolist() == [""]


@pytest.mark.parametrize(
    ("field", "value", "rule_id"),
    [
        ("temperature", settings.TEMPERATURE_MAX + 0.01, "temperature_range"),
        ("wind_speed", settings.WIND_SPEED_MAX + 0.01, "wind_speed_range"),
        ("atmospheric_pressure", settings.PRESSURE_MAX + 0.01, "pressure_range"),
    ],
)
def test_upper_bound_violations_are_detected(field: str, value: float, rule_id: str):
    result = detect_anomalies(frame(unseen_observation(**{field: value})))
    assert result[PREDICTION_COLUMN].tolist() == [True]
    assert result[REASON_COLUMN].tolist() == [rule_id]


@pytest.mark.parametrize(
    ("field", "value", "rule_id"),
    [
        ("temperature", settings.TEMPERATURE_MIN - 0.01, "temperature_range"),
        ("humidity", settings.HUMIDITY_MIN - 0.01, "humidity_range"),
        ("wind_speed", settings.WIND_SPEED_MIN - 0.01, "wind_speed_range"),
        ("atmospheric_pressure", settings.PRESSURE_MIN - 0.01, "pressure_range"),
    ],
)
def test_lower_bound_violations_are_detected(field: str, value: float, rule_id: str):
    result = detect_anomalies(frame(unseen_observation(**{field: value})))
    assert result[PREDICTION_COLUMN].tolist() == [True]
    assert result[REASON_COLUMN].tolist() == [rule_id]


def test_multiple_violations_report_all_applicable_rules():
    result = detect_anomalies(
        frame(
            unseen_observation(
                temperature=settings.TEMPERATURE_MAX + 1,
                humidity=settings.HUMIDITY_MIN - 1,
                atmospheric_pressure=settings.PRESSURE_MAX + 1,
            ),
        ),
    )
    assert result[PREDICTION_COLUMN].tolist() == [True]
    assert result[REASON_COLUMN].tolist() == ["temperature_range;humidity_range;pressure_range"]


def test_every_inclusive_threshold_boundary_is_normal():
    boundary_observations = []
    for field, lower, upper in (
        ("temperature", settings.TEMPERATURE_MIN, settings.TEMPERATURE_MAX),
        ("humidity", settings.HUMIDITY_MIN, settings.HUMIDITY_MAX),
        ("wind_speed", settings.WIND_SPEED_MIN, settings.WIND_SPEED_MAX),
        ("atmospheric_pressure", settings.PRESSURE_MIN, settings.PRESSURE_MAX),
    ):
        boundary_observations.extend(
            [
                unseen_observation(station_id=f"BOUNDARY_{field}_LOW", **{field: lower}),
                unseen_observation(station_id=f"BOUNDARY_{field}_HIGH", **{field: upper}),
            ],
        )
    result = detect_anomalies(frame(*boundary_observations))
    assert result[PREDICTION_COLUMN].tolist() == [False] * 8
    assert result[REASON_COLUMN].tolist() == [""] * 8


def test_mixed_batch_preserves_order_and_predictions():
    data = frame(
        unseen_observation(station_id="MIXED_NORMAL"),
        unseen_observation(station_id="MIXED_HOT", temperature=settings.TEMPERATURE_MAX + 1),
        unseen_observation(station_id="MIXED_WIND", wind_speed=settings.WIND_SPEED_MAX + 1),
        unseen_observation(station_id="MIXED_NORMAL_2"),
    )
    result = detect_anomalies(data)
    assert result.station_id.tolist() == ["MIXED_NORMAL", "MIXED_HOT", "MIXED_WIND", "MIXED_NORMAL_2"]
    assert result[PREDICTION_COLUMN].tolist() == [False, True, True, False]


def test_ground_truth_does_not_influence_unseen_predictions_or_input():
    false_label = frame(unseen_observation(is_anomaly=False))
    true_label = frame(unseen_observation(is_anomaly=True))
    original = false_label.copy(deep=True)
    false_result = detect_anomalies(false_label)
    true_result = detect_anomalies(true_label)
    assert false_result[PREDICTION_COLUMN].tolist() == true_result[PREDICTION_COLUMN].tolist()
    assert false_result[REASON_COLUMN].tolist() == true_result[REASON_COLUMN].tolist()
    assert false_label.equals(original)
    assert false_result["is_anomaly"].tolist() == [False]
    assert true_result["is_anomaly"].tolist() == [True]


def test_invalid_unseen_inputs_fail():
    missing_column = frame(unseen_observation()).drop(columns=["temperature"])
    missing_value = frame(unseen_observation(temperature=float("nan")))
    non_finite_value = frame(unseen_observation(humidity=float("inf")))
    for invalid_data in (missing_column, missing_value, non_finite_value):
        with pytest.raises(DataValidationError):
            detect_anomalies(invalid_data)
