"""Independent Phase 2.2 tests for the opt-in cross-feature rule."""

import pandas as pd
import pytest

from anomaly_detector.config import settings
from anomaly_detector.detection.detector import (
    HEAT_STRESS_RULE,
    PREDICTION_COLUMN,
    REASON_COLUMN,
    detect_anomalies,
)
from anomaly_detector.utils.exceptions import DataValidationError


def observation(**overrides):
    values = {
        "timestamp": pd.Timestamp("2026-08-01T12:00:00Z"),
        "station_id": "PHASE22_001",
        "temperature": 20.0,
        "humidity": 50.0,
        "wind_speed": 5.0,
        "atmospheric_pressure": 1013.0,
        "is_anomaly": False,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_global_behavior_is_unchanged_without_opt_in_rule():
    data = observation(temperature=40.0, humidity=90.0)
    default_result = detect_anomalies(data)
    opt_in_result = detect_anomalies(data, cross_feature_rules=(HEAT_STRESS_RULE,))
    assert default_result[PREDICTION_COLUMN].tolist() == [False]
    assert opt_in_result[PREDICTION_COLUMN].tolist() == [True]
    assert opt_in_result[REASON_COLUMN].tolist() == ["heat_stress_condition"]


def test_individually_valid_heat_stress_condition_is_detected():
    data = observation(
        temperature=settings.HEAT_STRESS_TEMPERATURE_MIN,
        humidity=settings.HEAT_STRESS_HUMIDITY_MIN,
    )
    result = detect_anomalies(data, cross_feature_rules=(HEAT_STRESS_RULE,))
    assert result[PREDICTION_COLUMN].tolist() == [True]


def test_cross_feature_rule_requires_both_features():
    warm_dry = observation(temperature=settings.HEAT_STRESS_TEMPERATURE_MIN + 1, humidity=settings.HEAT_STRESS_HUMIDITY_MIN - 1)
    cool_humid = observation(temperature=settings.HEAT_STRESS_TEMPERATURE_MIN - 1, humidity=settings.HEAT_STRESS_HUMIDITY_MIN + 1)
    result = detect_anomalies(pd.concat([warm_dry, cool_humid]), cross_feature_rules=(HEAT_STRESS_RULE,))
    assert result[PREDICTION_COLUMN].tolist() == [False, False]


def test_cross_feature_rule_preserves_input_and_ignores_ground_truth():
    false_label = observation(temperature=40.0, humidity=90.0, is_anomaly=False)
    true_label = observation(temperature=40.0, humidity=90.0, is_anomaly=True)
    original = false_label.copy(deep=True)
    false_result = detect_anomalies(false_label, cross_feature_rules=(HEAT_STRESS_RULE,))
    true_result = detect_anomalies(true_label, cross_feature_rules=(HEAT_STRESS_RULE,))
    assert false_result[PREDICTION_COLUMN].tolist() == true_result[PREDICTION_COLUMN].tolist() == [True]
    assert false_label.equals(original)
    assert true_result["is_anomaly"].tolist() == [True]


def test_cross_feature_rule_rejects_invalid_configuration_and_inputs():
    invalid_rule = type(HEAT_STRESS_RULE)("unknown", 1.0, "humidity", 2.0, "invalid")
    with pytest.raises(DataValidationError):
        detect_anomalies(observation(), cross_feature_rules=(invalid_rule,))
    with pytest.raises(DataValidationError):
        detect_anomalies(observation(temperature=float("inf")), cross_feature_rules=(HEAT_STRESS_RULE,))
