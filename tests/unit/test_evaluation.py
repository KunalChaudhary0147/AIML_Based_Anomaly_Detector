import pandas as pd
import pytest

from anomaly_detector.evaluation import evaluate_predictions
from anomaly_detector.utils.exceptions import DataValidationError


def test_evaluation_metrics():
    data = pd.DataFrame({
        "is_anomaly": [True, True, False, False],
        "predicted_is_anomaly": [True, False, True, False],
    })
    result = evaluate_predictions(data)
    assert result is not None
    assert result.true_positive == 1
    assert result.false_positive == 1
    assert result.true_negative == 1
    assert result.false_negative == 1
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1_score == 0.5


def test_evaluation_zero_denominators_and_missing_labels():
    no_predictions = pd.DataFrame({"is_anomaly": [False, False], "predicted_is_anomaly": [False, False]})
    result = evaluate_predictions(no_predictions)
    assert result is not None
    assert result.precision == result.recall == result.f1_score == 0.0
    assert evaluate_predictions(pd.DataFrame({"predicted_is_anomaly": [True]})) is None


@pytest.mark.parametrize(
    "values",
    [["False"], [0], [1], [None]],
)
def test_evaluation_rejects_non_boolean_prediction_values(values):
    data = pd.DataFrame({"predicted_is_anomaly": values})
    with pytest.raises(DataValidationError):
        evaluate_predictions(data)


def test_evaluation_rejects_non_boolean_ground_truth_values():
    data = pd.DataFrame({"predicted_is_anomaly": [True], "is_anomaly": ["False"]})
    with pytest.raises(DataValidationError):
        evaluate_predictions(data)


def test_evaluation_accepts_actual_boolean_values():
    data = pd.DataFrame({"predicted_is_anomaly": [True, False], "is_anomaly": [True, False]})
    result = evaluate_predictions(data)
    assert result is not None
    assert result.true_positive == 1
    assert result.true_negative == 1
