import pytest

from anomaly_detector.alert_management import SeverityPolicy
from anomaly_detector.detection.generic_detector import NumericThreshold
from anomaly_detector.detection_configuration import (
    DetectionConfiguration,
    load_detection_configuration,
    save_detection_configuration,
)
from anomaly_detector.utils.exceptions import DataValidationError


def config(name="default"):
    return DetectionConfiguration(name, {"cpu": NumericThreshold(0, 100)}, 3.0, 5, 3.0, SeverityPolicy())


def test_configuration_create_retrieve_and_update(tmp_path):
    path = tmp_path / "config.sqlite"
    save_detection_configuration(path, config())
    loaded = load_detection_configuration(path, "default")
    assert loaded == config()
    updated = config()
    updated = DetectionConfiguration("default", updated.thresholds, 4.0, 10, 4.0, SeverityPolicy(1, 2, 4))
    save_detection_configuration(path, updated)
    assert load_detection_configuration(path, "default") == updated


@pytest.mark.parametrize("bad", [DetectionConfiguration("", {}, 3, 5, 3), DetectionConfiguration("bad", {"cpu": NumericThreshold(10, 1)}, 3, 5, 3), DetectionConfiguration("bad", {}, 0, 5, 3), DetectionConfiguration("bad", {}, 3, 0, 3)])
def test_invalid_configuration_rejected(bad):
    with pytest.raises(DataValidationError):
        bad.validate()
