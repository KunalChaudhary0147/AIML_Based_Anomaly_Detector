import os
import subprocess
import sys

from anomaly_detector.config import settings


def test_default_configuration_is_resolved():
    assert settings.PROJECT_ROOT.is_dir()
    assert settings.RAW_DATA_DIR == settings.PROJECT_ROOT / "data" / "raw"
    assert settings.EXPECTED_COLUMNS[-1] == "is_anomaly"


def test_invalid_environment_value_reports_configuration_error():
    environment = os.environ.copy()
    environment["ANOMALY_STATION_COUNT"] = "not-an-integer"
    result = subprocess.run(
        [sys.executable, "-c", "import anomaly_detector.config.settings"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode != 0
    assert "ConfigurationError" in result.stderr


def test_contradictory_detection_thresholds_are_rejected():
    environment = os.environ.copy()
    environment["ANOMALY_TEMPERATURE_MIN"] = "60"
    environment["ANOMALY_TEMPERATURE_MAX"] = "50"
    result = subprocess.run(
        [sys.executable, "-c", "import anomaly_detector.config.settings"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode != 0
    assert "lower bound exceeds upper bound" in result.stderr
