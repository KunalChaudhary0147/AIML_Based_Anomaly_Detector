import os

import pytest

from anomaly_detector.database import is_postgresql


def test_database_target_detection():
    assert is_postgresql("postgresql://user:pass@localhost/db")
    assert is_postgresql("postgres://user:pass@localhost/db")
    assert not is_postgresql("data/telemetry.sqlite")


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_postgresql_database_is_available():
    from anomaly_detector.data_ingestion.telemetry_persistence import initialize_database

    initialize_database(os.environ["TEST_DATABASE_URL"])
