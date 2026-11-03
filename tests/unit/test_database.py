import os

import pytest

from anomaly_detector.database import database_target, is_postgresql
from anomaly_detector.utils.exceptions import ConfigurationError


def test_database_target_detection():
    assert is_postgresql("postgresql://user:pass@localhost/db")
    assert is_postgresql("postgres://user:pass@localhost/db")
    assert not is_postgresql("data/telemetry.sqlite")


def test_database_url_takes_precedence_over_sqlite_path(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/db")
    monkeypatch.setenv("ANOMALY_DATABASE_PATH", "data/fallback.sqlite")
    monkeypatch.delenv("VERCEL", raising=False)
    assert database_target() == "postgresql://example.invalid/db"
    assert is_postgresql()


def test_sqlite_remains_default_for_local_development(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ANOMALY_DATABASE_PATH", str(tmp_path / "local.sqlite"))
    monkeypatch.delenv("VERCEL", raising=False)
    assert database_target() == str(tmp_path / "local.sqlite")
    assert not is_postgresql()


def test_vercel_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ANOMALY_DATABASE_PATH", "data/fallback.sqlite")
    monkeypatch.setenv("VERCEL", "1")
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        database_target()


def test_vercel_app_creation_does_not_fall_back_to_sqlite(monkeypatch, tmp_path):
    from anomaly_detector.api import create_app

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ANOMALY_DATABASE_PATH", str(tmp_path / "should-not-exist.sqlite"))
    monkeypatch.setenv("VERCEL", "1")
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        create_app()
    assert not (tmp_path / "should-not-exist.sqlite").exists()


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_postgresql_database_is_available():
    from anomaly_detector.data_ingestion.telemetry_persistence import initialize_database

    initialize_database(os.environ["TEST_DATABASE_URL"])
