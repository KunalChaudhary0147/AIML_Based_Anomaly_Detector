import sys
import types

import pytest

from anomaly_detector.data_ingestion.aws_client import create_cloudwatch_client
from anomaly_detector.utils.exceptions import CloudWatchClientError


def test_factory_uses_explicit_region_and_cloudwatch_service(monkeypatch):
    calls = {}
    fake_client = object()

    class FakeSession:
        def __init__(self, region_name=None):
            calls["region"] = region_name

        def client(self, service_name):
            calls["service"] = service_name
            return fake_client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=FakeSession))
    assert create_cloudwatch_client("eu-west-1") is fake_client
    assert calls == {"region": "eu-west-1", "service": "cloudwatch"}


def test_factory_allows_standard_boto3_region_resolution(monkeypatch):
    calls = {}

    class FakeSession:
        def __init__(self, region_name=None):
            calls["region"] = region_name

        def client(self, service_name):
            calls["service"] = service_name
            return "client"

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=FakeSession))
    assert create_cloudwatch_client() == "client"
    assert calls == {"region": None, "service": "cloudwatch"}


@pytest.mark.parametrize("region", ["", " ", 123, False])
def test_factory_rejects_invalid_explicit_region(monkeypatch, region):
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=lambda **_: None))
    with pytest.raises(CloudWatchClientError):
        create_cloudwatch_client(region)


def test_factory_reports_missing_boto3(monkeypatch):
    real_import = __import__

    def missing_boto3(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("missing boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_boto3)
    with pytest.raises(CloudWatchClientError, match="boto3"):
        create_cloudwatch_client()
