"""Minimal real AWS client factory.

The factory is intentionally separate from the injected CloudWatch client
boundary. Importing this module does not import boto3 or create a client.
"""

from __future__ import annotations

from anomaly_detector.utils.exceptions import CloudWatchClientError


def create_cloudwatch_client(region_name: str | None = None):
    """Create a boto3 CloudWatch client using the standard AWS credential chain.

    ``region_name`` overrides the standard AWS region configuration when
    provided. Otherwise boto3 resolves region and credentials from its normal
    environment, profile, config-file, instance-role, or task-role sources.
    """
    if region_name is not None and (not isinstance(region_name, str) or not region_name.strip()):
        raise CloudWatchClientError("region_name must be a non-empty string or None.")
    try:
        import boto3
    except ImportError as exc:
        raise CloudWatchClientError(
            "boto3 is required to create a real CloudWatch client; install project dependencies first.",
        ) from exc
    try:
        session = boto3.Session(region_name=region_name)
        return session.client("cloudwatch")
    except Exception as exc:
        raise CloudWatchClientError("Unable to create the CloudWatch client.") from exc
