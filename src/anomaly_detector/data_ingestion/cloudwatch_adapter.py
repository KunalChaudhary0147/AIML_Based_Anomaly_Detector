"""Offline adapter for CloudWatch GetMetricData-style responses.

This module converts handcrafted or mocked AWS response dictionaries into
validated :class:`TelemetryRecord` objects. It does not import an AWS SDK,
make network calls, infer metadata from result IDs, or deduplicate records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from typing import Any

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import DataValidationError

CLOUDWATCH_SOURCE_TYPE = "aws_cloudwatch"

_METADATA_FIELDS = frozenset({
    "namespace",
    "metric_name",
    "dimensions",
    "metric_unit",
    "account_id",
    "region",
    "resource_arn",
    "entity_id",
    "period",
    "statistic",
})


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{description} must be a mapping.")
    return value


def _require_sequence(value: object, description: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DataValidationError(f"{description} must be a sequence.")
    return value


def _status_error(result_id: str, result: Mapping[str, Any]) -> None:
    status = result.get("StatusCode", "Complete")
    if not isinstance(status, str) or not status.strip():
        raise DataValidationError(f"StatusCode for result {result_id!r} must be a non-empty string.")
    if status == "PartialData":
        return
    if status != "Complete":
        messages = result.get("Messages", [])
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise DataValidationError(f"Messages for result {result_id!r} must be a sequence.")
        details: list[str] = []
        for message in messages:
            message_mapping = _require_mapping(message, f"Message for result {result_id!r}")
            details.append(str(message_mapping.get("Value", message_mapping.get("Code", "unknown status message"))))
        suffix = f": {'; '.join(details)}" if details else ""
        raise DataValidationError(f"CloudWatch result {result_id!r} has status {status!r}{suffix}.")


def _normalize_timestamp(value: object, result_id: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise DataValidationError(f"Timestamp for result {result_id!r} must be a finite Unix timestamp.")
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise DataValidationError(f"Timestamp for result {result_id!r} is outside the supported range.") from exc


def _metadata_for(result_id: object, metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(result_id, str) or not result_id.strip():
        raise DataValidationError("Each CloudWatch result must have a non-empty string Id.")
    if result_id not in metadata:
        raise DataValidationError(f"No metadata mapping supplied for CloudWatch result Id {result_id!r}.")
    mapped = _require_mapping(metadata[result_id], f"Metadata for result {result_id!r}")
    unknown = set(mapped) - _METADATA_FIELDS
    if unknown:
        raise DataValidationError(f"Unsupported metadata fields for result {result_id!r}: {sorted(unknown)}")
    if "metric_name" not in mapped:
        raise DataValidationError(f"Metadata for result {result_id!r} requires metric_name.")
    return dict(mapped)


def cloudwatch_response_to_records(
    response: Mapping[str, Any],
    metric_metadata: Mapping[str, Mapping[str, Any]],
) -> list[TelemetryRecord]:
    """Convert a mocked GetMetricData response into normalized telemetry records.

    ``metric_metadata`` is keyed by the exact CloudWatch result ``Id``. IDs are
    never parsed to infer metric names or dimensions. A missing or failed result
    raises ``DataValidationError``. An absent/empty ``MetricDataResults`` list,
    or a result with ``StatusCode='Complete'`` and no pairs, returns no records.
    Every timestamp must be a timezone-aware ``datetime`` because validation is
    delegated to ``TelemetryRecord``.
    """
    response_mapping = _require_mapping(response, "CloudWatch response")
    metadata_mapping = _require_mapping(metric_metadata, "metric_metadata")
    results = response_mapping.get("MetricDataResults", [])
    results = _require_sequence(results, "MetricDataResults")
    if not results:
        return []

    records: list[TelemetryRecord] = []
    for raw_result in results:
        result = _require_mapping(raw_result, "CloudWatch metric result")
        result_id = result.get("Id")
        mapped_metadata = _metadata_for(result_id, metadata_mapping)
        _status_error(result_id, result)
        if "Timestamps" not in result or "Values" not in result:
            raise DataValidationError(f"Result {result_id!r} must contain Timestamps and Values.")
        timestamps = _require_sequence(result.get("Timestamps", []), f"Timestamps for result {result_id!r}")
        values = _require_sequence(result.get("Values", []), f"Values for result {result_id!r}")
        if len(timestamps) != len(values):
            raise DataValidationError(f"Timestamps and Values length mismatch for result {result_id!r}.")
        for timestamp, value in zip(timestamps, values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise DataValidationError(f"Value for result {result_id!r} must be numeric and non-Boolean.")
            records.append(
                TelemetryRecord(
                    timestamp=_normalize_timestamp(timestamp, result_id),
                    source_type=CLOUDWATCH_SOURCE_TYPE,
                    metric_value=value,
                    is_anomaly=None,
                    **mapped_metadata,
                ),
            )
    return records
