"""Validated normalized telemetry records for future metric ingestion.

This contract is intentionally separate from the weather CSV schema. It can
represent an AWS CloudWatch observation without making generic metrics valid
inputs for the weather detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from anomaly_detector.utils.exceptions import DataValidationError


@dataclass(frozen=True)
class TelemetryRecord:
    """One normalized numeric metric observation.

    ``timestamp`` must be a timezone-aware ``datetime`` and is stored in UTC.
    ``dimensions`` contains non-empty string keys and values and is copied into
    an immutable mapping. ``period`` is a positive CloudWatch-style period in
    whole seconds. ``is_anomaly`` is optional ground truth and is never inferred.
    """

    timestamp: datetime
    source_type: str
    metric_name: str
    metric_value: Real
    metric_unit: str | None = None
    entity_id: str | None = None
    is_anomaly: bool | None = None
    namespace: str | None = None
    dimensions: Mapping[str, str] = field(default_factory=dict)
    resource_arn: str | None = None
    account_id: str | None = None
    region: str | None = None
    period: int | None = None
    statistic: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the record without mutating caller data."""
        self._validate_timestamp()
        self._validate_non_empty_text(self.source_type, "source_type")
        self._validate_non_empty_text(self.metric_name, "metric_name")
        if isinstance(self.metric_value, bool) or not isinstance(self.metric_value, Real):
            raise DataValidationError("metric_value must be a numeric value, not a Boolean.")
        numeric_value = float(self.metric_value)
        if not isfinite(numeric_value):
            raise DataValidationError("metric_value must be finite.")
        object.__setattr__(self, "metric_value", numeric_value)

        for field_name in (
            "metric_unit", "entity_id", "namespace", "resource_arn", "account_id", "region", "statistic",
        ):
            value = getattr(self, field_name)
            if value is not None:
                self._validate_non_empty_text(value, field_name)
        if self.is_anomaly is not None and not isinstance(self.is_anomaly, bool):
            raise DataValidationError("is_anomaly must be True, False, or None.")
        self._validate_dimensions()
        if self.period is not None:
            if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period <= 0:
                raise DataValidationError("period must be a positive integer number of seconds.")

    def _validate_timestamp(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise DataValidationError("timestamp must be a datetime value.")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise DataValidationError("timestamp must be timezone-aware; naive timestamps are rejected.")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

    @staticmethod
    def _validate_non_empty_text(value: object, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise DataValidationError(f"{field_name} must be a non-empty string.")

    def _validate_dimensions(self) -> None:
        if not isinstance(self.dimensions, Mapping):
            raise DataValidationError("dimensions must be a mapping of string keys to string values.")
        copied_dimensions: dict[str, str] = {}
        for key, value in self.dimensions.items():
            self._validate_non_empty_text(key, "dimension key")
            self._validate_non_empty_text(value, f"dimension value for {key!r}")
            copied_dimensions[key] = value
        object.__setattr__(self, "dimensions", MappingProxyType(copied_dimensions))

    @property
    def canonical_identity(self) -> tuple[object, ...]:
        """Return stable identity components for future deduplication."""
        return (
            self.account_id,
            self.region,
            self.namespace,
            self.metric_name,
            tuple(sorted(self.dimensions.items())),
            self.resource_arn,
            self.entity_id,
            self.timestamp,
            self.period,
            self.statistic,
        )

    @property
    def time_series_identity(self) -> tuple[object, ...]:
        """Return identity fields for the underlying metric time series."""
        resource_identity = self.resource_arn if self.resource_arn is not None else self.entity_id
        return (
            self.source_type,
            self.account_id,
            self.region,
            self.namespace,
            self.metric_name,
            tuple(sorted(self.dimensions.items())),
            resource_identity,
            self.metric_unit,
        )

    @property
    def datapoint_identity(self) -> tuple[object, ...]:
        """Return time-series identity plus aggregation and UTC timestamp."""
        return self.time_series_identity + (
            self.timestamp,
            self.period,
            self.statistic,
        )
