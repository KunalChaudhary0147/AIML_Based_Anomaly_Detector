"""Deterministic in-memory deduplication for normalized telemetry records."""

from __future__ import annotations

from collections.abc import Iterable

from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import DataValidationError


def deduplicate_telemetry_records(records: Iterable[TelemetryRecord]) -> list[TelemetryRecord]:
    """Keep the first record for each ``datapoint_identity``.

    Input order is preserved. Later records with the same datapoint identity
    are discarded even when their metric values or ground-truth labels differ.
    The input iterable and its records are never mutated.
    """
    if isinstance(records, (str, bytes)):
        raise DataValidationError("records must be an iterable of TelemetryRecord objects.")
    try:
        iterator = iter(records)
    except TypeError as exc:
        raise DataValidationError("records must be an iterable of TelemetryRecord objects.") from exc

    unique_records: list[TelemetryRecord] = []
    seen_identities: set[tuple[object, ...]] = set()
    for record in iterator:
        if not isinstance(record, TelemetryRecord):
            raise DataValidationError("records must contain only TelemetryRecord objects.")
        identity = record.datapoint_identity
        if identity not in seen_identities:
            seen_identities.add(identity)
            unique_records.append(record)
    return unique_records
