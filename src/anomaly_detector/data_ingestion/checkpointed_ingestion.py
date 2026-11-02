"""Restart-aware mocked CloudWatch ingestion orchestration.

This module composes the existing page client, SQLite persistence, and
checkpoint APIs. It does not create AWS clients, perform network calls, or
implement production restart scheduling.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Mapping

from anomaly_detector.data_ingestion.checkpoint_store import (
    Checkpoint,
    create_checkpoint,
    get_checkpoint,
    query_identity_fingerprint,
    update_checkpoint,
    update_checkpoint_in_connection,
)
from anomaly_detector.data_ingestion.cloudwatch_client import fetch_cloudwatch_page
from anomaly_detector.data_ingestion.telemetry_persistence import (
    PersistenceResult,
    TelemetryPersistenceError,
    _connect,
    _validate_schema,
    persist_telemetry_records_in_connection,
    _identity_json,
)
from anomaly_detector.utils.exceptions import (
    CloudWatchClientError,
    CloudWatchDeadlineError,
    CloudWatchPaginationError,
    CloudWatchRetryError,
    DataValidationError,
)


def _error_code(error: BaseException) -> str | None:
    cause = error.__cause__ or error
    response = getattr(cause, "response", None)
    if isinstance(response, Mapping):
        details = response.get("Error")
        if isinstance(details, Mapping) and isinstance(details.get("Code"), str):
            return details["Code"]
    return getattr(cause, "code", None) if isinstance(getattr(cause, "code", None), str) else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mark_failed(database_path: str | Path, checkpoint: Checkpoint, error: BaseException) -> None:
    try:
        update_checkpoint(
            database_path,
            checkpoint.run_id,
            expected_query_fingerprint=checkpoint.query_fingerprint,
            new_status="failed",
            error=error,
        )
    except Exception as failure_error:
        error.add_note(f"Unable to record checkpoint failure: {failure_error}")


def run_checkpointed_ingestion(
    database_path: str | Path,
    client: Any,
    request_parameters: Mapping[str, Any],
    metric_metadata: Mapping[str, Mapping[str, Any]],
    *,
    query_configuration: Mapping[str, Any],
    source_type: str,
    window_start: datetime,
    window_end: datetime,
    run_id: str | None = None,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    jitter_seconds: float = 0.1,
    max_pages: int = 100,
    max_records: int = 100_000,
    collection_timeout_seconds: float | None = None,
    sleep=None,
    random_value=None,
) -> Checkpoint:
    """Fetch pages and atomically persist each page with checkpoint progress."""
    fingerprint = query_identity_fingerprint(query_configuration, window_start, window_end)
    if collection_timeout_seconds is not None:
        if isinstance(collection_timeout_seconds, bool) or not isinstance(collection_timeout_seconds, (int, float)) or collection_timeout_seconds <= 0:
            raise DataValidationError("collection_timeout_seconds must be a finite positive number or None.")
    ingestion_start = time.monotonic()
    if run_id is None:
        checkpoint = create_checkpoint(
            database_path,
            query_configuration=query_configuration,
            source_type=source_type,
            window_start=window_start,
            window_end=window_end,
        )
    else:
        try:
            checkpoint = get_checkpoint(str(database_path), run_id)
        except TelemetryPersistenceError:
            checkpoint = create_checkpoint(
                database_path,
                run_id=run_id,
                query_configuration=query_configuration,
                source_type=source_type,
                window_start=window_start,
                window_end=window_end,
            )
        if checkpoint.query_fingerprint != fingerprint:
            raise DataValidationError("Checkpoint query fingerprint does not match the requested query.")
        if checkpoint.run_status == "completed":
            return checkpoint

    try:
        if checkpoint.run_status == "pending":
            checkpoint = update_checkpoint(
                database_path, checkpoint.run_id,
                expected_query_fingerprint=fingerprint,
                new_status="running",
            )
        token = checkpoint.pagination_token
        page_count = checkpoint.committed_page_count
        record_count = checkpoint.committed_record_count
        seen_tokens = {token} if token else set()
        replayed_invalid_token = False

        while True:
            if collection_timeout_seconds is not None and time.monotonic() - ingestion_start >= collection_timeout_seconds:
                raise CloudWatchDeadlineError("CloudWatch collection deadline exceeded before the next page.")
            if page_count >= max_pages:
                raise CloudWatchPaginationError(f"CloudWatch pagination exceeded max_pages={max_pages}.")
            request = dict(request_parameters)
            if token:
                request["NextToken"] = token
            try:
                page = fetch_cloudwatch_page(
                    client, request, metric_metadata,
                    max_attempts=max_attempts,
                    base_delay_seconds=base_delay_seconds,
                    jitter_seconds=jitter_seconds,
                    sleep=sleep or __import__("time").sleep,
                    random_value=random_value or __import__("random").random,
                    collection_timeout_seconds=collection_timeout_seconds,
                    monotonic=time.monotonic,
                    start_time=ingestion_start,
                )
            except CloudWatchClientError as error:
                if _error_code(error) == "InvalidNextToken" and token and not replayed_invalid_token:
                    replayed_invalid_token = True
                    checkpoint = update_checkpoint(
                        database_path, checkpoint.run_id,
                        expected_query_fingerprint=fingerprint,
                        new_status="needs_recovery",
                        error=error,
                    )
                    checkpoint = update_checkpoint(
                        database_path, checkpoint.run_id,
                        expected_query_fingerprint=fingerprint,
                        new_status="running",
                        clear_pagination_token=True,
                    )
                    token = None
                    seen_tokens.clear()
                    continue
                raise

            next_token = page.next_token
            if next_token and next_token in seen_tokens:
                raise CloudWatchPaginationError(f"Repeated CloudWatch pagination token: {next_token!r}.")
            if record_count + len(page.records) > max_records:
                raise DataValidationError(f"CloudWatch collection exceeded max_records={max_records}.")
            terminal = not next_token
            next_page_count = page_count + 1
            new_status = "completed" if terminal else "page_committed"

            connection = None
            try:
                connection = _connect(database_path)
                _validate_schema(connection)
                connection.execute("BEGIN IMMEDIATE")
                persist_result = persist_telemetry_records_in_connection(connection, page.records)
                _ = persist_result
                checkpoint = update_checkpoint_in_connection(
                    connection,
                    checkpoint.run_id,
                    expected_query_fingerprint=fingerprint,
                    new_status=new_status,
                    pagination_token=next_token,
                    committed_page_count=next_page_count,
                    committed_record_count=record_count + persist_result.inserted,
                    last_committed_datapoint_identity=(_identity_json(page.records[-1].datapoint_identity) if page.records else checkpoint.last_committed_datapoint_identity),
                    now=_now(),
                )
                connection.commit()
            except Exception:
                if connection is not None:
                    connection.rollback()
                raise
            finally:
                if connection is not None:
                    connection.close()

            page_count = next_page_count
            record_count = record_count + persist_result.inserted
            if terminal:
                return checkpoint
            seen_tokens.add(next_token)
            token = next_token
    except Exception as error:
        _mark_failed(database_path, checkpoint, error)
        raise
