"""Dependency-injected, offline-testable CloudWatch client boundary."""

from __future__ import annotations

import random
import time
from copy import deepcopy
from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any
from dataclasses import dataclass

from anomaly_detector.data_ingestion.cloudwatch_adapter import cloudwatch_response_to_records
from anomaly_detector.data_ingestion.telemetry import TelemetryRecord
from anomaly_detector.utils.exceptions import (
    CloudWatchClientError,
    CloudWatchDeadlineError,
    CloudWatchPaginationError,
    CloudWatchRecordLimitError,
    CloudWatchRetryError,
    DataValidationError,
)

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BASE_DELAY_SECONDS = 0.5
_DEFAULT_JITTER_SECONDS = 0.1
_DEFAULT_MAX_PAGES = 100
_DEFAULT_MAX_RECORDS = 100_000
_RETRYABLE_CODES = frozenset({
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
    "InternalError",
    "InternalFailure",
    "InternalServiceError",
    "ServiceUnavailable",
    "ServiceUnavailableException",
    "RequestTimeout",
    "RequestTimeoutException",
})
_NON_RETRYABLE_CODES = frozenset({
    "AccessDenied",
    "AccessDeniedException",
    "Unauthorized",
    "UnauthorizedException",
    "ExpiredTokenException",
    "NotAuthorized",
    "UnrecognizedClientException",
    "InvalidClientTokenId",
    "ValidationException",
    "ValidationError",
    "InvalidNextToken",
    "InvalidParameter",
    "InvalidParameterValue",
    "ResourceNotFoundException",
})


@dataclass(frozen=True)
class CloudWatchPage:
    """Validated records and continuation token for one response page."""

    records: list[TelemetryRecord]
    next_token: str | None


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        details = response.get("Error")
        if isinstance(details, Mapping):
            code = details.get("Code")
            if isinstance(code, str):
                return code
        status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode") if isinstance(response.get("ResponseMetadata"), Mapping) else None
        if isinstance(status_code, int) and status_code >= 500:
            return str(status_code)
    code = getattr(error, "code", None)
    return code if isinstance(code, str) else None


def is_retryable_cloudwatch_error(error: BaseException) -> bool:
    """Classify only known transport, throttling, and temporary service errors."""
    if isinstance(error, DataValidationError):
        return False
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    code = _error_code(error)
    if code in _NON_RETRYABLE_CODES:
        return False
    if code in _RETRYABLE_CODES:
        return True
    if code is not None and code.isdigit() and int(code) >= 500:
        return True
    name = type(error).__name__.lower()
    return any(marker in name for marker in ("timeout", "connection", "throttl"))


def _validate_retry_settings(max_attempts: int, base_delay_seconds: float, jitter_seconds: float) -> None:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise CloudWatchClientError("max_attempts must be a positive integer.")
    if isinstance(base_delay_seconds, bool) or not isinstance(base_delay_seconds, (int, float)) or not isfinite(float(base_delay_seconds)) or base_delay_seconds < 0:
        raise CloudWatchClientError("base_delay_seconds must be a finite non-negative number.")
    if isinstance(jitter_seconds, bool) or not isinstance(jitter_seconds, (int, float)) or not isfinite(float(jitter_seconds)) or jitter_seconds < 0:
        raise CloudWatchClientError("jitter_seconds must be a finite non-negative number.")


def _validate_collection_settings(max_records: int, collection_timeout_seconds: float | None) -> None:
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
        raise CloudWatchRecordLimitError("max_records must be a positive integer.")
    if collection_timeout_seconds is not None:
        if isinstance(collection_timeout_seconds, bool) or not isinstance(collection_timeout_seconds, (int, float)):
            raise CloudWatchDeadlineError("collection_timeout_seconds must be a finite positive number or None.")
        if not isfinite(float(collection_timeout_seconds)) or collection_timeout_seconds <= 0:
            raise CloudWatchDeadlineError("collection_timeout_seconds must be a finite positive number or None.")


def fetch_cloudwatch_records(
    client: Any,
    request_parameters: Mapping[str, Any],
    metric_metadata: Mapping[str, Mapping[str, Any]],
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
    jitter_seconds: float = _DEFAULT_JITTER_SECONDS,
    max_pages: int = _DEFAULT_MAX_PAGES,
    max_records: int = _DEFAULT_MAX_RECORDS,
    collection_timeout_seconds: float | None = None,
    sleep: Callable[[float], object] = time.sleep,
    random_value: Callable[[], float] = random.random,
    monotonic: Callable[[], float] = time.monotonic,
) -> list[TelemetryRecord]:
    """Collect paginated responses from an injected client and adapt them.

    A caller-supplied nonempty ``NextToken`` is treated as an explicit resume
    point and is preserved on the first request; callers starting from the
    beginning should omit it. Each page is retried independently and adapted
    before the next page is requested. A missing, ``None``, or empty
    ``NextToken`` ends collection. Request parameters are copied before invocation. Retry delay is
    ``base_delay_seconds * 2**(attempt - 1) + random_value() * jitter_seconds``
    and is applied only before another attempt. No AWS SDK or credentials are
    created here. Adapter validation errors are permanent and never retried.
    """
    _validate_retry_settings(max_attempts, base_delay_seconds, jitter_seconds)
    _validate_collection_settings(max_records, collection_timeout_seconds)
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
        raise CloudWatchPaginationError("max_pages must be a positive integer.")
    if not isinstance(request_parameters, Mapping):
        raise CloudWatchClientError("request_parameters must be a mapping.")
    if not callable(sleep) or not callable(random_value):
        raise CloudWatchClientError("sleep and random_value must be callable.")
    if not callable(monotonic):
        raise CloudWatchDeadlineError("monotonic must be callable.")
    start_time = monotonic()
    request = deepcopy(dict(request_parameters))
    collected: list[TelemetryRecord] = []
    initial_token = request.get("NextToken")
    if initial_token is not None and not isinstance(initial_token, str):
        raise CloudWatchPaginationError("request_parameters NextToken must be a string, None, or empty.")
    next_token: str | None = initial_token if initial_token else None
    seen_tokens: set[str] = {initial_token} if initial_token else set()

    for page_number in range(1, max_pages + 1):
        if collection_timeout_seconds is not None and monotonic() - start_time >= collection_timeout_seconds:
            raise CloudWatchDeadlineError("CloudWatch collection deadline exceeded before the next page request.")
        page_request = deepcopy(request)
        if next_token is not None:
            page_request["NextToken"] = next_token
        response = _request_with_retries(
            client,
            page_request,
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            jitter_seconds=jitter_seconds,
            sleep=sleep,
            random_value=random_value,
            monotonic=monotonic,
            start_time=start_time,
            collection_timeout_seconds=collection_timeout_seconds,
        )
        if not isinstance(response, Mapping):
            raise DataValidationError("CloudWatch response must be a mapping.")
        raw_next_token = response.get("NextToken")
        statuses = {
            result.get("StatusCode")
            for result in response.get("MetricDataResults", [])
            if isinstance(result, Mapping)
        }
        if "PartialData" in statuses and (raw_next_token is None or raw_next_token == ""):
            raise DataValidationError("CloudWatch PartialData result requires NextToken for completion.")
        if "InternalError" in statuses and raw_next_token not in (None, ""):
            next_token = raw_next_token
            if not isinstance(next_token, str):
                raise CloudWatchPaginationError("CloudWatch NextToken must be a string, None, or empty.")
            if next_token in seen_tokens:
                raise CloudWatchPaginationError(f"Repeated CloudWatch pagination token: {next_token!r}.")
            if page_number == max_pages:
                raise CloudWatchPaginationError(f"CloudWatch pagination exceeded max_pages={max_pages}.")
            seen_tokens.add(next_token)
            continue
        page_records = cloudwatch_response_to_records(response, metric_metadata)
        if len(collected) + len(page_records) > max_records:
            raise CloudWatchRecordLimitError(f"CloudWatch collection exceeded max_records={max_records}.")
        collected.extend(page_records)
        if raw_next_token is None or raw_next_token == "":
            return collected


        def fetch_cloudwatch_page(
            client: Any,
            request_parameters: Mapping[str, Any],
            metric_metadata: Mapping[str, Mapping[str, Any]],
            *,
            max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
            base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
            jitter_seconds: float = _DEFAULT_JITTER_SECONDS,
            sleep: Callable[[float], object] = time.sleep,
            random_value: Callable[[], float] = random.random,
        ) -> CloudWatchPage:
            """Fetch, retry, and validate exactly one CloudWatch page."""
            _validate_retry_settings(max_attempts, base_delay_seconds, jitter_seconds)
            if not isinstance(request_parameters, Mapping):
                raise CloudWatchClientError("request_parameters must be a mapping.")
            if not callable(sleep) or not callable(random_value):
                raise CloudWatchClientError("sleep and random_value must be callable.")
            response = _request_with_retries(
                client,
                deepcopy(dict(request_parameters)),
                max_attempts=max_attempts,
                base_delay_seconds=base_delay_seconds,
                jitter_seconds=jitter_seconds,
                sleep=sleep,
                random_value=random_value,
                monotonic=time.monotonic,
                start_time=time.monotonic(),
                collection_timeout_seconds=None,
            )
            raw_next_token = response.get("NextToken")
            statuses = {result.get("StatusCode") for result in response.get("MetricDataResults", []) if isinstance(result, Mapping)}
            if "PartialData" in statuses and not raw_next_token:
                raise DataValidationError("CloudWatch PartialData result requires NextToken for completion.")
            if "InternalError" in statuses and raw_next_token:
                return CloudWatchPage([], raw_next_token)
            records = cloudwatch_response_to_records(response, metric_metadata)
            if raw_next_token is not None and not isinstance(raw_next_token, str):
                raise CloudWatchPaginationError("CloudWatch NextToken must be a string, None, or empty.")
            return CloudWatchPage(records, raw_next_token or None)
        if not isinstance(raw_next_token, str):
            raise CloudWatchPaginationError("CloudWatch NextToken must be a string, None, or empty.")
        if raw_next_token in seen_tokens:
            raise CloudWatchPaginationError(f"Repeated CloudWatch pagination token: {raw_next_token!r}.")
        if page_number == max_pages:
            raise CloudWatchPaginationError(f"CloudWatch pagination exceeded max_pages={max_pages}.")
        seen_tokens.add(raw_next_token)
        next_token = raw_next_token

    raise CloudWatchPaginationError("CloudWatch pagination ended without a terminal response.")


def fetch_cloudwatch_page(
    client: Any,
    request_parameters: Mapping[str, Any],
    metric_metadata: Mapping[str, Mapping[str, Any]],
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
    jitter_seconds: float = _DEFAULT_JITTER_SECONDS,
    sleep: Callable[[float], object] = time.sleep,
    random_value: Callable[[], float] = random.random,
    collection_timeout_seconds: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    start_time: float | None = None,
) -> CloudWatchPage:
    """Fetch, retry, and validate exactly one CloudWatch page."""
    _validate_retry_settings(max_attempts, base_delay_seconds, jitter_seconds)
    _validate_collection_settings(1, collection_timeout_seconds)
    if not isinstance(request_parameters, Mapping):
        raise CloudWatchClientError("request_parameters must be a mapping.")
    response = _request_with_retries(
        client, deepcopy(dict(request_parameters)),
        max_attempts=max_attempts, base_delay_seconds=base_delay_seconds,
        jitter_seconds=jitter_seconds, sleep=sleep, random_value=random_value,
        monotonic=monotonic,
        start_time=monotonic() if start_time is None else start_time,
        collection_timeout_seconds=collection_timeout_seconds,
    )
    raw_next_token = response.get("NextToken")
    statuses = {result.get("StatusCode") for result in response.get("MetricDataResults", []) if isinstance(result, Mapping)}
    if "PartialData" in statuses and not raw_next_token:
        raise DataValidationError("CloudWatch PartialData result requires NextToken for completion.")
    if "InternalError" in statuses and raw_next_token:
        return CloudWatchPage([], raw_next_token)
    records = cloudwatch_response_to_records(response, metric_metadata)
    if raw_next_token is not None and not isinstance(raw_next_token, str):
        raise CloudWatchPaginationError("CloudWatch NextToken must be a string, None, or empty.")
    return CloudWatchPage(records, raw_next_token or None)


def _request_with_retries(
    client: Any,
    request: Mapping[str, Any],
    *,
    max_attempts: int,
    base_delay_seconds: float,
    jitter_seconds: float,
    sleep: Callable[[float], object],
    random_value: Callable[[], float],
    monotonic: Callable[[], float],
    start_time: float,
    collection_timeout_seconds: float | None,
) -> Mapping[str, Any]:
    """Request one page using the existing bounded retry policy."""
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        if collection_timeout_seconds is not None and monotonic() - start_time >= collection_timeout_seconds:
            raise CloudWatchDeadlineError("CloudWatch collection deadline exceeded before a page attempt.")
        try:
            response = client.get_metric_data(**request)
            if not isinstance(response, Mapping):
                raise DataValidationError("CloudWatch response must be a mapping.")
            return response
        except DataValidationError:
            raise
        except Exception as error:
            last_error = error
            if not is_retryable_cloudwatch_error(error):
                raise CloudWatchClientError("CloudWatch request failed with a non-retryable error.") from error
            if attempt == max_attempts:
                raise CloudWatchRetryError(f"CloudWatch request failed after {max_attempts} attempts.") from error
            jitter_sample = float(random_value())
            if not 0 <= jitter_sample <= 1:
                raise CloudWatchClientError("random_value must return a value between 0 and 1.")
            delay = float(base_delay_seconds) * (2 ** (attempt - 1)) + jitter_sample * float(jitter_seconds)
            sleep(delay)
            if collection_timeout_seconds is not None and monotonic() - start_time >= collection_timeout_seconds:
                raise CloudWatchDeadlineError("CloudWatch collection deadline exceeded after retry wait.")
    raise CloudWatchRetryError("CloudWatch request failed without a recorded cause.") from last_error
