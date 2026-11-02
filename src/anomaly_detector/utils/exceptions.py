"""Application-specific exception types."""


class AnomalyDetectorError(Exception):
    """Base exception for expected application errors."""


class ConfigurationError(AnomalyDetectorError):
    """Raised when configuration values are invalid."""


class DataIngestionError(AnomalyDetectorError):
    """Raised when a source dataset cannot be read."""


class CloudWatchClientError(DataIngestionError):
    """Raised for non-retryable or exhausted CloudWatch client failures."""


class CloudWatchRetryError(CloudWatchClientError):
    """Raised when retryable CloudWatch failures exhaust the attempt limit."""


class CloudWatchPaginationError(CloudWatchClientError):
    """Raised when paginated collection exceeds or violates its bounds."""


class CloudWatchRecordLimitError(CloudWatchClientError):
    """Raised when a collection exceeds its maximum record budget."""


class CloudWatchDeadlineError(CloudWatchClientError):
    """Raised when a collection exceeds its overall application deadline."""


class DataValidationError(AnomalyDetectorError):
    """Raised when a dataset violates the structural contract."""


class PreprocessingError(AnomalyDetectorError):
    """Raised when preprocessing cannot safely prepare a dataset."""
