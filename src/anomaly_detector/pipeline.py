"""End-to-end Phase 1 data pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from anomaly_detector.config import settings
from anomaly_detector.data_ingestion.csv_loader import load_weather_csv
from anomaly_detector.data_ingestion.weather_generator import generate_weather_data
from anomaly_detector.detection.detector import HEAT_STRESS_RULE, detect_anomalies
from anomaly_detector.evaluation import EvaluationResult, evaluate_predictions
from anomaly_detector.preprocessing.processor import preprocess_weather_data
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError
from anomaly_detector.utils.logging_config import configure_logging

LOGGER = logging.getLogger("anomaly_detector.pipeline")


@dataclass(frozen=True)
class PipelineSummary:
    """Metadata returned after a successful pipeline run."""

    input_path: Path
    output_path: Path
    input_rows: int
    output_rows: int
    station_count: int
    anomaly_count: int
    prediction_path: Path | None = None
    evaluation: EvaluationResult | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly summary."""
        values = asdict(self)
        values["input_path"] = str(self.input_path)
        values["output_path"] = str(self.output_path)
        if self.prediction_path is not None:
            values["prediction_path"] = str(self.prediction_path)
        return values


def _prepare_output(data: pd.DataFrame, output_path: Path) -> Path:
    """Serialize one output to a sibling temporary file without publishing it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f".{output_path.name}.",
            dir=output_path.parent, delete=False, encoding="utf-8", newline="",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            serializable = data.copy()
            serializable["timestamp"] = serializable["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            serializable.to_csv(temporary_file, index=False)
        completed = True
        return temporary_path
    except OSError as exc:
        raise DataIngestionError(f"Unable to save processed CSV to {output_path}: {exc}") from exc
    finally:
        if not completed and temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _cleanup(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


def _publish_outputs(staged_outputs: list[tuple[Path, Path]]) -> None:
    """Publish staged files with best-effort rollback across both destinations.

    The filesystem does not provide a true transaction across independent files.
    Existing destinations are backed up before publication and restored if a
    later replacement fails.
    """
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for _, destination in staged_outputs:
            if destination.exists():
                backup_fd, backup_name = tempfile.mkstemp(prefix=f".{destination.name}.backup.", dir=destination.parent)
                os.close(backup_fd)
                backup = Path(backup_name)
                os.unlink(backup)
                os.replace(destination, backup)
                backups[destination] = backup
        for staged, destination in staged_outputs:
            os.replace(staged, destination)
            published.append(destination)
    except OSError as exc:
        for destination in published:
            _cleanup(destination)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
                _cleanup(backup)
        raise DataIngestionError(f"Unable to publish pipeline outputs: {exc}") from exc
    finally:
        for staged, _ in staged_outputs:
            _cleanup(staged)
        for backup in backups.values():
            _cleanup(backup)


def _write_outputs(data: pd.DataFrame, output_path: Path, prediction_data: pd.DataFrame | None, prediction_path: Path | None) -> None:
    staged_outputs = [( _prepare_output(data, output_path), output_path)]
    try:
        if prediction_data is not None and prediction_path is not None:
            staged_outputs.append((_prepare_output(prediction_data, prediction_path), prediction_path))
        _publish_outputs(staged_outputs)
    except Exception:
        for staged, _ in staged_outputs:
            _cleanup(staged)
        raise


def run_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    prediction_output_path: str | Path | None = None,
    *,
    generate_if_missing: bool = False,
    station_count: int = settings.DEFAULT_STATION_COUNT,
    observations_per_station: int = settings.DEFAULT_OBSERVATIONS_PER_STATION,
    seed: int = settings.DEFAULT_RANDOM_SEED,
    anomaly_rate: float = settings.DEFAULT_ANOMALY_RATE,
    enable_cross_feature_rules: bool = False,
) -> PipelineSummary:
    """Load or explicitly generate, preprocess, and atomically save observations."""
    configure_logging(settings.LOG_LEVEL)
    resolved_input = Path(input_path) if input_path is not None else settings.RAW_DATA_DIR / settings.DEFAULT_RAW_FILENAME
    resolved_output = Path(output_path) if output_path is not None else settings.PROCESSED_DATA_DIR / settings.DEFAULT_PROCESSED_FILENAME
    resolved_prediction = Path(prediction_output_path) if prediction_output_path is not None else None
    if resolved_prediction is not None and resolved_prediction == resolved_output:
        raise DataValidationError("Prediction output must be separate from the processed output.")

    if generate_if_missing:
        generated = generate_weather_data(station_count, observations_per_station, seed, anomaly_rate)
        resolved_input.parent.mkdir(parents=True, exist_ok=True)
        generated_to_save = generated.copy()
        generated_to_save["timestamp"] = generated_to_save["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        generated_to_save.to_csv(resolved_input, index=False)
        source = load_weather_csv(resolved_input)
    else:
        source = load_weather_csv(resolved_input)

    processed = preprocess_weather_data(source)
    evaluation: EvaluationResult | None = None
    predictions: pd.DataFrame | None = None
    if resolved_prediction is not None:
        cross_feature_rules = (HEAT_STRESS_RULE,) if enable_cross_feature_rules else ()
        predictions = detect_anomalies(processed, cross_feature_rules=cross_feature_rules)
        evaluation = evaluate_predictions(predictions)
    _write_outputs(processed, resolved_output, predictions, resolved_prediction)
    summary = PipelineSummary(
        input_path=resolved_input,
        output_path=resolved_output,
        input_rows=len(source),
        output_rows=len(processed),
        station_count=int(processed["station_id"].nunique()),
        anomaly_count=int(processed["is_anomaly"].sum()),
        prediction_path=resolved_prediction,
        evaluation=evaluation,
    )
    LOGGER.info("Pipeline completed: %s rows, %s anomalies", summary.output_rows, summary.anomaly_count)
    return summary
