"""Thin FastAPI API over existing ingestion, detection, persistence, and history APIs."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from anomaly_detector.alert_management import (
    acknowledge_alert,
    generate_alerts,
    persist_alerts,
    retrieve_alerts,
    resolve_alert,
    SeverityPolicy,
)
from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.data_ingestion.telemetry_persistence import (
    persist_advanced_detection_results,
    persist_generic_detection_results,
)
from anomaly_detector.detection.advanced_detector import (
    detect_multivariate_anomalies,
    detect_rolling_z_scores,
    detect_z_scores,
)
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry
from anomaly_detector.detection_configuration import DetectionConfiguration, load_detection_configuration, save_detection_configuration
from anomaly_detector.historical import (
    export_advanced_detection_csv,
    export_generic_detection_csv,
    export_telemetry_csv,
    retrieve_advanced_detection_history,
    retrieve_generic_detection_history,
    retrieve_telemetry_history,
)
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class GenericDetectionRequest(BaseModel):
    thresholds: dict[str, "ThresholdModel"]
    metric_name: str | None = None
    entity_id: str | None = None
    source_type: str | None = None


class AdvancedDetectionRequest(BaseModel):
    method: str = Field(pattern="^(z_score|rolling_z_score|multivariate)$")
    metric_names: list[str] = []
    z_score_sensitivity: float = 3.0
    rolling_window_size: int = 10
    multivariate_sensitivity: float = 3.0
    min_samples: int = 3


class ThresholdModel(BaseModel):
    lower: float
    upper: float


class ConfigurationRequest(BaseModel):
    name: str
    thresholds: dict[str, ThresholdModel] = {}
    z_score_sensitivity: float = 3.0
    rolling_window_size: int = 10
    multivariate_sensitivity: float = 3.0
    medium_score: float = 2.0
    high_score: float = 3.0
    critical_score: float = 5.0


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, (DataValidationError, DataIngestionError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal API error")


def _record_dict(record):
    return {
        "timestamp": record.timestamp.isoformat(), "source_type": record.source_type,
        "metric_name": record.metric_name, "metric_value": record.metric_value,
        "entity_id": record.entity_id, "is_anomaly": record.is_anomaly,
    }


def _configuration_dict(configuration):
    return {
        "name": configuration.name,
        "thresholds": {name: {"lower": value.lower, "upper": value.upper} for name, value in configuration.thresholds.items()},
        "z_score_sensitivity": configuration.z_score_sensitivity,
        "rolling_window_size": configuration.rolling_window_size,
        "multivariate_sensitivity": configuration.multivariate_sensitivity,
        "severity_policy": configuration.severity_policy.__dict__,
    }


def create_app(database_path: str | Path | None = None) -> FastAPI:
    """Create an API application without opening a database or AWS client."""
    resolved_database = Path(database_path or os.getenv("ANOMALY_DATABASE_PATH", "data/telemetry.sqlite"))
    app = FastAPI(title="AWS Anomaly Detector API", version="1.0.0")

    @app.get("/health")
    def health():
        database_status = "unavailable"
        try:
            if resolved_database.exists():
                with sqlite3.connect(resolved_database) as connection:
                    connection.execute("SELECT 1")
                database_status = "ok"
        except sqlite3.Error:
            database_status = "unavailable"
        return {"status": "ok", "database": database_status}

    @app.post("/api/v1/ingestion/csv")
    async def ingest_csv(
        file: UploadFile = File(...), timestamp_column: str = Form("timestamp"),
        metric_columns: str | None = Form(None), entity_column: str | None = Form(None),
        label_column: str | None = Form(None),
    ):
        temporary = None
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise DataValidationError("Uploaded CSV exceeds the 10 MiB limit.")
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                handle.write(content); temporary = Path(handle.name)
            metrics = [item.strip() for item in metric_columns.split(",")] if metric_columns else None
            records = load_infrastructure_csv(temporary, timestamp_column=timestamp_column, metric_columns=metrics, entity_column=entity_column, label_column=label_column)
            return {"records": len(records), "metrics": sorted({record.metric_name for record in records}), "source_type": "csv"}
        except Exception as exc:
            raise _error(exc)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()

    @app.get("/api/v1/telemetry")
    def telemetry(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None):
        try:
            records = retrieve_telemetry_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly)
            return [_record_dict(record) for record in records]
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/detection/generic")
    def generic_detection(request: GenericDetectionRequest):
        try:
            records = retrieve_telemetry_history(resolved_database, metric_name=request.metric_name, entity_id=request.entity_id, source_type=request.source_type)
            results = detect_numeric_telemetry(records, {name: NumericThreshold(value.lower, value.upper) for name, value in request.thresholds.items()})
            saved = persist_generic_detection_results(resolved_database, results)
            alerts = generate_alerts(results); alert_count = persist_alerts(resolved_database, alerts)
            return {"records": len(results), "inserted_results": saved.inserted, "alerts": alert_count}
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/detection/advanced")
    def advanced_detection(request: AdvancedDetectionRequest):
        try:
            records = retrieve_telemetry_history(resolved_database)
            if request.method == "z_score": results = detect_z_scores(records, z_threshold=request.z_score_sensitivity, min_samples=request.min_samples)
            elif request.method == "rolling_z_score": results = detect_rolling_z_scores(records, window_size=request.rolling_window_size, z_threshold=request.z_score_sensitivity, min_samples=request.min_samples)
            else: results = detect_multivariate_anomalies(records, request.metric_names, sensitivity=request.multivariate_sensitivity, min_samples=request.min_samples)
            saved = persist_advanced_detection_results(resolved_database, results)
            alert_count = persist_alerts(resolved_database, generate_alerts(results))
            return {"records": len(results), "inserted_results": saved.inserted, "alerts": alert_count}
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/detection/generic/results")
    def generic_history(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None):
        try: return [result.__dict__ | {"record": _record_dict(result.record)} for result in retrieve_generic_detection_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly)]
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/detection/advanced/results")
    def advanced_history(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None, detection_method=None):
        try: return [result.__dict__ | {"record": _record_dict(result.record)} for result in retrieve_advanced_detection_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly, detection_method=detection_method)]
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/alerts")
    def alerts(severity=None, status=None, metric_name=None, entity_id=None):
        try: return [alert.__dict__ for alert in retrieve_alerts(resolved_database, severity=severity, status=status, metric_name=metric_name, entity_id=entity_id)]
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/alerts/{alert_id}/acknowledge")
    def acknowledge(alert_id: str):
        try: return acknowledge_alert(resolved_database, alert_id).__dict__
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/alerts/{alert_id}/resolve")
    def resolve(alert_id: str):
        try: return resolve_alert(resolved_database, alert_id).__dict__
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/configurations")
    def create_configuration(request: ConfigurationRequest):
        try:
            configuration = DetectionConfiguration(request.name, {name: NumericThreshold(item.lower, item.upper) for name, item in request.thresholds.items()}, request.z_score_sensitivity, request.rolling_window_size, request.multivariate_sensitivity, SeverityPolicy(request.medium_score, request.high_score, request.critical_score))
            save_detection_configuration(resolved_database, configuration)
            return _configuration_dict(configuration) | {"status": "saved"}
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/configurations/{name}")
    def get_configuration(name: str):
        try: return _configuration_dict(load_detection_configuration(resolved_database, name))
        except Exception as exc: raise _error(exc)

    def export_response(exporter, records, filename):
        temporary = tempfile.NamedTemporaryFile(suffix=".csv", delete=False); temporary.close(); path = Path(temporary.name)
        try:
            exporter(records, path); return Response(path.read_bytes(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
        finally:
            if path.exists(): path.unlink()

    @app.get("/api/v1/export/telemetry")
    def export_telemetry(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None):
        try: return export_response(export_telemetry_csv, retrieve_telemetry_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly), "telemetry.csv")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/export/detection/generic")
    def export_generic(metric_name=None, anomaly: bool | None = None):
        try: return export_response(export_generic_detection_csv, retrieve_generic_detection_history(resolved_database, metric_name=metric_name, anomaly=anomaly), "generic_detection.csv")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/export/detection/advanced")
    def export_advanced(metric_name=None, anomaly: bool | None = None, detection_method=None):
        try: return export_response(export_advanced_detection_csv, retrieve_advanced_detection_history(resolved_database, metric_name=metric_name, anomaly=anomaly, detection_method=detection_method), "advanced_detection.csv")
        except Exception as exc: raise _error(exc)

    return app


app = create_app()
