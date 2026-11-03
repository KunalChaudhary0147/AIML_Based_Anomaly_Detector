"""Thin FastAPI API over existing ingestion, detection, persistence, and history APIs."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from anomaly_detector.data_ingestion.datasets import (
    delete_datasets, get_dataset, ingest_dataset, list_datasets, preview_dataset, quality_dataset,
)

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
    _connect,
    initialize_database,
    persist_advanced_detection_results,
    persist_generic_detection_results,
)
from anomaly_detector.detection.advanced_detector import (
    detect_multivariate_anomalies,
    detect_rolling_z_scores,
    detect_z_scores,
)
from anomaly_detector.detection.generic_detector import NumericThreshold, detect_numeric_telemetry
from anomaly_detector.detection_configuration import (
    DetectionConfiguration, delete_detection_configuration, list_detection_configurations,
    load_detection_configuration, save_detection_configuration,
)
from anomaly_detector.historical import (
    export_advanced_detection_csv,
    export_generic_detection_csv,
    export_telemetry_csv,
    retrieve_advanced_detection_history,
    retrieve_generic_detection_history,
    retrieve_telemetry_history,
)
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError
from anomaly_detector.database import database_target

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class GenericDetectionRequest(BaseModel):
    thresholds: dict[str, "ThresholdModel"]
    metric_name: str | None = None
    entity_id: str | None = None
    source_type: str | None = None
    dataset_ids: list[str] | None = None


class AdvancedDetectionRequest(BaseModel):
    method: str = Field(pattern="^(z_score|rolling_z_score|multivariate)$")
    metric_names: list[str] = []
    z_score_sensitivity: float = 3.0
    rolling_window_size: int = 10
    multivariate_sensitivity: float = 3.0
    min_samples: int = 3
    dataset_ids: list[str] | None = None


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


class BulkDatasetDeleteRequest(BaseModel):
    dataset_ids: list[str]


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, (DataValidationError, DataIngestionError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal API error")


def _record_dict(record):
    return {
        "timestamp": record.timestamp.isoformat(), "source_type": record.source_type,
        "metric_name": record.metric_name, "metric_value": record.metric_value,
        "entity_id": record.entity_id, "is_anomaly": record.is_anomaly, "dataset_id": record.dataset_id,
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
    """Create an API application with SQLite initialized during startup."""
    resolved_database = database_target(database_path)
    cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "ANOMALY_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    cors_credentials = os.getenv("ANOMALY_CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        initialize_database(resolved_database)
        yield

    app = FastAPI(title="AWS Anomaly Detector API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS", "PUT", "PATCH", "HEAD"],
        allow_headers=["*"],
    )

    def dataset_filter(dataset_ids):
        if dataset_ids is None:
            return None
        if dataset_ids == ["__none__"]:
            return []
        if not isinstance(dataset_ids, list) or any(not isinstance(item, str) or not item.strip() for item in dataset_ids):
            raise DataValidationError("dataset_ids must contain non-empty strings.")
        known = {item["dataset_id"] for item in list_datasets(resolved_database)}
        invalid = sorted(set(dataset_ids) - known)
        if invalid:
            raise DataValidationError(f"Unknown dataset IDs: {invalid}")
        return list(dict.fromkeys(dataset_ids))

    @app.get("/health")
    def health():
        database_status = "unavailable"
        try:
            with _connect(resolved_database) as connection:
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
            dataset = ingest_dataset(resolved_database, file.filename or "upload.csv", content, records)
            return {"dataset_id": dataset["dataset_id"], "records": len(records), "metrics": sorted({record.metric_name for record in records}), "source_type": "csv"}
        except Exception as exc:
            raise _error(exc)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()

    @app.get("/api/v1/telemetry")
    def telemetry(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None, dataset_ids: list[str] | None = Query(None)):
        try:
            records = retrieve_telemetry_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly, dataset_ids=dataset_filter(dataset_ids))
            return [_record_dict(record) for record in records]
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/detection/generic")
    def generic_detection(request: GenericDetectionRequest):
        try:
            records = retrieve_telemetry_history(resolved_database, metric_name=request.metric_name, entity_id=request.entity_id, source_type=request.source_type, dataset_ids=dataset_filter(request.dataset_ids))
            results = detect_numeric_telemetry(records, {name: NumericThreshold(value.lower, value.upper) for name, value in request.thresholds.items()})
            saved = persist_generic_detection_results(resolved_database, results)
            alerts = generate_alerts(results); alert_count = persist_alerts(resolved_database, alerts)
            return {"records": len(results), "inserted_results": saved.inserted, "alerts": alert_count}
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/detection/advanced")
    def advanced_detection(request: AdvancedDetectionRequest):
        try:
            records = retrieve_telemetry_history(resolved_database, dataset_ids=dataset_filter(request.dataset_ids))
            if request.method == "z_score": results = detect_z_scores(records, z_threshold=request.z_score_sensitivity, min_samples=request.min_samples)
            elif request.method == "rolling_z_score": results = detect_rolling_z_scores(records, window_size=request.rolling_window_size, z_threshold=request.z_score_sensitivity, min_samples=request.min_samples)
            else: results = detect_multivariate_anomalies(records, request.metric_names, sensitivity=request.multivariate_sensitivity, min_samples=request.min_samples)
            saved = persist_advanced_detection_results(resolved_database, results)
            alert_count = persist_alerts(resolved_database, generate_alerts(results))
            return {"records": len(results), "inserted_results": saved.inserted, "alerts": alert_count}
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/detection/generic/results")
    def generic_history(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None, dataset_ids: list[str] | None = Query(None)):
        try: return [result.__dict__ | {"record": _record_dict(result.record)} for result in retrieve_generic_detection_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly, dataset_ids=dataset_filter(dataset_ids))]
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/detection/advanced/results")
    def advanced_history(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None, detection_method=None, dataset_ids: list[str] | None = Query(None)):
        try: return [result.__dict__ | {"record": _record_dict(result.record)} for result in retrieve_advanced_detection_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly, detection_method=detection_method, dataset_ids=dataset_filter(dataset_ids))]
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/alerts")
    def alerts(severity=None, status=None, metric_name=None, entity_id=None, start: datetime | None = None, end: datetime | None = None, dataset_ids: list[str] | None = Query(None)):
        try:
            results = retrieve_alerts(resolved_database, severity=severity, status=status, metric_name=metric_name, entity_id=entity_id, dataset_ids=dataset_filter(dataset_ids))
            if start is not None:
                results = [alert for alert in results if alert.created_at_utc >= start.isoformat()]
            if end is not None:
                results = [alert for alert in results if alert.created_at_utc <= end.isoformat()]
            return [alert.__dict__ for alert in results]
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

    @app.get("/api/v1/configurations")
    def configurations():
        try:
            return [_configuration_dict(item["configuration"]) | {"id": item["id"], "updated_at": item["updated_at"]} for item in list_detection_configurations(resolved_database)]
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/configurations/{name}")
    def get_configuration(name: str):
        try: return _configuration_dict(load_detection_configuration(resolved_database, name))
        except Exception as exc: raise _error(exc)

    @app.delete("/api/v1/configurations/{name}")
    def delete_configuration(name: str):
        try:
            delete_detection_configuration(resolved_database, name)
            return {"name": name, "deleted": True}
        except Exception as exc: raise _error(exc)

    def export_response(exporter, records, filename):
        temporary = tempfile.NamedTemporaryFile(suffix=".csv", delete=False); temporary.close(); path = Path(temporary.name)
        try:
            exporter(records, path); return Response(path.read_bytes(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
        finally:
            if path.exists(): path.unlink()

    @app.get("/api/v1/export/telemetry")
    def export_telemetry(start: datetime | None = None, end: datetime | None = None, metric_name=None, entity_id=None, source_type=None, anomaly: bool | None = None, dataset_ids: list[str] | None = Query(None)):
        try: return export_response(export_telemetry_csv, retrieve_telemetry_history(resolved_database, start=start, end=end, metric_name=metric_name, entity_id=entity_id, source_type=source_type, anomaly=anomaly, dataset_ids=dataset_filter(dataset_ids)), "telemetry.csv")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/export/detection/generic")
    def export_generic(metric_name=None, anomaly: bool | None = None, dataset_ids: list[str] | None = Query(None)):
        try: return export_response(export_generic_detection_csv, retrieve_generic_detection_history(resolved_database, metric_name=metric_name, anomaly=anomaly, dataset_ids=dataset_filter(dataset_ids)), "generic_detection.csv")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/export/detection/advanced")
    def export_advanced(metric_name=None, anomaly: bool | None = None, detection_method=None, dataset_ids: list[str] | None = Query(None)):
        try: return export_response(export_advanced_detection_csv, retrieve_advanced_detection_history(resolved_database, metric_name=metric_name, anomaly=anomaly, detection_method=detection_method, dataset_ids=dataset_filter(dataset_ids)), "advanced_detection.csv")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/datasets")
    def datasets(search: str | None = None, source_type: str | None = None):
        try:
            return list_datasets(resolved_database, search=search, source_type=source_type)
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/datasets/stats")
    def dataset_stats():
        try:
            rows = list_datasets(resolved_database)
            with _connect(resolved_database) as connection:
                total_entities, total_metrics = connection.execute(
                    "SELECT COUNT(DISTINCT entity_id), COUNT(DISTINCT metric_name) FROM telemetry_records"
                ).fetchone()
            return {
                "total_datasets": len(rows),
                "total_records": sum(row["record_count"] for row in rows),
                "total_entities": total_entities or 0,
                "total_metrics": total_metrics or 0,
            }
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/datasets/{dataset_id}")
    def dataset_details(dataset_id: str):
        try: return get_dataset(resolved_database, dataset_id)
        except KeyError: raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/datasets/{dataset_id}/preview")
    def dataset_preview(dataset_id: str):
        try: return [_record_dict(record) for record in preview_dataset(resolved_database, dataset_id)]
        except KeyError: raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/datasets/{dataset_id}/quality")
    def dataset_quality(dataset_id: str):
        try: return quality_dataset(resolved_database, dataset_id)
        except KeyError: raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as exc: raise _error(exc)

    @app.delete("/api/v1/datasets/{dataset_id}")
    def dataset_delete(dataset_id: str):
        try: return delete_datasets(resolved_database, [dataset_id])
        except KeyError: raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as exc: raise _error(exc)

    @app.post("/api/v1/datasets/delete")
    def datasets_delete(request: BulkDatasetDeleteRequest):
        try:
            if not request.dataset_ids:
                raise DataValidationError("dataset_ids must not be empty.")
            return delete_datasets(resolved_database, request.dataset_ids)
        except KeyError: raise HTTPException(status_code=404, detail="One or more datasets not found")
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/overview")
    def overview(dataset_ids: list[str] | None = Query(None)):
        try:
            selected = dataset_filter(dataset_ids)
            records = retrieve_telemetry_history(resolved_database, dataset_ids=selected)
            alerts = retrieve_alerts(resolved_database, dataset_ids=selected)
            generic = retrieve_generic_detection_history(resolved_database, dataset_ids=selected)
            advanced = retrieve_advanced_detection_history(resolved_database, dataset_ids=selected)
            dates = sorted({record.timestamp.date().isoformat() for record in records})
            activity = {date: 0 for date in dates}
            split = {date: {"date": date, "normal": 0, "anomalous": 0} for date in dates}
            for record in records:
                date = record.timestamp.date().isoformat()
                activity[date] += 1
                if record.is_anomaly is True:
                    split[date]["anomalous"] += 1
                elif record.is_anomaly is False:
                    split[date]["normal"] += 1
            detection_results = generic + advanced
            detection_dates = {}
            for result in detection_results:
                date = result.record.timestamp.date().isoformat()
                detection_dates[date] = detection_dates.get(date, 0) + 1
            severity_counts = {severity: 0 for severity in ("low", "medium", "high", "critical")}
            for alert in alerts:
                severity_counts[alert.severity.lower()] += 1
            active_alerts = [alert for alert in alerts if alert.status != "RESOLVED"]
            recent_detections = sorted(detection_results, key=lambda item: item.record.timestamp, reverse=True)[:10]
            return {
                "totals": {
                    "telemetry_records": len(records),
                    "anomalies": sum(record.is_anomaly is True for record in records),
                    "normal_observations": sum(record.is_anomaly is False for record in records),
                    "anomalous_observations": sum(record.is_anomaly is True for record in records),
                    "total_metrics": len({record.metric_name for record in records}),
                    "active_alerts": len(active_alerts),
                    "critical_alerts": sum(alert.severity == "CRITICAL" and alert.status != "RESOLVED" for alert in alerts),
                    "monitored_entities": len({record.entity_id for record in records if record.entity_id is not None}),
                },
                "metric_activity": [{"date": date, "observations": activity[date]} for date in dates],
                "observation_split": list(split.values()),
                "severity_distribution": [{"severity": severity, "count": severity_counts[severity]} for severity in severity_counts],
                "detection_activity": [{"date": date, "detections": detection_dates[date]} for date in sorted(detection_dates)],
                "recent_alerts": [alert.__dict__ for alert in sorted(alerts, key=lambda item: item.created_at_utc, reverse=True)[:10]],
                "recent_detections": [{
                    "timestamp": item.record.timestamp.isoformat(),
                    "entity": item.record.entity_id or "",
                    "metric": item.record.metric_name,
                    "observed_value": item.record.metric_value,
                    "reference_value": getattr(item, "reference_value", None),
                    "score": getattr(item, "statistical_score", 0) or 0,
                    "method": getattr(item, "detection_method", "generic_threshold"),
                    "prediction": item.predicted_is_anomaly,
                    "severity": "low" if item.predicted_is_anomaly else "normal",
                    "explanation": item.explanation,
                } for item in recent_detections],
            }
        except Exception as exc: raise _error(exc)

    @app.get("/api/v1/catalog")
    def catalog(dataset_ids: list[str] | None = Query(None)):
        try:
            records = retrieve_telemetry_history(resolved_database, dataset_ids=dataset_filter(dataset_ids))
            return {
                "metrics": sorted({record.metric_name for record in records}),
                "entities": sorted({record.entity_id for record in records if record.entity_id is not None}),
                "source_types": sorted({record.source_type for record in records}),
            }
        except Exception as exc: raise _error(exc)

    return app


app = create_app()
