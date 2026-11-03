"""Validated persistent configuration for generic detection and alerts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from anomaly_detector.alert_management import SeverityPolicy
from anomaly_detector.data_ingestion.telemetry_persistence import _connect, _validate_schema, initialize_database
from anomaly_detector.detection.generic_detector import NumericThreshold
from anomaly_detector.utils.exceptions import DataValidationError


@dataclass(frozen=True)
class DetectionConfiguration:
	"""Versioned generic/advanced detection configuration."""

	name: str
	thresholds: Mapping[str, NumericThreshold]
	z_score_sensitivity: float = 3.0
	rolling_window_size: int = 10
	multivariate_sensitivity: float = 3.0
	severity_policy: SeverityPolicy = SeverityPolicy()

	def validate(self) -> None:
		if not isinstance(self.name, str) or not self.name.strip():
			raise DataValidationError("Configuration name must be non-empty.")
		if not isinstance(self.thresholds, Mapping):
			raise DataValidationError("thresholds must be a mapping.")
		for metric, threshold in self.thresholds.items():
			if not isinstance(metric, str) or not metric.strip() or not isinstance(threshold, NumericThreshold):
				raise DataValidationError("Invalid metric threshold configuration.")
			threshold.validate(metric)
		for value, label in ((self.z_score_sensitivity, "z_score_sensitivity"), (self.multivariate_sensitivity, "multivariate_sensitivity")):
			if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
				raise DataValidationError(f"{label} must be positive and finite.")
		if isinstance(self.rolling_window_size, bool) or not isinstance(self.rolling_window_size, int) or self.rolling_window_size < 1:
			raise DataValidationError("rolling_window_size must be a positive integer.")
		self.severity_policy.validate()


def _payload(configuration: DetectionConfiguration) -> str:
	configuration.validate()
	return json.dumps({
		"name": configuration.name,
		"thresholds": {key: {"lower": value.lower, "upper": value.upper} for key, value in sorted(configuration.thresholds.items())},
		"z_score_sensitivity": configuration.z_score_sensitivity,
		"rolling_window_size": configuration.rolling_window_size,
		"multivariate_sensitivity": configuration.multivariate_sensitivity,
		"severity_policy": configuration.severity_policy.__dict__,
	}, sort_keys=True, separators=(",", ":"))


def save_detection_configuration(database_path, configuration: DetectionConfiguration) -> DetectionConfiguration:
	"""Create or replace a named configuration transactionally."""
	payload = _payload(configuration)
	initialize_database(database_path)
	connection = None
	try:
		connection = _connect(database_path); _validate_schema(connection); connection.execute("BEGIN IMMEDIATE")
		connection.execute("INSERT OR REPLACE INTO detection_configurations (name, payload_json, updated_at_utc) VALUES (?, ?, ?)", (configuration.name, payload, datetime.now(timezone.utc).isoformat()))
		connection.commit()
	except sqlite3.Error as exc:
		if connection is not None: connection.rollback()
		raise DataValidationError(f"Unable to save detection configuration: {exc}") from exc
	finally:
		if connection is not None: connection.close()
	return configuration


def load_detection_configuration(database_path, name: str) -> DetectionConfiguration:
	"""Retrieve and validate a named detection configuration."""
	if not isinstance(name, str) or not name.strip():
		raise DataValidationError("Configuration name must be non-empty.")
	initialize_database(database_path)
	connection = None
	try:
		connection = _connect(database_path); _validate_schema(connection)
		row = connection.execute("SELECT payload_json FROM detection_configurations WHERE name = ?", (name,)).fetchone()
	except sqlite3.Error as exc:
		raise DataValidationError(f"Unable to load detection configuration: {exc}") from exc
	finally:
		if connection is not None: connection.close()
	if row is None:
		raise DataValidationError(f"Detection configuration not found: {name}")
	payload = json.loads(row["payload_json"])
	config = DetectionConfiguration(
		name=payload["name"],
		thresholds={key: NumericThreshold(**value) for key, value in payload["thresholds"].items()},
		z_score_sensitivity=payload["z_score_sensitivity"],
		rolling_window_size=payload["rolling_window_size"],
		multivariate_sensitivity=payload["multivariate_sensitivity"],
		severity_policy=SeverityPolicy(**payload["severity_policy"]),
	)
	config.validate()
	return config


def list_detection_configurations(database_path) -> list[dict]:
	"""Return persisted configuration payloads and update timestamps."""
	initialize_database(database_path)
	connection = _connect(database_path)
	try:
		rows = connection.execute(
			"SELECT name, payload_json, updated_at_utc FROM detection_configurations ORDER BY name"
		).fetchall()
	finally:
		connection.close()
	result = []
	for row in rows:
		payload = json.loads(row["payload_json"])
		config = DetectionConfiguration(
			name=payload["name"],
			thresholds={key: NumericThreshold(**value) for key, value in payload["thresholds"].items()},
			z_score_sensitivity=payload["z_score_sensitivity"],
			rolling_window_size=payload["rolling_window_size"],
			multivariate_sensitivity=payload["multivariate_sensitivity"],
			severity_policy=SeverityPolicy(**payload["severity_policy"]),
		)
		config.validate()
		result.append({
			"id": row["name"],
			"updated_at": row["updated_at_utc"],
			"configuration": config,
		})
	return result


def delete_detection_configuration(database_path, name: str) -> None:
	if not isinstance(name, str) or not name.strip():
		raise DataValidationError("Configuration name must be non-empty.")
	initialize_database(database_path)
	connection = _connect(database_path)
	try:
		cursor = connection.execute("DELETE FROM detection_configurations WHERE name = ?", (name,))
		connection.commit()
	finally:
		connection.close()
	if cursor.rowcount == 0:
		raise DataValidationError(f"Detection configuration not found: {name}")
