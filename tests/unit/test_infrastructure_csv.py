from pathlib import Path

import pytest

from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv
from anomaly_detector.utils.exceptions import DataIngestionError, DataValidationError


def write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "infrastructure.csv"
    path.write_text(text, encoding="utf-8")
    return path


def valid_csv():
    return (
        "timestamp,entity,cpu,latency,is_anomaly\n"
        "2026-11-02T12:00:00+02:00,host-a,25.5,120.0,false\n"
        "2026-11-02T12:01:00+02:00,host-a,85.0,250.0,true\n"
    )


def test_valid_csv_maps_multiple_metrics_and_normalizes_utc(tmp_path):
    records = load_infrastructure_csv(
        write_csv(tmp_path, valid_csv()),
        metric_columns=["cpu", "latency"],
        entity_column="entity",
        label_column="is_anomaly",
    )
    assert len(records) == 4
    assert records[0].source_type == "csv"
    assert records[0].metric_name == "cpu"
    assert records[0].metric_value == 25.5
    assert records[0].timestamp.isoformat() == "2026-11-02T10:00:00+00:00"
    assert records[0].entity_id == "host-a"
    assert records[1].metric_name == "latency"
    assert records[2].is_anomaly is True


def test_metric_columns_can_be_inferred_and_entity_can_be_absent(tmp_path):
    path = write_csv(tmp_path, "timestamp,cpu,requests\n2026-11-02T12:00:00Z,10,5\n")
    records = load_infrastructure_csv(path)
    assert [record.metric_name for record in records] == ["cpu", "requests"]
    assert all(record.entity_id is None for record in records)


def test_invalid_timestamp_missing_metric_and_non_numeric_are_rejected(tmp_path):
    for content in (
        "timestamp,cpu\nnot-time,10\n",
        "timestamp,cpu\n2026-11-02T12:00:00Z,\n",
        "timestamp,cpu\n2026-11-02T12:00:00Z,not-number\n",
        "timestamp,cpu\n2026-11-02T12:00:00Z,inf\n",
        "timestamp,cpu\n2026-11-02T12:00:00Z,nan\n",
    ):
        with pytest.raises(DataValidationError):
            load_infrastructure_csv(write_csv(tmp_path, content))


def test_missing_file_empty_file_and_missing_columns_fail(tmp_path):
    with pytest.raises(DataIngestionError):
        load_infrastructure_csv(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_infrastructure_csv(empty)
    with pytest.raises(DataValidationError):
        load_infrastructure_csv(write_csv(tmp_path, "time,cpu\n2026-11-02T12:00:00Z,1\n"))


def test_malformed_csv_duplicate_and_invalid_optional_columns_fail(tmp_path):
    malformed = write_csv(tmp_path, 'timestamp,cpu\n"2026-11-02T12:00:00Z,1\n')
    with pytest.raises(DataValidationError):
        load_infrastructure_csv(malformed)
    duplicate = write_csv(tmp_path, "timestamp,entity,cpu\n2026-11-02T12:00:00Z,a,1\n2026-11-02T12:00:00Z,a,2\n")
    with pytest.raises(DataValidationError):
        load_infrastructure_csv(duplicate, entity_column="entity")
    with pytest.raises(DataValidationError):
        load_infrastructure_csv(write_csv(tmp_path, "timestamp,cpu\n2026-11-02T12:00:00Z,1\n"), entity_column="host")


def test_labels_are_optional_but_strict(tmp_path):
    path = write_csv(tmp_path, "timestamp,cpu,label\n2026-11-02T12:00:00Z,10,false\n")
    records = load_infrastructure_csv(path, metric_columns=["cpu"], label_column="label")
    assert records[0].is_anomaly is False
    invalid = write_csv(tmp_path, "timestamp,cpu,label\n2026-11-02T12:00:00Z,10,maybe\n")
    with pytest.raises(DataValidationError):
        load_infrastructure_csv(invalid, metric_columns=["cpu"], label_column="label")
