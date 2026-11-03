# AWS Anomaly Detector

Phase 1 is a tested foundation for synthetic weather data ingestion and anomaly-label preservation. Phase 2 adds a configurable rule-based detector and evaluation output. This project does not implement machine learning, AWS services, or deployment.

## Architecture

- `src/anomaly_detector/config/settings.py`: environment-aware paths, defaults, schema, and validation.
- `data_ingestion/weather_generator.py`: deterministic multi-station generation using a local NumPy generator.
- `data_ingestion/csv_loader.py`: strict CSV reading and structural validation.
- `preprocessing/processor.py`: copy-based UTC/type normalization and duplicate/missing-value rejection.
- `pipeline.py`: load, preprocess, and atomic processed-output writing.
- `detection/detector.py`: explicit inclusive threshold rules and prediction reasons.
- `detection/generic_detector.py`: configurable per-metric threshold detection for `TelemetryRecord` objects.
- `evaluation.py`: confusion-matrix counts and precision, recall, and F1 metrics.
- `data_ingestion/telemetry.py`: validated, normalized metric observations for future non-weather sources; separate from the weather CSV contract.
- `data_ingestion/cloudwatch_adapter.py`: offline conversion of mocked GetMetricData-style responses into `TelemetryRecord` instances.
- Phase 2.2 adds an opt-in heat-stress cross-feature rule; station-specific baselines remain deferred because the current history is too small and anomaly-contaminated.
- `scripts/`: thin command-line entry points.
- `tests/`: unit, integration, and small CSV fixtures.

## Environment

Python was detected and tested with Python 3.14.0 at `C:\Python314\python.exe`. The project uses that existing installation to create `.venv`; it does not install or change Python itself.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

The package uses a `src` layout and editable installation. The documented commands below are run from the repository root.

## Schema

Every dataset uses this exact column order:

| Column | Meaning | Unit/type |
|---|---|---|
| `timestamp` | UTC observation time | timezone-aware datetime |
| `station_id` | station identifier | non-empty string |
| `temperature` | air temperature | Celsius, numeric |
| `humidity` | relative humidity | percent, numeric |
| `wind_speed` | wind speed | metres/second, numeric |
| `atmospheric_pressure` | pressure | hPa, numeric |
| `is_anomaly` | generator ground truth | Boolean |

The generator applies anomaly rate to the total dataset and rounds `rate * row_count` to the nearest integer, with halves rounded up. It deliberately injects extreme values and labels those rows `True`; processing never removes them based on physical limits. Structural checks reject missing/non-finite values, invalid timestamps, empty IDs, invalid labels, extra columns, and duplicate `timestamp` plus `station_id` pairs. Extra columns are rejected rather than silently dropped.

Naive timestamps are interpreted as UTC. CSV output uses `YYYY-MM-DDTHH:MM:SSZ`. Missing values are rejected; no rows are silently deleted or imputed. Raw input is never overwritten, and processed output is written through a temporary file before replacement.

When both processed and prediction outputs are requested, the pipeline prepares both temporary files before publishing either one. Existing outputs are backed up during publication and restored on a later replacement failure where possible. This is best-effort recovery, not a true filesystem transaction across two independent files.

## Commands

Generate a raw dataset:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --stations 3 --observations 24 --seed 42 --anomaly-rate 0.05
```

Process the default raw CSV into `data/processed/`:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

The pipeline keeps `data/processed/weather_processed.csv` unchanged in its Phase 1 schema and also writes rule-based predictions to `data/processed/weather_predictions.csv`. The command prints evaluation metrics when `is_anomaly` labels are present. Use `--predictions path\to\predictions.csv` to select another prediction output path.

Use custom paths:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --input path\to\input.csv --output path\to\output.csv
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit -q
.\.venv\Scripts\python.exe -m pytest tests\integration -q
.\.venv\Scripts\python.exe -m pytest -q
```

## Phase 2 rule detection

The detector applies explicit inclusive lower and upper bounds independently to `temperature`, `humidity`, `wind_speed`, and `atmospheric_pressure`. A value strictly below its lower bound or strictly above its upper bound is predicted anomalous. Values exactly on a threshold are normal. Reasons are recorded in `anomaly_reasons`; multiple violated rules are separated by semicolons.

Default thresholds are:

| Field | Lower | Upper |
|---|---:|---:|
| `temperature` | -40 C | 50 C |
| `humidity` | 0% | 100% |
| `wind_speed` | 0 m/s | 35 m/s |
| `atmospheric_pressure` | 900 hPa | 1100 hPa |

Override them with the corresponding non-secret environment variables in `.env`, such as `ANOMALY_TEMPERATURE_MIN` and `ANOMALY_TEMPERATURE_MAX`. The detector validates numeric, finite, non-contradictory bounds. It does not read `is_anomaly`; that column is ground truth supplied by the dataset. `predicted_is_anomaly` is the detector output and is kept separate from ground truth.

When labels are available, evaluation reports true positives, false positives, true negatives, false negatives, precision, recall, and F1. Zero denominators produce `0.0`; without ground truth, metrics are unavailable rather than invented. Rules are fixed operational checks, not learned patterns, and cannot identify every possible anomaly.

Evaluation accepts actual Boolean values only. Strings such as `"False"`, integers such as `0` or `1`, and missing values are rejected rather than coerced.

## Phase 4.2 generic telemetry detection

`detect_numeric_telemetry()` applies inclusive lower and upper `NumericThreshold` bounds by `TelemetryRecord.metric_name`. It returns the original record, `predicted_is_anomaly`, and an explanation. Values outside configured bounds are anomalous; metrics without thresholds are explicitly reported as `unconfigured_metric` rather than treated as normal. The detector never uses `is_anomaly`, does not mutate records, and remains separate from the weather detector. This is a deterministic rule-based component, not a generic AWS workflow or machine-learning model.

Generic results can be persisted separately through `persist_generic_detection_results()` and retrieved with `retrieve_generic_detection_results()`. Results use the existing telemetry datapoint identity as a unique key, preserve the original record association, anomaly prediction, and explanation, and support optional anomaly-status and metric-name filters. Persistence is transactional and remains separate from CloudWatch ingestion and the weather pipeline.

## Phase 6 advanced rule-based detection

`detection/advanced_detector.py` adds non-ML statistical detectors for `TelemetryRecord` objects: global per-entity/metric z-scores, leakage-resistant rolling z-scores using prior observations, and standardized multivariate scores for complete entity/timestamp metric vectors. Results include the original record, method, observed/reference values, score, prediction, and explanation. Insufficient samples, zero variance, and missing metric combinations are reported explicitly rather than forced into anomalous or normal classifications. `evaluation_engine.py` calculates TP, FP, TN, FN, precision, recall, F1, and accuracy only when Boolean ground-truth labels are available.

Advanced results can be persisted independently with `persist_advanced_detection_results()` and retrieved with `retrieve_advanced_detection_results()`. Their method-specific result identity prevents different detection methods from overwriting one another. These detectors remain offline and are not connected to AWS ingestion or the weather pipeline.

## Phase 7 historical retrieval and export

Historical APIs in `historical.py` query the existing SQLite database without loading unrelated rows: `retrieve_telemetry_history()`, `retrieve_generic_detection_history()`, and `retrieve_advanced_detection_history()` support UTC start/end timestamps, metric, entity, source, anomaly, and detection-method filters where applicable. CSV exports are available through `export_telemetry_csv()`, `export_generic_detection_csv()`, and `export_advanced_detection_csv()`. Exports use UTC ISO-8601 timestamps, clear headers, parameterized filters, and atomic temporary-file replacement; they do not modify source datasets.

## Phase 8 alert and configuration management

`generate_alerts()` converts anomalous generic or advanced detection results into persisted alerts. Normal results do not create alerts. The default severity policy assigns `LOW` to generic anomalies or advanced scores below `2`, `MEDIUM` at `2`, `HIGH` at `3`, and `CRITICAL` at `5`; `SeverityPolicy` validates and customizes these boundaries. Alerts use `NEW -> ACKNOWLEDGED -> RESOLVED` lifecycle transitions, preserve history, prevent duplicate detection identities, and support severity/status/metric/entity filters.

`DetectionConfiguration` stores validated metric thresholds, z-score sensitivity, rolling-window size, multivariate sensitivity, and severity policy in the existing SQLite database through `save_detection_configuration()` and `load_detection_configuration()`. Alerting and configuration are local persistence features only: they do not trigger AWS calls, notifications, or weather-pipeline behavior.

## Phase 9 FastAPI REST API

Start the API from the project root with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn anomaly_detector.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Set `ANOMALY_DATABASE_PATH` to select the SQLite database; otherwise the API uses `data/telemetry.sqlite`. No AWS credentials or AWS calls are required. Interactive documentation is available at `/docs`.

Available endpoint groups include `/health`, `/api/v1/ingestion/csv`, `/api/v1/telemetry`, `/api/v1/overview`, `/api/v1/catalog`, `/api/v1/detection/generic`, `/api/v1/detection/advanced`, `/api/v1/detection/generic/results`, `/api/v1/detection/advanced/results`, `/api/v1/alerts`, `/api/v1/configurations`, `/api/v1/datasets`, and CSV export endpoints under `/api/v1/export/`. Dataset-aware endpoints accept repeated `dataset_ids` query parameters; an explicit empty selection is represented internally by `dataset_ids=__none__`.

## Phase 5 infrastructure CSV ingestion

Infrastructure CSV data can be converted into generic telemetry records without AWS access:

```python
from anomaly_detector.data_ingestion.infrastructure_csv import load_infrastructure_csv

records = load_infrastructure_csv(
	"infrastructure.csv",
	timestamp_column="timestamp",
	metric_columns=["cpu", "latency"],
	entity_column="host",
	label_column="is_anomaly",
)
```

The timestamp column is required and is normalized to UTC; naive timestamps are interpreted as UTC. Selected metric columns must contain finite numeric values. Missing values, invalid timestamps, malformed CSV, empty files, and duplicate timestamp/entity/metric observations are rejected. If `metric_columns` is omitted, all non-timestamp/entity/label columns are treated as metrics. Entity and ground-truth label columns are optional; labels are explicitly parsed as Boolean values and remain separate from detector predictions. Records use `source_type="csv"`, no AWS metadata, and can flow through the generic detector and SQLite detection-result persistence.

## Phase 3.2.2 telemetry contract

`TelemetryRecord` represents one numeric metric observation without connecting to AWS or changing the weather pipeline. Required fields are a timezone-aware `timestamp`, non-empty `source_type`, non-empty `metric_name`, and a finite non-Boolean `metric_value`. Timestamps are normalized to UTC; naive timestamps are rejected. Optional fields include `metric_unit`, `entity_id`, optional Boolean `is_anomaly`, AWS-style `namespace`, `resource_arn`, `account_id`, `region`, positive-integer `period` seconds, and non-empty `statistic`.

`dimensions` is a copied immutable mapping of non-empty string keys to non-empty string values. The `canonical_identity` property preserves account, region, namespace, metric name, sorted dimensions, resource/entity identity, UTC timestamp, period, and statistic for future identity or deduplication work. This phase does not store, deduplicate, ingest, or pass telemetry records to the weather detector. Generic AWS metrics require a future source-specific adapter and detector/schema decision.

Telemetry identity is also separated into `time_series_identity` and `datapoint_identity`. The time-series identity contains source type, account, region, namespace, metric name, sorted dimensions, metric unit, and one canonical resource identity: `resource_arn` when present, otherwise `entity_id`, otherwise `None`. The datapoint identity appends the UTC timestamp, period, and statistic. Neither identity includes `metric_value` or `is_anomaly`. The original `canonical_identity` tuple remains unchanged for backward compatibility.

The offline CloudWatch adapter accepts a response containing `MetricDataResults` and a separate caller-provided metadata mapping keyed by the exact result `Id`. It never infers metric metadata from IDs. Each result must have matching `Timestamps` and `Values`; timestamps must be timezone-aware `datetime` values and values must be finite, non-Boolean numbers. `StatusCode="Complete"` is accepted, including explicit empty arrays; missing or empty overall results return an empty list. Failed or partial statuses such as `PartialData`, `InternalError`, or `Forbidden`, malformed messages, missing metadata mappings, and malformed result structures raise `DataValidationError`. CloudWatch records use `source_type="aws_cloudwatch"` and `is_anomaly=None`. No AWS SDK, network call, retry, storage, or deduplication is part of this phase.

The offline client boundary in `data_ingestion/cloudwatch_client.py` accepts an injected object with `get_metric_data(**request_parameters)`, deep-copies request parameters, and passes the response to the existing adapter. It retries only known transport, timeout, throttling, and temporary service failures. Authentication, authorization, validation, malformed-response, and other permanent errors are not retried. Defaults are three attempts, a `0.5` second base delay, and up to `0.1` seconds of injected jitter. Sleep and randomness are injectable for deterministic tests; no AWS SDK or credentials are created.

The optional real-client factory is `data_ingestion/aws_client.py:create_cloudwatch_client(region_name=None)`. It lazily imports boto3 and creates a CloudWatch client only when called. Pass an explicit region or omit it to use boto3's standard AWS region and credential provider chain. Configure credentials through standard AWS profiles/config files, environment variables, or workload roles; credentials are never accepted or stored by this project. The current environment may not have boto3 installed; the dependency is declared but is not installed automatically during offline tests.

## CloudWatch contract references

The offline adapter/client contract was checked against the official AWS references: [GetMetricData API](https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_GetMetricData.html), [MetricDataResult](https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_MetricDataResult.html), [MessageData](https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_MessageData.html), and [CloudWatch Common Errors](https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/CommonErrors.html). AWS documents numeric Unix timestamps, aligned `Timestamps`/`Values`, result statuses `Complete`, `PartialData`, `InternalError`, and `Forbidden`, and `NextToken` pagination. The adapter normalizes numeric Unix timestamps to UTC; the client continues `PartialData` and token-bearing `InternalError` pages, while rejecting incomplete `PartialData` without a token and failed statuses without a usable continuation.

The client collects CloudWatch-style `NextToken` pages with a default `max_pages=100` safeguard. A missing, `None`, or empty token ends collection; repeated tokens and a nonempty token after the page limit raise `CloudWatchPaginationError`. A caller-supplied nonempty `NextToken` is treated as an explicit resume point and preserved on the first request; callers starting from the beginning should omit it. Each page is passed through the existing adapter, and records are returned only after every page succeeds, so later failures or malformed pages cannot produce partial results. Pagination does not deduplicate records.

Collection also has a default `max_records=100000` budget, applied across all pages after each page is converted. A page that would exceed the remaining budget raises `CloudWatchRecordLimitError`; records are never silently truncated or returned partially. An optional `collection_timeout_seconds` applies one monotonic deadline across every page and retry wait. It raises `CloudWatchDeadlineError` without returning partial results. The timeout is an application-level collection deadline and cannot interrupt a client call that blocks indefinitely; a future real AWS client must also configure its own transport timeout.

`data_ingestion/telemetry_deduplication.py` provides an in-memory `deduplicate_telemetry_records()` utility. It uses each record's existing `datapoint_identity`, preserves first-seen order, returns a new list, and keeps the first record when duplicates occur. A later duplicate is discarded even if its metric value or optional ground-truth label differs. The utility does not persist data, mutate records, or perform AWS/network operations.

## Phase 3.3.4 SQLite telemetry persistence

`data_ingestion/telemetry_persistence.py` provides standalone local SQLite persistence:

```python
initialize_database(path)
result = persist_telemetry_records(path, records)
records = retrieve_telemetry_records(path)
conflicts = retrieve_conflict_events(path)
```

The schema stores schema version `1`, deterministic JSON serializations of the existing `datapoint_identity` and `time_series_identity`, UTC timestamps, all telemetry values and metadata, dimensions, and optional ground-truth labels. A unique database constraint enforces datapoint identity. Repeating an identical record increments `already_existing`; a same-identity record with a different metric value or label increments `conflicts`, preserves the original row, and stores a conflict event. This is a first-record-preserving, continue-after-conflict policy.

Each persistence batch uses an explicit SQLite transaction. Database failures, conflict-event failures, and commit failures roll back the complete batch; existing records are preserved. `retrieve_conflict_events()` returns conflict events in deterministic `conflict_id` order without changing observations. Database initialization validates required tables, columns, schema version `1`, and datapoint-identity uniqueness. Existing incompatible databases are rejected and never silently modified; migrations are not supported yet.

SQLite uses a conservative serialized-writer policy with a five-second busy timeout. Multiple-process write safety is not claimed. Record retrieval currently loads all persisted records into memory; bounded retrieval is deferred. The module does not implement checkpoints, restart recovery, migrations, retention, persistence integration with CloudWatch, or the weather pipeline. SQLite is an initial local storage choice; future deployment may require a managed durable store and explicit migration strategy.

## Phase 3.3.8 checkpoint storage

`data_ingestion/checkpoint_store.py` provides standalone checkpoint storage APIs:

```python
fingerprint = query_identity_fingerprint(query_configuration, window_start, window_end)
checkpoint = create_checkpoint(database_path, query_configuration=..., source_type=..., window_start=..., window_end=...)
checkpoint = update_checkpoint(database_path, checkpoint.run_id, expected_query_fingerprint=fingerprint, new_status="running")
incomplete = list_incomplete_checkpoints(database_path)
```

Checkpoint rows are stored in the same SQLite database as telemetry and use schema version `1`. Query fingerprints are SHA-256 hashes of canonicalized request configuration and UTC bounded-window timestamps. Checkpoints track run status, pagination token, committed page/record counters, errors, and timestamps. Valid states are `pending`, `running`, `page_committed`, `completed`, `failed`, and `needs_recovery`; invalid transitions and query-fingerprint mismatches are rejected. This phase stores checkpoint state only: it does not connect checkpoints to telemetry writes, CloudWatch collection, token restart, or restart orchestration. Migrations are unsupported, and the existing five-second serialized-writer policy applies.

`tests/integration/test_telemetry_ingestion.py` composes the injected CloudWatch client, response adapter, telemetry model, and deduplication utility with deterministic fake responses. It covers paginated success, retries, bounded pagination failures, all-or-nothing behavior, duplicate records across pages, and request immutability without AWS access.

## Phase 3.3.9 checkpointed ingestion

`data_ingestion/checkpointed_ingestion.py` provides `run_checkpointed_ingestion()` for offline-tested, page-at-a-time orchestration. Each page is validated, persisted, and checkpointed in one SQLite transaction before the next page is requested. A completed page advances `page_committed`; the terminal page advances `completed`. Failures are recorded as `failed` when possible, and no page progress is returned unless its transaction commits.

The orchestrator binds runs to the existing query fingerprint and preserves the existing `datapoint_identity` first-durable-record policy. An `InvalidNextToken` response can move a run through `needs_recovery`, clear the transient token, and replay the same bounded query window; durable identities prevent replay duplicates. This phase does not add AWS clients, scheduling, checkpoint integration to the lower-level CloudWatch collector, or production restart orchestration. A future operational layer must decide how to handle stale runs and token recovery across process restarts.

## Phase 2.2 stronger rules

The default detector behavior is unchanged. The optional heat-stress rule flags an observation only when both `temperature >= ANOMALY_HEAT_STRESS_TEMPERATURE_MIN` and `humidity >= ANOMALY_HEAT_STRESS_HUMIDITY_MIN`; its default thresholds are `35 C` and `80%`. The rule is deliberately separate from the univariate range rules and is enabled with:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --enable-cross-feature-rules
```

This is an explicit operational screening rule for a combined hot/humid condition, not a learned correlation. It is configurable through `.env`, deterministic, and reports `heat_stress_condition` in `anomaly_reasons`.

Station-specific baselines were investigated against the current raw dataset. There are only 24 observations per station, and injected anomalies contaminate two stations' pressure distributions (`850 hPa` values). The observed station means are close for temperature and humidity, while wind and pressure differences are not reliable evidence of persistent station behavior. No station baseline was added, avoiding arbitrary thresholds and leakage. A future baseline implementation would require a separate reference-history fit/transform API, a minimum sample policy, and held-out evaluation data. Unseen stations would need to use the existing global rules as fallback.

## Configuration

`.env` is optional. Copy `.env.example` only when overrides are needed. Supported non-secret values include station count, observations per station, seed, anomaly rate, log level, and raw/processed paths or filenames. No credentials are required.

## Data responsibilities

Raw CSV files are source observations and are not changed by preprocessing. Processed CSV files contain the same retained observations in normalized UTC/type form and preserve `is_anomaly`. Generated datasets are ignored by Git; see `data/README.md`.

## Known limitations

Phase 1, Phase 2, and Phase 2.2 intentionally have no ML model, TensorFlow, AWS integration, cloud storage, live weather feed, or advanced imputation. Synthetic tests validate deterministic rule application, not real-world generalization. Physical limits are documented as context only: unusual values can be deliberate ground truth and remain in the dataset. Rule thresholds must be reviewed for a different weather domain or sensor population.
