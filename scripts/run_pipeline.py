"""Run the weather preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anomaly_detector.config import settings
from anomaly_detector.pipeline import run_pipeline


def main() -> int:
    """Run the pipeline command."""
    parser = argparse.ArgumentParser(description="Validate, preprocess, and save a weather CSV.")
    parser.add_argument("--input", type=Path, default=settings.RAW_DATA_DIR / settings.DEFAULT_RAW_FILENAME)
    parser.add_argument("--output", type=Path, default=settings.PROCESSED_DATA_DIR / settings.DEFAULT_PROCESSED_FILENAME)
    parser.add_argument("--predictions", type=Path, default=settings.PROCESSED_DATA_DIR / settings.DEFAULT_PREDICTIONS_FILENAME)
    parser.add_argument("--enable-cross-feature-rules", action="store_true", help="Enable opt-in heat-stress cross-feature detection.")
    args = parser.parse_args()
    try:
        summary = run_pipeline(args.input, args.output, args.predictions, enable_cross_feature_rules=args.enable_cross_feature_rules)
        metrics = summary.evaluation
        metric_text = "evaluation unavailable" if metrics is None else f"precision={metrics.precision:.3f}, recall={metrics.recall:.3f}, f1={metrics.f1_score:.3f}"
        print(f"Processed {summary.output_rows} rows across {summary.station_count} stations; anomalies={summary.anomaly_count}; output={summary.output_path}; predictions={summary.prediction_path}; {metric_text}")
        return 0
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
