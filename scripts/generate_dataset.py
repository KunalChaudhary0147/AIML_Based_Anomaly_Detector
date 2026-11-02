"""Generate a raw synthetic weather CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anomaly_detector.config import settings
from anomaly_detector.data_ingestion.weather_generator import generate_weather_data


def main() -> int:
    """Run the dataset generation command."""
    parser = argparse.ArgumentParser(description="Generate synthetic weather observations.")
    parser.add_argument("--stations", type=int, default=settings.DEFAULT_STATION_COUNT)
    parser.add_argument("--observations", type=int, default=settings.DEFAULT_OBSERVATIONS_PER_STATION)
    parser.add_argument("--seed", type=int, default=settings.DEFAULT_RANDOM_SEED)
    parser.add_argument("--anomaly-rate", type=float, default=settings.DEFAULT_ANOMALY_RATE)
    parser.add_argument("--output", type=Path, default=settings.RAW_DATA_DIR / settings.DEFAULT_RAW_FILENAME)
    args = parser.parse_args()
    try:
        data = generate_weather_data(args.stations, args.observations, args.seed, args.anomaly_rate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serializable = data.copy()
        serializable["timestamp"] = serializable["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        serializable.to_csv(args.output, index=False)
        print(f"Generated {len(data)} rows for {data['station_id'].nunique()} stations; anomalies={int(data['is_anomaly'].sum())}; output={args.output}")
        return 0
    except Exception as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
