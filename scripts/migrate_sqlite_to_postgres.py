"""Copy the existing SQLite application database into PostgreSQL."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from anomaly_detector.data_ingestion.telemetry_persistence import _connect, initialize_database


def migrate(source: Path, target: str) -> dict[str, int]:
    if not source.exists():
        raise FileNotFoundError(source)
    initialize_database(target)
    source_connection = sqlite3.connect(source)
    source_connection.row_factory = sqlite3.Row
    target_connection = _connect(target)
    copied: dict[str, int] = {}
    try:
        tables = [
            row[0] for row in source_connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )
        ]
        target_connection.execute("BEGIN")
        for table in tables:
            columns = [
                row[1] for row in source_connection.execute(f'PRAGMA table_info("{table}")')
            ]
            quoted_columns = ", ".join(f'"{column}"' for column in columns)
            placeholders = ", ".join("?" for _ in columns)
            rows = source_connection.execute(f'SELECT {quoted_columns} FROM "{table}"')
            count = 0
            for row in rows:
                target_connection.execute(
                    f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders})',
                    tuple(row[column] for column in columns),
                )
                count += 1
            copied[table] = count
        target_connection.commit()
    except Exception:
        target_connection.rollback()
        raise
    finally:
        target_connection.close()
        source_connection.close()
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SQLite database to copy")
    parser.add_argument("--target", required=True, help="PostgreSQL DATABASE_URL")
    args = parser.parse_args()
    print(migrate(args.source, args.target))


if __name__ == "__main__":
    main()
