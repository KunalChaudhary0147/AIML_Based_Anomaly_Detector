"""Vercel ASGI entrypoint for the existing FastAPI application."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detector.api.app import app

__all__ = ["app"]
