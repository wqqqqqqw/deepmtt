#!/usr/bin/env python3
"""Compatibility wrapper for scripts/plot_train_log.py."""

from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
runpy.run_path(str(PROJECT_ROOT / "scripts" / "plot_train_log.py"), run_name="__main__")
