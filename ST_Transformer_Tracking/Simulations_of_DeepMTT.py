#!/usr/bin/env python3
"""Compatibility wrapper for scripts/legacy/Simulations_of_DeepMTT.py."""

from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
runpy.run_path(str(PROJECT_ROOT / "scripts" / "legacy" / "Simulations_of_DeepMTT.py"), run_name="__main__")
