#!/usr/bin/env python3
"""Compatibility wrapper for src/deepmtt/training_data.py."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepmtt.training_data import *  # noqa: F401,F403
