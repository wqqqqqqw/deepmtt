#!/usr/bin/env python3
"""Compatibility wrapper for src/deepmtt/trajectory_data_generator.py."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepmtt.trajectory_data_generator import *  # noqa: F401,F403
