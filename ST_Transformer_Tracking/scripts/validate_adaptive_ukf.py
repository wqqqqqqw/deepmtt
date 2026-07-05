#!/usr/bin/env python3
"""Validate adaptive UKF tracking with the ST-Transformer turn-rate model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepmtt.adaptive_ukf_turnrate import main


if __name__ == "__main__":
    main()
