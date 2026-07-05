#!/usr/bin/env python3
"""Validate adaptive UKF tracking with the ST-Transformer turn-rate model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepmtt.adaptive_ukf_turnrate import main


if __name__ == "__main__":
    main()


# python scripts/validate_adaptive_ukf.py --num-traj 3 --data-len 1000 --turn-rates "0,6,-6" --seed 7 --out adaptive_ukf_segment4_customrates_test.png
