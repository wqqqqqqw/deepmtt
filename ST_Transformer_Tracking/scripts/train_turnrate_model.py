#!/usr/bin/env python3
"""Train the ST-Transformer turn-rate model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepmtt.st_transformer_model import main


if __name__ == "__main__":
    main()
