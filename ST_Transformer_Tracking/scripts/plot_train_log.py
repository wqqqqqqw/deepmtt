#!/usr/bin/env python3
"""Parse train.log and plot loss / turn-rate curves versus step."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS_RE = re.compile(r"loss\s+tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
STEP_TURN_RE = re.compile(
    r"step\s+(\d+)\s*,\s*Tracking\s+RMSE\s+of\s+Turn\s+rate\s*:\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def parse_log(log_path: Path):
    """Return aligned lists: steps, losses, turn_rates."""
    steps = []
    losses = []
    turn_rates = []
    pending_loss = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            loss_match = LOSS_RE.search(line)
            if loss_match:
                pending_loss = float(loss_match.group(1))
                continue

            step_match = STEP_TURN_RE.search(line)
            if step_match:
                step = int(step_match.group(1))
                turn_rate = float(step_match.group(2))

                # Only keep points that have both loss and turn-rate.
                if pending_loss is not None:
                    steps.append(step)
                    losses.append(pending_loss)
                    turn_rates.append(turn_rate)
                    pending_loss = None

    return steps, losses, turn_rates


def plot_curves(steps, losses, turn_rates, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(steps, losses, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curves from train.log")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(steps, turn_rates, color="#d62728", linewidth=1.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Turn Rate RMSE")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Read train.log and plot loss / turn-rate vs step."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "logs" / "train.log",
        help="Path to train.log",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "train_curves.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"Log file not found: {args.log}")

    steps, losses, turn_rates = parse_log(args.log)
    if not steps:
        raise RuntimeError("No valid (step, loss, turn-rate) records found in log.")
    step_len = len(steps)
    plot_curves(steps[:1000], losses[:1000], turn_rates[:1000], args.out)

    print(f"Parsed points: {len(steps)}")
    print(f"Step range: {steps[0]} -> {steps[-1]}")
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
