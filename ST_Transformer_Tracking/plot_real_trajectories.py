#!/usr/bin/env python3
"""Load MTT_TrainingData.mat and plot a subset of real trajectories."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot partial real trajectories from MTT_TrainingData.mat"
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("MTT_TrainingData.mat"),
        help="Path to MTT_TrainingData.mat",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("real_trajectories.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=12,
        help="Number of trajectories to plot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling trajectories",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "head"],
        default="random",
        help="Sampling mode: random or first N",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window after saving",
    )
    return parser.parse_args()


def choose_indices(total: int, n: int, mode: str, seed: int):
    n = min(max(n, 1), total)
    if mode == "head":
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=n, replace=False)


def main():
    args = parse_args()
    if not args.mat.exists():
        raise FileNotFoundError(f"MAT file not found: {args.mat}")

    data = loadmat(args.mat)
    if "ori_traj_set" not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"'ori_traj_set' not found. Available keys: {keys}")

    # ori_traj_set shape: [num_samples, seq_len, 4], where [:, :, 0:2] are x/y.
    ori_traj_set = np.asarray(data["ori_traj_set"])
    if ori_traj_set.ndim != 3 or ori_traj_set.shape[2] < 2:
        raise ValueError(
            f"Unexpected ori_traj_set shape: {ori_traj_set.shape}, expected [N, T, >=2]."
        )

    total = ori_traj_set.shape[0]
    indices = choose_indices(total, args.num, args.mode, args.seed)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx in indices:
        traj = ori_traj_set[idx]
        x = traj[:, 0]
        y = traj[:, 1]
        ax.plot(x, y, linewidth=1.2, alpha=0.85)
        ax.scatter(x[0], y[0], s=16, marker="o")

    ax.set_title(f"Real Trajectories ({len(indices)} samples)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axis("equal")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"Saved figure: {args.out}")
    print(f"Plotted trajectories: {len(indices)} / {total}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
