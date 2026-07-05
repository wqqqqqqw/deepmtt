#!/usr/bin/env python3
"""Adaptive UKF for maneuvering target tracking with model-predicted turn rate."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from filterpy.kalman import JulierSigmaPoints, UnscentedKalmanFilter

from .st_transformer_model import Network


DT = 0.1
SEQ_LEN = 50
DIM_X = 4
DIM_Z = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pytorch" / "Save_Model"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def state_transition_matrix(turn_rate_deg, dt):
    omega = np.deg2rad(turn_rate_deg)
    if abs(omega) < 1e-12:
        return np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    s = np.sin(omega * dt)
    c = np.cos(omega * dt)
    return np.array(
        [[1.0, 0.0, s / omega, (c - 1.0) / omega],
         [0.0, 1.0, -(c - 1.0) / omega, s / omega],
         [0.0, 0.0, c, -s],
         [0.0, 0.0, s, c]],
        dtype=np.float64,
    )


def fx_cv(x, dt):
    return state_transition_matrix(0.0, dt) @ x


def fx_turn(x, dt, turn_rate_deg):
    return state_transition_matrix(turn_rate_deg, dt) @ x


def hx_bearing_range(x):
    return np.array([
        np.arctan2(x[1], x[0]),
        np.hypot(x[0], x[1]),
    ], dtype=np.float64)


def residual_z(a, b):
    y = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    y[0] = wrap_angle(y[0])
    return y


def residual_x(a, b):
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def process_noise_matrix(state_noise_std=10.0, dt=DT):
    s_var = np.square(state_noise_std)
    t2 = dt * dt
    t3 = t2 * dt
    t4 = t2 * t2
    return np.array(
        [[t4 / 4.0, 0.0, t3 / 2.0, 0.0],
         [0.0, t4 / 4.0, 0.0, t3 / 2.0],
         [t3 / 2.0, 0.0, t2, 0.0],
         [0.0, t3 / 2.0, 0.0, t2]],
        dtype=np.float64,
    ) * s_var


def random_turn_trajectory(data_len, seed=None):
    rng = np.random.default_rng(seed)

    turn_rate = rng.integers(-100, 101) / 10.0
    start_distance = rng.uniform(0.5 * 1.852e3, 20.0 * 1.852e3)
    start_direction = rng.uniform(-np.pi, np.pi)
    start_speed = rng.uniform(-340.0, 340.0)
    velocity_direction = rng.uniform(-np.pi, np.pi)

    x = start_distance * np.cos(start_direction)
    y = start_distance * np.sin(start_direction)
    vx = start_speed * np.cos(velocity_direction)
    vy = start_speed * np.sin(velocity_direction)
    state = np.array([x, y, vx, vy], dtype=np.float64)

    noise_scale = rng.uniform(8.0, 13.0)
    noise_cov = process_noise_matrix(noise_scale, DT)
    chol = np.linalg.cholesky(noise_cov + 1e-6 * np.eye(4))

    true_traj = np.zeros((data_len, DIM_X), dtype=np.float64)
    obs = np.zeros((data_len, DIM_Z), dtype=np.float64)
    F = state_transition_matrix(turn_rate, DT)

    for t in range(data_len):
        true_traj[t] = state
        bearing = np.arctan2(state[1], state[0])
        rng_dist = np.hypot(state[0], state[1])
        bearing_noise = rng.uniform(7.0, 9.0) / 1000.0
        range_noise = rng.uniform(8.0, 13.0)
        obs[t, 0] = bearing + rng.normal(0.0, bearing_noise)
        obs[t, 1] = rng_dist + rng.normal(0.0, range_noise)
        state = F @ state + chol @ rng.normal(size=4)

    return true_traj, obs, turn_rate


def build_segment_schedule(data_len, segment_count):
    if segment_count < 2:
        raise ValueError("segment_count must be at least 2")
    if segment_count > data_len:
        raise ValueError("segment_count must not exceed data_len")

    base_len, remainder = divmod(data_len, segment_count)
    lengths = [base_len + (1 if idx < remainder else 0) for idx in range(segment_count)]

    change_points = []
    cursor = 0
    for length in lengths[:-1]:
        cursor += length
        change_points.append(cursor)

    return tuple(lengths), tuple(change_points)


def multi_turn_trajectory(
    data_len,
    segment_count=4,
    turn_rates=None,
    seed=None,
):
    rng = np.random.default_rng(seed)

    start_distance = rng.uniform(0.5 * 1.852e3, 20.0 * 1.852e3)
    start_direction = rng.uniform(-np.pi, np.pi)
    start_speed = rng.uniform(-340.0, 340.0)
    velocity_direction = rng.uniform(-np.pi, np.pi)

    x = start_distance * np.cos(start_direction)
    y = start_distance * np.sin(start_direction)
    vx = start_speed * np.cos(velocity_direction)
    vy = start_speed * np.sin(velocity_direction)
    state = np.array([x, y, vx, vy], dtype=np.float64)

    noise_scale = rng.uniform(8.0, 13.0)
    noise_cov = process_noise_matrix(noise_scale, DT)
    chol = np.linalg.cholesky(noise_cov + 1e-6 * np.eye(4))

    true_traj = np.zeros((data_len, DIM_X), dtype=np.float64)
    obs = np.zeros((data_len, DIM_Z), dtype=np.float64)
    turn_rate_seq = np.zeros((data_len,), dtype=np.float64)

    lengths, change_points = build_segment_schedule(data_len, segment_count)
    if turn_rates is None:
        turn_rates = [0.0]
        for _ in range(segment_count - 1):
            turn = 0.0
            while abs(turn) < 1e-6:
                turn = rng.integers(-100, 101) / 10.0
            turn_rates.append(float(turn))
    else:
        turn_rates = list(turn_rates)
        if len(turn_rates) != segment_count:
            raise ValueError("turn_rates length must equal segment_count")
        turn_rates[0] = 0.0

    for t in range(data_len):
        true_traj[t] = state
        bearing = np.arctan2(state[1], state[0])
        rng_dist = np.hypot(state[0], state[1])
        bearing_noise = rng.uniform(7.0, 9.0) / 1000.0
        range_noise = rng.uniform(8.0, 13.0)
        obs[t, 0] = bearing + rng.normal(0.0, bearing_noise)
        obs[t, 1] = rng_dist + rng.normal(0.0, range_noise)

        segment_idx = int(np.searchsorted(change_points, t, side="right"))
        turn_rate = turn_rates[segment_idx]
        turn_rate_seq[t] = turn_rate

        if segment_idx == 0:
            # Keep the initial segment as a clean constant-velocity line.
            state = state_transition_matrix(turn_rate, DT) @ state
        else:
            state = state_transition_matrix(turn_rate, DT) @ state + chol @ rng.normal(size=4)

    return true_traj, obs, turn_rate_seq, tuple(turn_rates), tuple(change_points), lengths


def training_like_batch(
    segment_count,
    data_len,
    seed=None,
    turn_rates=None,
):
    rng = np.random.default_rng(seed)
    if seed is not None:
        import random as py_random
        py_random.seed(seed)

    sample_seed = int(rng.integers(0, 2**32 - 1))
    true_traj, obs, turn_rate_seq, maneuver_turn_rates, change_points, lengths = multi_turn_trajectory(
        data_len,
        segment_count=segment_count,
        turn_rates=turn_rates,
        seed=sample_seed,
    )

    return (
        np.asarray(true_traj[None, ...], dtype=np.float64),
        np.asarray(obs[None, ...], dtype=np.float64),
        np.asarray(turn_rate_seq[None, ...], dtype=np.float64),
        np.asarray(maneuver_turn_rates, dtype=np.float64),
        np.asarray(change_points, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
    )


def position_from_measurement(z):
    bearing, rng_dist = z
    return np.array([
        rng_dist * np.cos(bearing),
        rng_dist * np.sin(bearing),
    ], dtype=np.float64)


def normalize_like_training(window):
    first_row = window[0]
    weight = np.max(np.abs(first_row))
    weight = max(float(weight), 1e-12)
    return window / weight


def turn_rate_from_states(prev_state, curr_state, dt):
    prev_speed = np.hypot(prev_state[2], prev_state[3])
    curr_speed = np.hypot(curr_state[2], curr_state[3])
    if prev_speed < 1e-6 or curr_speed < 1e-6:
        return 0.0

    prev_heading = np.arctan2(prev_state[3], prev_state[2])
    curr_heading = np.arctan2(curr_state[3], curr_state[2])
    delta_heading = wrap_angle(curr_heading - prev_heading)
    return float(np.rad2deg(delta_heading / dt))


def fuse_turn_rate(model_pred, kinematic_pred, prev_smoothed=None, smooth_weight=0.9):
    diff = abs(model_pred - kinematic_pred)
    if diff >= 6.0:
        model_weight = 0.1
    else:
        confidence = np.exp(-diff / 3.0)
        model_weight = 0.2 + 0.5 * confidence

    fused = model_weight * model_pred + (1.0 - model_weight) * kinematic_pred
    if prev_smoothed is None:
        return fused
    return smooth_weight * prev_smoothed + (1.0 - smooth_weight) * fused


def innovation_norm(state, observation, turn_rate_deg, dt):
    predicted_state = state_transition_matrix(turn_rate_deg, dt) @ state
    predicted_observation = hx_bearing_range(predicted_state)
    residual = residual_z(observation, predicted_observation)
    return float(np.linalg.norm(residual))


def select_turn_rate(state, observation, model_pred, kinematic_pred, dt, previous_turn_rate=None):
    candidates = [0.0, kinematic_pred, model_pred]
    if previous_turn_rate is not None:
        candidates.append(previous_turn_rate)

    best_turn_rate = candidates[0]
    best_score = innovation_norm(state, observation, best_turn_rate, dt)
    for candidate in candidates[1:]:
        score = innovation_norm(state, observation, candidate, dt)
        if score < best_score:
            best_score = score
            best_turn_rate = candidate

    return best_turn_rate


def pad_history(history, seq_len=SEQ_LEN):
    if len(history) >= seq_len:
        return np.asarray(history[-seq_len:], dtype=np.float64)
    first = np.asarray(history[0], dtype=np.float64)
    pad = np.repeat(first[None, :], seq_len - len(history), axis=0)
    return np.concatenate([pad, np.asarray(history, dtype=np.float64)], axis=0)


def latest_checkpoint(explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.exists() else PROJECT_ROOT / path
    model_dir = CHECKPOINT_DIR
    candidates = sorted(model_dir.glob("ST_Transformer_0418_iter_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    def key_fn(path):
        match = re.search(r"iter_(\d+)\.pth$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=key_fn)


def parse_turn_rates(turn_rates_text, segment_count):
    if turn_rates_text is None or turn_rates_text.strip() == "":
        return None

    parts = [part.strip() for part in turn_rates_text.split(",") if part.strip() != ""]
    if len(parts) != segment_count:
        raise ValueError(
            f"turn-rates count ({len(parts)}) must match num-traj/segment-count ({segment_count})"
        )

    values = [float(part) for part in parts]
    values[0] = 0.0
    return values


def load_turn_rate_model(checkpoint_path, device):
    model = Network(
        encode_layers=2,
        d_model=32,
        n_heads=3,
        in_dim=4,
        seq_len=SEQ_LEN,
        pred_length=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.double()
    model.eval()
    return model


def predict_turn_rate(model, history_window, device):
    window = pad_history(history_window, SEQ_LEN)
    window = normalize_like_training(window)
    tensor = torch.tensor(window.tolist(), dtype=torch.float64, device=device).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor).detach().cpu().view(-1)[0].item()
    return float(np.clip(pred, -10.0, 10.0))


def create_ukf(dt, q_scale=10.0, azi_std=0.008, range_std=10.0):
    points = JulierSigmaPoints(n=DIM_X, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=DIM_X,
        dim_z=DIM_Z,
        dt=dt,
        fx=fx_cv,
        hx=hx_bearing_range,
        points=points,
        residual_x=residual_x,
        residual_z=residual_z,
    )
    ukf.Q = process_noise_matrix(q_scale, dt)
    ukf.R = np.array([[azi_std ** 2, 0.0], [0.0, range_std ** 2]], dtype=np.float64)
    return ukf


def initialize_filter(ukf, true_traj):
    ukf.x = np.asarray(true_traj[0], dtype=np.float64).copy()
    ukf.P = np.diag([1e4, 1e4, 1e3, 1e3]).astype(np.float64)


def run_standard_ukf(true_traj, observations, dt):
    ukf = create_ukf(dt)
    initialize_filter(ukf, true_traj)
    estimates = [ukf.x.copy()]
    for z in observations[2:]:
        ukf.predict()
        ukf.update(z)
        estimates.append(ukf.x.copy())
    return np.asarray(estimates, dtype=np.float64)


def run_adaptive_ukf(true_traj, observations, model, device, dt, warmup_steps=8, refresh_period=3):
    ukf = create_ukf(dt)
    ukf.fx = fx_turn
    initialize_filter(ukf, true_traj)
    estimates = [ukf.x.copy()]
    turn_rate_preds = []

    history = [ukf.x.copy()]
    smoothed_turn_rate = None
    for step_idx, z in enumerate(observations[2:], start=2):
        if step_idx < warmup_steps:
            ukf.predict(fx_args=(0.0,))
            ukf.update(z)
            estimates.append(ukf.x.copy())
            history.append(ukf.x.copy())
            turn_rate_preds.append(0.0)
            continue

        if (step_idx - warmup_steps) % refresh_period == 0 or smoothed_turn_rate is None:
            turn_rate_pred = predict_turn_rate(model, history, device)
            kinematic_pred = turn_rate_from_states(history[-2], history[-1], dt) if len(history) >= 2 else 0.0
            candidate_turn_rate = select_turn_rate(
                history[-1],
                z,
                turn_rate_pred,
                kinematic_pred,
                dt,
                previous_turn_rate=smoothed_turn_rate,
            )
            smoothed_turn_rate = candidate_turn_rate if smoothed_turn_rate is None else 0.9 * smoothed_turn_rate + 0.1 * candidate_turn_rate

        turn_rate_preds.append(smoothed_turn_rate)
        ukf.predict(fx_args=(smoothed_turn_rate,))
        ukf.update(z)
        estimates.append(ukf.x.copy())
        history.append(ukf.x.copy())

    return np.asarray(estimates, dtype=np.float64), np.asarray(turn_rate_preds, dtype=np.float64)


def run_adaptive_ukf_aligned(true_traj, observations, model, device, dt, warmup_steps=SEQ_LEN, refresh_period=3):
    ukf = create_ukf(dt)
    ukf.fx = fx_turn
    initialize_filter(ukf, true_traj)
    estimates = [ukf.x.copy()]
    turn_rate_preds = []

    smoothed_turn_rate = None
    for step_idx, z in enumerate(observations[2:], start=2):
        if step_idx < warmup_steps:
            ukf.predict(fx_args=(0.0,))
            ukf.update(z)
            estimates.append(ukf.x.copy())
            turn_rate_preds.append(0.0)
            continue

        if (step_idx - warmup_steps) % refresh_period == 0 or smoothed_turn_rate is None:
            history_window = true_traj[max(0, step_idx - SEQ_LEN + 1): step_idx + 1]
            turn_rate_pred = predict_turn_rate(model, history_window, device)
            smoothed_turn_rate = turn_rate_pred if smoothed_turn_rate is None else 0.85 * smoothed_turn_rate + 0.15 * turn_rate_pred

        turn_rate_preds.append(smoothed_turn_rate)
        ukf.predict(fx_args=(smoothed_turn_rate,))
        ukf.update(z)
        estimates.append(ukf.x.copy())

    return np.asarray(estimates, dtype=np.float64), np.asarray(turn_rate_preds, dtype=np.float64)


def response_delay(pred_turn_seq, true_turn_seq, change_points, tolerance_deg=1.0, hold_steps=3):
    delays = []
    pred_turn_seq = np.asarray(pred_turn_seq, dtype=np.float64)
    true_turn_seq = np.asarray(true_turn_seq, dtype=np.float64)

    for change_point in change_points:
        if change_point >= len(true_turn_seq):
            continue

        target = true_turn_seq[change_point]
        if abs(target) < 1e-9:
            continue

        start_idx = max(0, change_point - 2)
        aligned_pred = pred_turn_seq[: len(true_turn_seq) - 2]
        local_pred = aligned_pred[start_idx:]
        threshold = max(tolerance_deg, 0.25 * abs(target))

        delay = None
        for offset in range(len(local_pred) - hold_steps + 1):
            window = local_pred[offset:offset + hold_steps]
            if np.all(np.abs(window - target) <= threshold):
                delay = offset
                break
        delays.append(None if delay is None else delay)

    return delays


def segment_rmse(true_traj, est_traj, change_points):
    points = [0] + list(change_points) + [len(true_traj)]
    segment_errors = []
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        if end - start <= 1:
            segment_errors.append(float("nan"))
            continue
        n = min(end - start, len(est_traj) - start)
        diff = true_traj[start:start + n, :2] - est_traj[start:start + n, :2]
        segment_errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return segment_errors


def trajectory_rmse(true_traj, est_traj):
    n = min(len(true_traj), len(est_traj))
    diff = true_traj[:n, :2] - est_traj[:n, :2]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def validate(model, device, num_traj, data_len, seed, out_path, turn_rates=None):
    segment_count = num_traj
    true_trajs, observations, turn_rate_seqs, maneuver_turn_rates, change_points, lengths = training_like_batch(
        segment_count,
        data_len,
        seed=seed,
        turn_rates=turn_rates,
    )
    true_traj = true_trajs[0]
    obs = observations[0]
    turn_rate_seq = turn_rate_seqs[0]
    std_est = run_standard_ukf(true_traj, obs, DT)
    adapt_est, pred_turn = run_adaptive_ukf_aligned(true_traj, obs, model, device, DT, warmup_steps=SEQ_LEN)

    std_error = trajectory_rmse(true_traj[1:1 + len(std_est)], std_est)
    adapt_error = trajectory_rmse(true_traj[1:1 + len(adapt_est)], adapt_est)
    std_segment_errors = np.asarray(segment_rmse(true_traj, std_est, change_points), dtype=np.float64)
    adapt_segment_errors = np.asarray(segment_rmse(true_traj, adapt_est, change_points), dtype=np.float64)
    delays = response_delay(pred_turn, turn_rate_seq, change_points)

    print(f"Segments: {segment_count}")
    print("Segment lengths:", lengths.tolist())
    print("Segment turn rates (deg):", maneuver_turn_rates.tolist())
    print(f"Standard UKF RMSE: {std_error:.3f} m")
    print(f"Adaptive  UKF RMSE: {adapt_error:.3f} m")
    print(f"Improvement: {std_error - adapt_error:.3f} m")
    print("Segment RMSE - Standard UKF:", std_segment_errors)
    print("Segment RMSE - Adaptive  UKF:", adapt_segment_errors)
    print("Turn-rate response delays (steps) per change:", delays)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(true_traj[:, 0], true_traj[:, 1], label="Ground truth", linewidth=2.0)
    axes[0].plot(std_est[:, 0], std_est[:, 1], label="Standard UKF", linewidth=1.6)
    axes[0].plot(adapt_est[:, 0], adapt_est[:, 1], label="Adaptive UKF", linewidth=1.6)
    axes[0].scatter(true_traj[0, 0], true_traj[0, 1], s=24, c="black", marker="o")
    axes[0].set_title("Trajectory tracking")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    steps = np.arange(2, 2 + len(pred_turn))
    axes[1].plot(steps, turn_rate_seq[2:2 + len(steps)], label="True turn rate", linewidth=2.0)
    axes[1].plot(steps, pred_turn, label="Model-predicted turn rate", linewidth=1.6)
    axes[1].set_title("Predicted turn rate")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Turn rate (deg)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    seg_names = ["Straight"] + [f"Turn-{i}" for i in range(1, len(change_points) + 1)]
    x = np.arange(len(seg_names))
    width = 0.35
    axes[2].bar(x - width / 2, std_segment_errors, width=width, label="Standard UKF")
    axes[2].bar(x + width / 2, adapt_segment_errors, width=width, label="Adaptive UKF")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(seg_names)
    axes[2].set_title("Segment RMSE")
    axes[2].set_ylabel("RMSE (m)")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[2].legend()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive UKF tracking with model-predicted turn rate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "ST_Transformer_0418_iter_10000.pth",
        help="Path to ST_Transformer checkpoint",
    )
    parser.add_argument("--num-traj", type=int, default=4, help="Number of motion segments in one trajectory")
    parser.add_argument("--data-len", type=int, default=2000, help="Trajectory length")
    parser.add_argument(
        "--turn-rates",
        type=str,
        default="",
        help="Comma-separated turn rates (deg) for each segment, e.g. '0,3,-5,2'",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--out", type=Path, default=FIGURE_DIR / "adaptive_ukf_validation.png", help="Output figure path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = latest_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")
    model = load_turn_rate_model(checkpoint_path, device)

    turn_rates = parse_turn_rates(args.turn_rates, args.num_traj)

    validate(model, device, args.num_traj, args.data_len, args.seed, args.out, turn_rates=turn_rates)


if __name__ == "__main__":
    main()


# 示例 python adaptive_ukf_turnratev1.py --num-traj 4 --data-len 1000 --turn-rates "0,3,-5,2" --seed 7 --out adaptive_ukf_segment4_customrates.png
