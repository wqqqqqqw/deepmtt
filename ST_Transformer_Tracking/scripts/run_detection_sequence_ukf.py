#!/usr/bin/env python3
"""Track all targets from multi-target detection masks and run Adaptive UKF."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from deepmtt.adaptive_ukf_turnrate import (
    DT,
    CHECKPOINT_DIR,
    load_turn_rate_model,
    run_adaptive_ukf,
    run_standard_ukf,
    trajectory_rmse,
)


@dataclass
class Detection:
    frame_idx: int
    det_id: int
    cx: float
    cy: float
    x1: int
    y1: int
    x2: int
    y2: int
    area: int


@dataclass
class Track:
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self):
        return self.detections[-1]

    def append(self, detection):
        self.detections.append(detection)
        self.missed = 0

    def extend(self, detections):
        self.detections.extend(detections)
        self.detections.sort(key=lambda det: det.frame_idx)
        self.missed = 0

    @property
    def start_frame(self):
        return self.detections[0].frame_idx

    @property
    def end_frame(self):
        return self.detections[-1].frame_idx

    @property
    def velocity(self):
        if len(self.detections) < 2:
            return np.zeros(2, dtype=np.float64)
        prev = self.detections[-2]
        curr = self.detections[-1]
        frame_delta = max(curr.frame_idx - prev.frame_idx, 1)
        return np.array([(curr.cx - prev.cx) / frame_delta, (curr.cy - prev.cy) / frame_delta], dtype=np.float64)

    def predict_position(self, frame_idx):
        last = self.last
        delta = max(frame_idx - last.frame_idx, 1)
        return np.array([last.cx, last.cy], dtype=np.float64) + self.velocity * delta


def extract_components(mask_path, min_area=1):
    mask = np.asarray(Image.open(mask_path).convert("L"))
    labeled, component_count = ndimage.label(mask > 0)
    detections = []
    for label_id in range(1, component_count + 1):
        ys, xs = np.nonzero(labeled == label_id)
        area = len(xs)
        if area < min_area:
            continue
        detections.append({
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
            "area": int(area),
        })
    detections.sort(key=lambda item: (-item["area"], item["cy"], item["cx"]))
    return detections


def write_components_txt(frame_path, frame_id, detections):
    with frame_path.open("w", encoding="utf-8") as f:
        f.write("frame det_id cx cy x1 y1 x2 y2 area\n")
        for det_id, det in enumerate(detections):
            f.write(
                f"{frame_id} {det_id} {det['cx']:.6f} {det['cy']:.6f} "
                f"{det['x1']} {det['y1']} {det['x2']} {det['y2']} {det['area']}\n"
            )


def generate_multitarget_txt(result_root, sequence, output_root, min_area=1, keep_frame_txt=False):
    result_root = Path(result_root)
    output_root = Path(output_root)
    pred_dir = result_root / "predict" / sequence
    gt_dir = result_root / "序列真值掩码" / sequence
    if not gt_dir.exists():
        alt_dir = result_root / "序列真值掩码" / sequence.replace("-", "_")
        if alt_dir.exists():
            gt_dir = alt_dir
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth mask directory not found: {gt_dir}")

    seq_root = output_root / sequence
    pred_frame_dir = seq_root / "predict_frames"
    gt_frame_dir = seq_root / "gt_frames"
    if keep_frame_txt:
        pred_frame_dir.mkdir(parents=True, exist_ok=True)
        gt_frame_dir.mkdir(parents=True, exist_ok=True)
    else:
        for frame_dir in (pred_frame_dir, gt_frame_dir):
            if frame_dir.exists():
                shutil.rmtree(frame_dir)

    pred_summary = seq_root / f"{sequence}_predict_components.txt"
    gt_summary = seq_root / f"{sequence}_gt_components.txt"
    pred_rows = []
    gt_rows = []
    frame_ids = []
    frame_detections = []
    gt_components = {}

    pred_paths = sorted(pred_dir.glob("*.png"))
    progress = tqdm(pred_paths, desc=f"component txt {sequence}", unit="frame", dynamic_ncols=True)
    for frame_idx, pred_path in enumerate(progress):
        frame_id = pred_path.stem
        frame_ids.append(frame_id)
        pred_components = extract_components(pred_path, min_area=min_area)
        gt_path = gt_dir / pred_path.name
        gt_frame_components = extract_components(gt_path, min_area=min_area) if gt_path.exists() else []
        frame_detections.append([
            Detection(frame_idx, det_id, **det)
            for det_id, det in enumerate(pred_components)
        ])
        gt_components[frame_idx] = gt_frame_components

        if keep_frame_txt:
            write_components_txt(pred_frame_dir / f"{frame_id}.txt", frame_id, pred_components)
            write_components_txt(gt_frame_dir / f"{frame_id}.txt", frame_id, gt_frame_components)

        for det_id, det in enumerate(pred_components):
            pred_rows.append({"frame": frame_id, "frame_idx": frame_idx, "det_id": det_id, **det})
        for det_id, det in enumerate(gt_frame_components):
            gt_rows.append({"frame": frame_id, "frame_idx": frame_idx, "det_id": det_id, **det})

    fieldnames = ["frame", "frame_idx", "det_id", "cx", "cy", "x1", "y1", "x2", "y2", "area"]
    for path, rows in [(pred_summary, pred_rows), (gt_summary, gt_rows)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=" ")
            writer.writeheader()
            writer.writerows(rows)

    print(f"sequence: {sequence}")
    print(f"frames: {len(frame_ids)}")
    print(f"predict components: {len(pred_rows)}")
    print(f"gt components: {len(gt_rows)}")
    print(f"predict summary txt: {pred_summary}")
    print(f"gt summary txt: {gt_summary}")
    if keep_frame_txt:
        print(f"predict frame txt dir: {pred_frame_dir}")
        print(f"gt frame txt dir: {gt_frame_dir}")
    return frame_ids, frame_detections, gt_components, seq_root


def association_gate(track, frame_idx, max_link_distance, gap_distance_growth, velocity_gate_weight):
    gap = max(frame_idx - track.last.frame_idx, 1)
    speed = float(np.linalg.norm(track.velocity))
    return max_link_distance + gap_distance_growth * max(gap - 1, 0) + velocity_gate_weight * speed * gap


def link_tracks(
    frame_detections,
    max_link_distance=35.0,
    max_gap=12,
    velocity_weight=0.5,
    gap_distance_growth=8.0,
    velocity_gate_weight=0.8,
    merge_gap=30,
    merge_distance=45.0,
):
    tracks = []
    active_tracks = []
    progress = tqdm(frame_detections, desc="track link", unit="frame", dynamic_ncols=True)
    for frame_idx, detections in enumerate(progress):
        unused = set(range(len(detections)))
        matched_tracks = set()

        if active_tracks and detections:
            cost_matrix = np.full((len(active_tracks), len(detections)), 1e9, dtype=np.float64)
            for track_idx, track in enumerate(active_tracks):
                predicted_xy = track.predict_position(frame_idx)
                current_velocity = track.velocity
                gate = association_gate(
                    track,
                    frame_idx,
                    max_link_distance,
                    gap_distance_growth,
                    velocity_gate_weight,
                )
                for det_idx, det in enumerate(detections):
                    det_xy = np.array([det.cx, det.cy], dtype=np.float64)
                    distance_cost = float(np.linalg.norm(det_xy - predicted_xy))
                    if distance_cost > gate:
                        continue
                    frame_delta = max(det.frame_idx - track.last.frame_idx, 1)
                    det_velocity = (det_xy - np.array([track.last.cx, track.last.cy], dtype=np.float64)) / frame_delta
                    velocity_cost = float(np.linalg.norm(det_velocity - current_velocity))
                    cost_matrix[track_idx, det_idx] = distance_cost + velocity_weight * velocity_cost

            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            for track_idx, det_idx in zip(row_idx, col_idx):
                if cost_matrix[track_idx, det_idx] >= 1e9:
                    continue
                active_tracks[track_idx].append(detections[det_idx])
                matched_tracks.add(track_idx)
                unused.discard(det_idx)

        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_tracks:
                track.missed += 1

        for track in list(active_tracks):
            if track.missed > max_gap:
                active_tracks.remove(track)

        for det_idx in unused:
            track = Track([detections[det_idx]])
            tracks.append(track)
            active_tracks.append(track)

    raw_track_count = len(tracks)
    tracks = merge_track_segments(
        tracks,
        max_gap=merge_gap,
        max_distance=merge_distance,
        gap_distance_growth=gap_distance_growth,
        velocity_gate_weight=velocity_gate_weight,
    )
    print(f"track segments before merge: {raw_track_count}")
    print(f"track segments after merge: {len(tracks)}")
    tracks.sort(key=lambda track: (len(track.detections), sum(det.area for det in track.detections)), reverse=True)
    return tracks


def merge_track_segments(tracks, max_gap=30, max_distance=45.0, gap_distance_growth=8.0, velocity_gate_weight=0.8):
    tracks = sorted(tracks, key=lambda track: (track.start_frame, track.end_frame))
    removed = set()
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(tracks):
            if i in removed:
                continue
            best_j = None
            best_cost = None
            for j, second in enumerate(tracks):
                if j in removed or i == j:
                    continue
                gap = second.start_frame - first.end_frame
                if gap <= 0 or gap > max_gap:
                    continue
                predicted_xy = first.predict_position(second.start_frame)
                start_xy = np.array([second.detections[0].cx, second.detections[0].cy], dtype=np.float64)
                distance = float(np.linalg.norm(start_xy - predicted_xy))
                gate = max_distance + gap_distance_growth * max(gap - 1, 0) + velocity_gate_weight * float(np.linalg.norm(first.velocity)) * gap
                if distance > gate:
                    continue
                speed_cost = float(np.linalg.norm(second.velocity - first.velocity))
                cost = distance + speed_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_j = j
            if best_j is not None:
                first.extend(tracks[best_j].detections)
                removed.add(best_j)
                changed = True
    return [track for idx, track in enumerate(tracks) if idx not in removed]


def interpolate_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    result = xy.copy()
    for dim in range(result.shape[1]):
        values = result[:, dim]
        valid = np.isfinite(values)
        if valid.all():
            continue
        if not valid.any():
            raise ValueError("No valid coordinates to interpolate")
        idx = np.arange(len(values))
        values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
        result[:, dim] = values
    return result


def build_single_track_inputs(track, frame_ids, gt_components, image_size, output_path):
    start_frame = track.detections[0].frame_idx
    end_frame = track.detections[-1].frame_idx
    selected_frame_ids = frame_ids[start_frame:end_frame + 1]
    frame_count = len(selected_frame_ids)
    pred_xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
    pred_area = np.zeros((frame_count,), dtype=np.float64)
    detected = np.zeros((frame_count,), dtype=np.int64)
    for det in track.detections:
        local_idx = det.frame_idx - start_frame
        pred_xy[local_idx] = [det.cx, det.cy]
        pred_area[local_idx] = det.area
        detected[local_idx] = 1
    pred_xy_interp = interpolate_xy(pred_xy)

    gt_xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
    gt_area = np.zeros((frame_count,), dtype=np.float64)
    for local_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        components = gt_components.get(frame_idx, [])
        if not components:
            continue
        point = pred_xy_interp[local_idx]
        nearest = min(
            components,
            key=lambda det: np.linalg.norm(np.array([det["cx"], det["cy"]], dtype=np.float64) - point),
        )
        gt_xy[local_idx] = [nearest["cx"], nearest["cy"]]
        gt_area[local_idx] = nearest["area"]
    gt_xy_interp = interpolate_xy(gt_xy)

    center = np.array([image_size[0] / 2.0, image_size[1] / 2.0], dtype=np.float64)
    pred_centered = pred_xy_interp - center
    gt_centered = gt_xy_interp - center

    velocities = np.gradient(gt_centered, DT, axis=0)
    true_traj = np.column_stack([gt_centered, velocities]).astype(np.float64)
    observations = np.column_stack([
        np.arctan2(pred_centered[:, 1], pred_centered[:, 0]),
        np.hypot(pred_centered[:, 0], pred_centered[:, 1]),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["frame", "detected", "pred_cx", "pred_cy", "pred_area", "gt_cx", "gt_cy", "gt_area"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=" ")
        writer.writeheader()
        for idx, frame_id in enumerate(selected_frame_ids):
            writer.writerow({
                "frame": frame_id,
                "detected": int(detected[idx]),
                "pred_cx": float(pred_xy_interp[idx, 0]),
                "pred_cy": float(pred_xy_interp[idx, 1]),
                "pred_area": float(pred_area[idx]),
                "gt_cx": float(gt_xy_interp[idx, 0]),
                "gt_cy": float(gt_xy_interp[idx, 1]),
                "gt_area": float(gt_area[idx]),
            })

    return true_traj, observations, pred_centered, gt_centered, output_path


def run_track_ukf(model, device, track, track_index, frame_ids, gt_components, image_size, seq_root, args):
    track_txt = seq_root / "tracks" / f"track_{track_index:03d}.txt"
    true_traj, observations, pred_xy, gt_xy, track_txt = build_single_track_inputs(
        track,
        frame_ids,
        gt_components,
        image_size,
        track_txt,
    )

    if len(true_traj) < 3:
        return None

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else min(getattr(model, "seq_len", 50), max(len(true_traj) // 3, 2))
    std_est = run_standard_ukf(true_traj, observations, DT)
    adapt_est, pred_turn = run_adaptive_ukf(
        true_traj,
        observations,
        model,
        device,
        DT,
        warmup_steps=warmup_steps,
        refresh_period=args.refresh_period,
        turn_smoothing=args.turn_smoothing,
    )

    raw_error = trajectory_rmse(true_traj, np.column_stack([pred_xy, np.zeros_like(pred_xy)]))
    std_error = trajectory_rmse(true_traj[1:1 + len(std_est)], std_est)
    adapt_error = trajectory_rmse(true_traj[1:1 + len(adapt_est)], adapt_est)
    return {
        "track_index": track_index,
        "length": len(track.detections),
        "start_frame": track.detections[0].frame_idx,
        "end_frame": track.detections[-1].frame_idx,
        "track_txt": track_txt,
        "raw_rmse": raw_error,
        "standard_rmse": std_error,
        "adaptive_rmse": adapt_error,
        "improvement": std_error - adapt_error,
        "gt_xy": gt_xy,
        "pred_xy": pred_xy,
        "std_est": std_est,
        "adapt_est": adapt_est,
        "pred_turn": pred_turn,
    }


def plot_all_tracks(results, out_path, sequence):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for result in results:
        axes[0].plot(result["adapt_est"][:, 0], result["adapt_est"][:, 1], linewidth=1.2, alpha=0.85)
        axes[0].scatter(result["pred_xy"][:, 0], result["pred_xy"][:, 1], s=4, alpha=0.25)
    axes[0].set_title(f"{sequence}: all associated detection tracks")
    axes[0].set_xlabel("X (px, centered)")
    axes[0].set_ylabel("Y (px, centered)")
    axes[0].axis("equal")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    labels = [str(result["track_index"]) for result in results]
    values = [result["adaptive_rmse"] for result in results]
    axes[1].bar(labels, values)
    axes[1].set_title("Adaptive UKF RMSE by track")
    axes[1].set_xlabel("Track index")
    axes[1].set_ylabel("RMSE (px)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def centered_to_image(points, image_size):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 2 and points.shape[1] > 2:
        points = points[:, :2]
    center = np.array([image_size[0] / 2.0, image_size[1] / 2.0], dtype=np.float64)
    return points + center


def color_for_track(track_index):
    palette = [
        (255, 64, 64),
        (64, 180, 255),
        (80, 220, 120),
        (255, 190, 64),
        (200, 120, 255),
        (255, 120, 190),
        (120, 255, 230),
        (230, 230, 80),
    ]
    return palette[track_index % len(palette)]


def draw_polyline(draw, points, fill, width=2):
    if len(points) < 2:
        return
    draw.line([tuple(map(float, point)) for point in points], fill=fill, width=width)


def visible_detection_box(det, image_size, min_size=14):
    x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
    cx, cy = float(det.cx), float(det.cy)
    half = min_size / 2.0
    if x2 - x1 < min_size:
        x1, x2 = cx - half, cx + half
    if y2 - y1 < min_size:
        y1, y2 = cy - half, cy + half
    width, height = image_size
    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(width - 1), x2),
        min(float(height - 1), y2),
    ]


def resolve_gt_mask_dir(result_root, sequence):
    gt_dir = Path(result_root) / "序列真值掩码" / sequence
    if gt_dir.exists():
        return gt_dir
    alt_dir = Path(result_root) / "序列真值掩码" / sequence.replace("-", "_")
    if alt_dir.exists():
        return alt_dir
    raise FileNotFoundError(f"Ground-truth mask directory not found for sequence: {sequence}")


def render_tracking_video(frame_ids, frame_detections, results, result_root, sequence, frame_dir, video_out, fps=12):
    data_dir = Path(result_root) / "data" / sequence
    gt_dir = resolve_gt_mask_dir(result_root, sequence)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_out = Path(video_out)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    first_image = Image.open(data_dir / f"{frame_ids[0]}.png").convert("RGB")
    image_size = first_image.size
    track_lookup = {result["track_index"]: result for result in results}

    progress = tqdm(frame_ids, desc="render frames", unit="frame", dynamic_ncols=True)
    for frame_idx, frame_id in enumerate(progress):
        left_image = Image.open(data_dir / f"{frame_id}.png").convert("RGB")
        gt_path = gt_dir / f"{frame_id}.png"
        if gt_path.exists():
            right_image = Image.open(gt_path).convert("L").convert("RGB")
        else:
            right_image = Image.new("RGB", image_size, (0, 0, 0))
        if right_image.size != image_size:
            right_image = right_image.resize(image_size, Image.Resampling.NEAREST)

        left_draw = ImageDraw.Draw(left_image)
        right_draw = ImageDraw.Draw(right_image)

        for det in frame_detections[frame_idx]:
            det_box = visible_detection_box(det, image_size)
            left_draw.rectangle(det_box, outline=(170, 170, 170), width=1)
            r = 2
            left_draw.ellipse([det.cx - r, det.cy - r, det.cx + r, det.cy + r], fill=(230, 230, 230))
            right_draw.rectangle(det_box, outline=(255, 0, 0), width=3)
            right_draw.line([det.cx - 5, det.cy, det.cx + 5, det.cy], fill=(255, 0, 0), width=1)
            right_draw.line([det.cx, det.cy - 5, det.cx, det.cy + 5], fill=(255, 0, 0), width=1)

        for track_index, result in track_lookup.items():
            if frame_idx < result["start_frame"] or frame_idx > result["end_frame"]:
                continue
            local_idx = frame_idx - result["start_frame"]
            color = color_for_track(track_index)
            pred_points = centered_to_image(result["pred_xy"][:local_idx + 1], image_size)
            adapt_points = centered_to_image(result["adapt_est"][:min(local_idx + 1, len(result["adapt_est"]))], image_size)
            gt_points = centered_to_image(result["gt_xy"][:local_idx + 1], image_size)

            draw_polyline(left_draw, gt_points, fill=(40, 255, 40), width=1)
            draw_polyline(left_draw, pred_points, fill=color, width=2)
            draw_polyline(left_draw, adapt_points, fill=(255, 255, 255), width=2)

            if len(pred_points) > 0:
                x, y = pred_points[-1]
                left_draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color, width=2)
                left_draw.text((x + 5, y + 5), f"T{track_index}", fill=color)
            if len(adapt_points) > 0:
                x, y = adapt_points[-1]
                left_draw.rectangle([x - 3, y - 3, x + 3, y + 3], outline=(255, 255, 255), width=2)

        left_draw.rectangle([8, 8, 276, 58], fill=(0, 0, 0))
        left_draw.text((14, 14), f"{sequence} | tracking", fill=(255, 255, 255))
        left_draw.text((14, 34), f"frame {frame_id} | tracks {len(results)}", fill=(255, 255, 255))
        right_draw.rectangle([8, 8, 300, 58], fill=(0, 0, 0))
        right_draw.text((14, 14), "GT mask + all detections", fill=(255, 255, 255))
        right_draw.text((14, 34), f"red boxes: {len(frame_detections[frame_idx])}", fill=(255, 64, 64))

        combined_size = (image_size[0] * 2, image_size[1] + image_size[1] % 2)
        combined = Image.new("RGB", combined_size, (0, 0, 0))
        combined.paste(left_image, (0, 0))
        combined.paste(right_image, (image_size[0], 0))
        combined.save(frame_dir / f"{frame_idx:06d}.png")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None and video_out.suffix.lower() == ".mp4":
        print(f"Encoding mp4 with ffmpeg: {video_out}")
        command = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(video_out),
        ]
        subprocess.run(command, check=True)
        return video_out

    gif_out = video_out.with_suffix(".gif") if video_out.suffix.lower() != ".gif" else video_out
    print(f"ffmpeg not found; encoding gif fallback: {gif_out}")
    frames = [Image.open(frame_dir / f"{idx:06d}.png").convert("P", palette=Image.ADAPTIVE) for idx in range(len(frame_ids))]
    frames[0].save(
        gif_out,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(1000 / fps), 1),
        loop=0,
        optimize=True,
    )
    return gif_out


def run_detection_ukf(args):
    frame_ids, frame_detections, gt_components, seq_root = generate_multitarget_txt(
        args.result_root,
        args.sequence,
        args.output_root,
        min_area=args.min_area,
        keep_frame_txt=args.keep_frame_txt,
    )

    tracks = link_tracks(
        frame_detections,
        max_link_distance=args.max_link_distance,
        max_gap=args.max_gap,
        velocity_weight=args.velocity_weight,
        gap_distance_growth=args.gap_distance_growth,
        velocity_gate_weight=args.velocity_gate_weight,
        merge_gap=args.merge_gap,
        merge_distance=args.merge_distance,
    )
    if not tracks:
        raise RuntimeError("No detection tracks were created")
    tracks = [track for track in tracks if len(track.detections) >= args.min_track_length]
    if args.max_tracks is not None:
        tracks = tracks[:args.max_tracks]
    print(f"qualified tracks: {len(tracks)}")

    first_image = next((Path(args.result_root) / "data" / args.sequence).glob("*.png"))
    image_size = Image.open(first_image).size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_turn_rate_model(args.checkpoint, device)

    results = []
    progress = tqdm(list(enumerate(tracks)), desc="track ukf", unit="track", dynamic_ncols=True)
    for track_index, track in progress:
        result = run_track_ukf(model, device, track, track_index, frame_ids, gt_components, image_size, seq_root, args)
        if result is not None:
            results.append(result)

    summary_path = seq_root / f"{args.sequence}_tracks_summary.txt"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "track_index",
            "length",
            "start_frame",
            "end_frame",
            "track_txt",
            "raw_rmse",
            "standard_rmse",
            "adaptive_rmse",
            "improvement",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=" ")
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})

    print(f"tracked targets: {len(results)}")
    print(f"tracks summary txt: {summary_path}")
    if results:
        print(f"mean raw detection RMSE: {np.mean([r['raw_rmse'] for r in results]):.3f} px")
        print(f"mean Standard UKF RMSE: {np.mean([r['standard_rmse'] for r in results]):.3f} px")
        print(f"mean Adaptive UKF RMSE: {np.mean([r['adaptive_rmse'] for r in results]):.3f} px")
        plot_all_tracks(results, args.out, args.sequence)
        print(f"Saved figure: {args.out}")
        if args.show:
            frame_dir = args.show_frame_dir or (seq_root / "visualization_frames")
            video_out = args.video_out or (PROJECT_ROOT / "outputs" / "figures" / "detection_tracking" / f"{args.sequence}_tracking.mp4")
            rendered_video = render_tracking_video(
                frame_ids,
                frame_detections,
                results,
                args.result_root,
                args.sequence,
                frame_dir,
                video_out,
                fps=args.fps,
            )
            print(f"Saved visualization frames: {frame_dir}")
            print(f"Saved visualization video: {rendered_video}")


def main():
    parser = argparse.ArgumentParser(description="Build a target track from multi-target detections and run Adaptive UKF.")
    parser.add_argument("--result-root", type=Path, default=Path("detection/result-0705"))
    parser.add_argument("--sequence", type=str, default="WestAfrica-9_49")
    parser.add_argument("--output-root", type=Path, default=Path("detection/result-0705/detection_txt_multi"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "ST_Transformer_response_feature_transformer_0705_iter_60.pth",
    )
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-link-distance", type=float, default=35.0)
    parser.add_argument("--max-gap", type=int, default=12)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--gap-distance-growth", type=float, default=8.0)
    parser.add_argument("--velocity-gate-weight", type=float, default=0.8)
    parser.add_argument("--merge-gap", type=int, default=30)
    parser.add_argument("--merge-distance", type=float, default=45.0)
    parser.add_argument("--min-track-length", type=int, default=20)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--keep-frame-txt", action="store_true", help="Also write one txt file per frame")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--refresh-period", type=int, default=1)
    parser.add_argument("--turn-smoothing", type=float, default=0.25)
    parser.add_argument("--show", action="store_true", help="Render detections and tracking results on original images")
    parser.add_argument("--video-out", type=Path, default=None, help="Output video path. Uses GIF fallback when ffmpeg is absent")
    parser.add_argument("--show-frame-dir", type=Path, default=None, help="Directory for rendered visualization frames")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "detection_tracking" / "detection_ukf_WestAfrica-9_49_all_tracks.png",
    )
    args = parser.parse_args()
    run_detection_ukf(args)


if __name__ == "__main__":
    main()

# conda run -n deepmtt python ST_Transformer_Tracking/scripts/run_detection_sequence_ukf.py \
#   --result-root detection/result-0705 \
#   --sequence WestAfrica-9_49 \
#   --output-root detection/result-0705/detection_txt_multi \
#   --checkpoint ST_Transformer_Tracking/checkpoints/pytorch/Save_Model/ST_Transformer_response_feature_transformer_0705_iter_60.pth \
#   --min-area 1 \
#   --max-link-distance 35 \
#   --velocity-weight 0.5 \
#   --max-gap 5 \
#   --min-track-length 20 \
#   --out ST_Transformer_Tracking/outputs/figures/detection_ukf_WestAfrica-9_49_all_tracks.png
