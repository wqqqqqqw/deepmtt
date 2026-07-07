#!/usr/bin/env python3
"""Convert sequence prediction masks into per-frame detection txt files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def mask_measurements(mask_path):
    mask = np.asarray(Image.open(mask_path).convert("L"))
    binary = mask > 0
    labeled, component_count = ndimage.label(binary)
    if component_count > 0:
        areas = ndimage.sum(binary, labeled, index=np.arange(1, component_count + 1))
        largest_label = int(np.argmax(areas) + 1)
        binary = labeled == largest_label

    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return {
            "detected": 0,
            "cx": np.nan,
            "cy": np.nan,
            "x1": np.nan,
            "y1": np.nan,
            "x2": np.nan,
            "y2": np.nan,
            "area": 0,
        }

    return {
        "detected": 1,
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max()),
        "y2": int(ys.max()),
        "area": int(len(xs)),
    }


def write_frame_txt(path, frame_id, pred, gt):
    with path.open("w", encoding="utf-8") as f:
        f.write("frame detected pred_cx pred_cy pred_x1 pred_y1 pred_x2 pred_y2 pred_area gt_detected gt_cx gt_cy gt_area\n")
        f.write(
            f"{frame_id} {pred['detected']} {pred['cx']:.6f} {pred['cy']:.6f} "
            f"{pred['x1']} {pred['y1']} {pred['x2']} {pred['y2']} {pred['area']} "
            f"{gt['detected']} {gt['cx']:.6f} {gt['cy']:.6f} {gt['area']}\n"
        )


def generate_txt(result_root, sequence, output_root):
    result_root = Path(result_root)
    output_root = Path(output_root)
    pred_dir = result_root / "predict" / sequence
    gt_dir = result_root / "序列真值掩码" / sequence
    if not gt_dir.exists():
        alt_name = sequence.replace("-", "_")
        alt_dir = result_root / "序列真值掩码" / alt_name
        if alt_dir.exists():
            gt_dir = alt_dir

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth mask directory not found: {gt_dir}")

    frame_dir = output_root / sequence / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / sequence / f"{sequence}_detections.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.png"))
    rows = []
    for pred_path in pred_files:
        frame_id = pred_path.stem
        gt_path = gt_dir / pred_path.name
        pred = mask_measurements(pred_path)
        gt = mask_measurements(gt_path) if gt_path.exists() else mask_measurements(pred_path)
        write_frame_txt(frame_dir / f"{frame_id}.txt", frame_id, pred, gt)
        rows.append({
            "frame": frame_id,
            "detected": pred["detected"],
            "pred_cx": pred["cx"],
            "pred_cy": pred["cy"],
            "pred_x1": pred["x1"],
            "pred_y1": pred["y1"],
            "pred_x2": pred["x2"],
            "pred_y2": pred["y2"],
            "pred_area": pred["area"],
            "gt_detected": gt["detected"],
            "gt_cx": gt["cx"],
            "gt_cy": gt["cy"],
            "gt_area": gt["area"],
        })

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=" ")
        writer.writeheader()
        writer.writerows(rows)

    detected_count = sum(row["detected"] for row in rows)
    print(f"sequence: {sequence}")
    print(f"frames: {len(rows)}")
    print(f"detected frames: {detected_count}")
    print(f"frame txt dir: {frame_dir}")
    print(f"summary txt: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate detection txt files from prediction masks.")
    parser.add_argument("--result-root", type=Path, default=Path("detection/result-0705"))
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("detection/result-0705/detection_txt"))
    args = parser.parse_args()
    generate_txt(args.result_root, args.sequence, args.output_root)


if __name__ == "__main__":
    main()
