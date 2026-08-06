#!/usr/bin/env python3
"""Render a deterministic head-on collision rollout comparison as MP4."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from myriad.data_billiards import render_billiard_frame
from scripts.myriad_eval.compare_billiard_models import (
    FRAME_SIZE,
    GIVEN_STEPS,
    N_BALLS,
    RADIUS_NORM,
    fixed_scene,
    load_model,
    seed_everything,
)


def predict(kind: str, checkpoint: Path, image: torch.Tensor, truth: torch.Tensor,
            ts: torch.Tensor, device: torch.device) -> np.ndarray:
    model, _, _ = load_model(kind, checkpoint, device)
    image = image.to(device)
    truth = truth.to(device)
    ts = ts.to(device)
    given_pos = truth[:, :GIVEN_STEPS].reshape(1, GIVEN_STEPS * N_BALLS, 2)
    camera_static = torch.ones(1, dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        d_img = model.embed_image(image)
        seed_everything(20260805)
        prediction = model.predict_simulate(
            n_traj=N_BALLS,
            ts=ts,
            given_pos=given_pos,
            camera_static=camera_static,
            d_img=d_img,
            verbose=False,
        )
    del model, d_img
    torch.cuda.empty_cache()
    return prediction.float().cpu().numpy()[0]


def panel(positions: np.ndarray, label: str, frame_index: int,
          collision: bool, trail: list[np.ndarray]) -> np.ndarray:
    border = [35, 35, 35, 35]
    radius = int(round(RADIUS_NORM * FRAME_SIZE))
    frame = render_billiard_frame(
        positions * FRAME_SIZE,
        [radius] * N_BALLS,
        FRAME_SIZE,
        border,
        antialiasing=True,
    )
    frame = np.ascontiguousarray(frame)
    for history in trail:
        for ball in (0, 1):
            points = (history[:, ball] * FRAME_SIZE).round().astype(np.int32)
            color = (0, 220, 0) if ball == 0 else (0, 80, 255)
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(frame, tuple(start), tuple(end), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (FRAME_SIZE - 1, 42), (20, 20, 20), -1)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    if collision:
        cv2.circle(frame, (int(positions[0, 0] * FRAME_SIZE), int(positions[0, 1] * FRAME_SIZE)),
                   radius + 8, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "COLLISION", (FRAME_SIZE - 190, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    if not (args.dino_path / "config.json").is_file():
        parser.error(f"Missing DINO snapshot: {args.dino_path}")
    os.environ["MYRIAD_DINO_PATH"] = str(args.dino_path)
    device = torch.device("cuda:0")
    torch.cuda.init()
    image, truth, ts, collisions = fixed_scene()
    temporal = predict("long_history_bias", args.checkpoint, image, truth, ts, device)
    baseline = predict("original", Path("checkpoints/myriad_billiard.pt"), image, truth, ts, device)
    truth_np = truth.numpy()[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (FRAME_SIZE * 3, FRAME_SIZE),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")
    histories = [[], [], []]
    for t in range(truth_np.shape[0]):
        histories[0].append(truth_np[max(0, t - 15):t + 1])
        histories[1].append(temporal[max(0, t - 15):t + 1])
        histories[2].append(baseline[max(0, t - 15):t + 1])
        collision = bool(collisions[t]) if t < len(collisions) else False
        frames = [
            panel(truth_np[t], "GROUND TRUTH", t, collision, [histories[0][-1]]),
            panel(temporal[t], "LONG HISTORY 1500", t, collision, [histories[1][-1]]),
            panel(baseline[t], "NO-BIAS BASELINE", t, collision, [histories[2][-1]]),
        ]
        canvas = np.concatenate(frames, axis=1)
        cv2.putText(canvas, f"t={t:02d}  time={t * 0.01:.2f}s", (12, FRAME_SIZE - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    print(f"Saved {args.output} ({truth_np.shape[0]} frames, {FRAME_SIZE * 3}x{FRAME_SIZE})")


if __name__ == "__main__":
    main()
