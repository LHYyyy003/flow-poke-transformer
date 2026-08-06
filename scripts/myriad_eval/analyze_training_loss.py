#!/usr/bin/env python3
"""Summarize TensorBoard training scalars and checkpoint parameter movement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar_values(accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = accumulator.Scalars(tag)
    return (
        np.asarray([event.step for event in events], dtype=np.int64),
        np.asarray([event.value for event in events], dtype=np.float64),
    )


def series_summary(steps: np.ndarray, values: np.ndarray, window: int) -> dict:
    first = values[:window]
    last = values[-window:]
    slope = float(np.polyfit(steps, values, 1)[0])
    moving = np.convolve(values, np.ones(window) / window, mode="valid")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "coefficient_of_variation": float(values.std() / max(abs(values.mean()), 1e-12)),
        "first_window_mean": float(first.mean()),
        "last_window_mean": float(last.mean()),
        "first_to_last_change_percent": float((last.mean() / first.mean() - 1.0) * 100.0),
        "linear_slope_per_1000_steps": slope * 1000.0,
        "moving_average_min": float(moving.min()),
        "moving_average_min_step": int(steps[moving.argmin() + window - 1]),
        "moving_average_max": float(moving.max()),
        "moving_average_max_step": int(steps[moving.argmax() + window - 1]),
    }


def checkpoint_deltas(start_path: Path, end_path: Path) -> dict:
    start = torch.load(start_path, map_location="cpu", mmap=True, weights_only=False)["model"]
    end = torch.load(end_path, map_location="cpu", mmap=True, weights_only=False)["model"]
    result = {}
    for key, start_value in start.items():
        if "physics_bias_generator" not in key:
            continue
        end_value = end[key]
        start_float = start_value.float()
        delta = end_value.float() - start_float
        norm = start_float.norm().clamp_min(1e-12)
        result[key] = {
            "start_norm": float(norm),
            "delta_norm": float(delta.norm()),
            "relative_delta": float(delta.norm() / norm),
            "max_abs_delta": float(delta.abs().max()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--end-checkpoint", type=Path)
    args = parser.parse_args()
    if args.window < 2:
        parser.error("--window must be at least 2")
    if bool(args.start_checkpoint) != bool(args.end_checkpoint):
        parser.error("Provide both --start-checkpoint and --end-checkpoint")

    accumulator = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = [tag for tag in accumulator.Tags()["scalars"] if tag.startswith("train/")]
    if "train/loss" not in tags:
        parser.error("TensorBoard events do not contain train/loss")
    steps, loss = scalar_values(accumulator, "train/loss")
    if loss.size < args.window * 2:
        parser.error(f"Need at least {args.window * 2} loss values, found {loss.size}")

    series = {}
    correlations = {}
    for tag in tags:
        tag_steps, values = scalar_values(accumulator, tag)
        if not np.array_equal(tag_steps, steps):
            continue
        name = tag.removeprefix("train/")
        series[name] = series_summary(steps, values, args.window)
        if values.std() > 0:
            correlations[name] = float(np.corrcoef(loss, values)[0, 1])

    payload = {
        "events": str(args.events),
        "step_range": [int(steps[0]), int(steps[-1])],
        "window": args.window,
        "series": series,
        "loss_correlations": correlations,
    }
    if args.start_checkpoint:
        payload["checkpoint_parameter_deltas"] = checkpoint_deltas(
            args.start_checkpoint, args.end_checkpoint
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
