#!/usr/bin/env python3
"""Compare physics-bias training losses on fixed data with fixed RF noise."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from myriad.data_billiards import BilliardSimDataset, dict_collation_fn
from scripts.myriad_eval.compare_billiard_models import load_model, seed_everything
from train import myriad_make_train_fns


DATA_SEED = 20260806
RF_SEED = 20260860
DEFAULT_DINO_PATH = Path(
    "/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master"
)


def make_fixed_batches(count: int, batch_size: int) -> list[dict[str, torch.Tensor]]:
    random.seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)
    dataset = BilliardSimDataset(
        nr_balls=16,
        frame_size=512,
        duration=0.5,
        dt=0.01,
        collision_loss_weight=3.0,
        collision_window_steps=10,
    )
    iterator = iter(dataset)
    return [dict_collation_fn([next(iterator) for _ in range(batch_size)]) for _ in range(count)]


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "standard_error": float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate(
    name: str,
    checkpoint: Path,
    bias_enabled: bool,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    kind = "physics_bias" if bias_enabled else "physics_bias_disabled"
    print(f"Evaluating fixed validation loss: {name}", flush=True)
    model, step, _ = load_model(
        kind,
        checkpoint,
        device,
        physics_kinematics_mode="collision-smooth",
        collision_long_history_distance=0.04,
        collision_long_history_temperature=0.008,
    )
    init_caches, compute_step = myriad_make_train_fns(model, model, device, "cuda", False)
    block_mask = is_query = None
    l_poke = None
    per_batch: list[dict[str, float]] = []
    for index, cpu_batch in enumerate(batches):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in cpu_batch.items()
        }
        if block_mask is None:
            block_mask, is_query, l_poke = init_caches(batch)
        seed_everything(RF_SEED + index)
        with torch.inference_mode():
            loss, metrics = compute_step(
                batch,
                block_mask=block_mask,
                is_query=is_query,
                L_poke=l_poke,
                compute_metrics=True,
                flow_mask=batch.get("flow_loss_mask"),
            )
        row = {"loss": float(loss)} | {key: float(value) for key, value in metrics.items()}
        per_batch.append(row)
        print(f"  batch {index + 1}/{len(batches)} loss={row['loss']:.6f}", flush=True)
        del batch, loss, metrics

    keys = per_batch[0].keys()
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "physics_bias_enabled": bias_enabled,
        "per_batch": per_batch,
        "summary": {key: summarize([row[key] for row in per_batch]) for key in keys},
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def paired_difference(first: dict, second: dict) -> dict:
    result = {}
    for key in first["summary"]:
        first_values = np.asarray([row[key] for row in first["per_batch"]], dtype=np.float64)
        second_values = np.asarray([row[key] for row in second["per_batch"]], dtype=np.float64)
        differences = first_values - second_values
        result[key] = summarize(differences.tolist()) | {
            "relative_to_second_mean_percent": float(
                differences.mean() / max(abs(second_values.mean()), 1e-12) * 100.0
            )
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-500", type=Path, required=True)
    parser.add_argument("--smooth-1500", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO_PATH)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 2 or args.batch_size < 1:
        parser.error("Use at least two batches and a positive batch size")
    if not (args.dino_path / "config.json").is_file():
        parser.error(f"Missing local DINOv3 snapshot: {args.dino_path}")

    os.environ["MYRIAD_DINO_PATH"] = str(args.dino_path)
    batches = make_fixed_batches(args.batches, args.batch_size)
    device = torch.device("cuda:0")
    torch.cuda.init()
    variants = {}
    for name, checkpoint, enabled in (
        ("smooth_500_enabled", args.smooth_500, True),
        ("smooth_500_disabled", args.smooth_500, False),
        ("smooth_1500_enabled", args.smooth_1500, True),
        ("smooth_1500_disabled", args.smooth_1500, False),
    ):
        variants[name] = evaluate(name, checkpoint, enabled, batches, device)

    comparisons = {
        "smooth_500_enabled_minus_disabled": paired_difference(
            variants["smooth_500_enabled"], variants["smooth_500_disabled"]
        ),
        "smooth_1500_enabled_minus_disabled": paired_difference(
            variants["smooth_1500_enabled"], variants["smooth_1500_disabled"]
        ),
        "smooth_1500_enabled_minus_500_enabled": paired_difference(
            variants["smooth_1500_enabled"], variants["smooth_500_enabled"]
        ),
        "disabled_1500_minus_500": paired_difference(
            variants["smooth_1500_disabled"], variants["smooth_500_disabled"]
        ),
    }
    payload = {
        "protocol": {
            "data_seed": DATA_SEED,
            "rf_seed_base": RF_SEED,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "samples": args.batches * args.batch_size,
            "fixed_data": True,
            "fixed_rf_noise_per_batch": True,
            "collision_loss_weight": 3.0,
            "collision_window_steps": 10,
        },
        "variants": variants,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
