#!/usr/bin/env python3
"""Evaluate baseline and non-physical long-history attention checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.myriad_eval.evaluate_fixed_validation_loss import (
    DATA_SEED,
    RF_SEED,
    evaluate as evaluate_fixed,
    make_fixed_batches,
    paired_difference,
)
from scripts.myriad_eval.evaluate_physics_multiscene import (
    COLLISION_WINDOW_STEPS,
    DT,
    DURATION,
    FRAME_SIZE,
    GIVEN_STEPS,
    SCENE_SPECS,
    SEED,
    build_scene,
    evaluate_variant,
)


DEFAULT_DINO_PATH = Path(
    "/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--long-history-500", type=Path, required=True)
    parser.add_argument("--long-history-1000", type=Path, required=True)
    parser.add_argument("--long-history-1500", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO_PATH)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    checkpoints = {
        "baseline": args.original,
        "long_history_500": args.long_history_500,
        "long_history_1000": args.long_history_1000,
        "long_history_1500": args.long_history_1500,
    }
    for path in checkpoints.values():
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")
    if not (args.dino_path / "config.json").is_file():
        parser.error(f"Missing DINOv3 snapshot: {args.dino_path}")

    os.environ["MYRIAD_DINO_PATH"] = str(args.dino_path)
    device = torch.device("cuda:0")
    torch.cuda.init()
    batches = make_fixed_batches(args.batches, args.batch_size)
    fixed = {
        "baseline": evaluate_fixed(
            "baseline", args.original, False, batches, device, kind="original"
        ),
        "long_history_500": evaluate_fixed(
            "long_history_500", args.long_history_500, True, batches, device,
            kind="long_history_bias",
        ),
        "long_history_1000": evaluate_fixed(
            "long_history_1000", args.long_history_1000, True, batches, device,
            kind="long_history_bias",
        ),
        "long_history_1500": evaluate_fixed(
            "long_history_1500", args.long_history_1500, True, batches, device,
            kind="long_history_bias",
        ),
        "long_history_1500_disabled": evaluate_fixed(
            "long_history_1500_disabled", args.long_history_1500, False, batches, device,
            kind="long_history_bias_disabled",
        ),
    }
    fixed_comparisons = {
        f"{name}_minus_baseline": paired_difference(result, fixed["baseline"])
        for name, result in fixed.items() if name != "baseline"
    }

    scenes = [build_scene(spec) for spec in SCENE_SPECS]
    rollout = {
        "baseline": evaluate_variant(
            "baseline", "original", args.original, "none", scenes, device
        ),
        "long_history_500": evaluate_variant(
            "long_history_500", "long_history_bias", args.long_history_500,
            "temporal-only", scenes, device,
        ),
        "long_history_1000": evaluate_variant(
            "long_history_1000", "long_history_bias", args.long_history_1000,
            "temporal-only", scenes, device,
        ),
        "long_history_1500": evaluate_variant(
            "long_history_1500", "long_history_bias", args.long_history_1500,
            "temporal-only", scenes, device,
        ),
        "long_history_1500_disabled": evaluate_variant(
            "long_history_1500_disabled", "long_history_bias_disabled",
            args.long_history_1500, "disabled", scenes, device,
        ),
    }
    payload = {
        "protocol": {
            "data_seed": DATA_SEED,
            "rf_seed_base": RF_SEED,
            "rollout_seed": SEED,
            "fixed_batches": args.batches,
            "fixed_batch_size": args.batch_size,
            "fixed_samples": args.batches * args.batch_size,
            "frame_size": FRAME_SIZE,
            "dt_seconds": DT,
            "duration_seconds": DURATION,
            "given_steps": GIVEN_STEPS,
            "collision_window_steps": COLLISION_WINDOW_STEPS,
            "physical_inputs": False,
        },
        "fixed_validation": fixed,
        "fixed_paired_comparisons": fixed_comparisons,
        "multiscene_rollout": rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
