#!/usr/bin/env python3
"""Compare the original and physics-bias MYRIAD billiards models on one fixed scene."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import billiards
import numpy as np
import torch
from einops import rearrange
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from myriad.data_billiards import render_billiard_frame, simulate_billiard_game
import myriad.model as myriad_model
from myriad.model import (
    MyriadStepByStep_Large_Billiard,
    MyriadStepByStep_Large_Billiard_PhysicsBias,
    MyriadStepByStep_Large_Billiard_LongHistoryBias,
)

# The repository's unconditional compiled FlexAttention kernel exceeds the
# per-block resource limit on RTX 5090 for this prefill shape. The physics path
# already uses eager FlexAttention, so use that same backend for both models.
myriad_model.flex_attention_compiled = myriad_model.flex_attention


SEED = 20260805
FRAME_SIZE = 512
DT = 0.01
DURATION = 0.5
N_BALLS = 16
RADIUS_NORM = 0.0333
GIVEN_STEPS = 2
DEFAULT_DINO_PATH = Path(
    "/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
        "repeats": len(values),
    }


def timed_cuda(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def fixed_scene() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """Create a deterministic scene where ball 0 collides head-on with ball 1."""
    border = [35, 35, 35, 35]
    radius_px = RADIUS_NORM * FRAME_SIZE
    table = billiards.Billiard(
        obstacles=[
            billiards.InfiniteWall((0, border[0]), (FRAME_SIZE - border[1], border[0])),
            billiards.InfiniteWall(
                (FRAME_SIZE - border[1], border[0]),
                (FRAME_SIZE - border[1], FRAME_SIZE - border[2]),
            ),
            billiards.InfiniteWall(
                (FRAME_SIZE - border[1], FRAME_SIZE - border[2]),
                (border[3], FRAME_SIZE - border[2]),
            ),
            billiards.InfiniteWall((border[3], FRAME_SIZE - border[2]), (border[3], border[0])),
        ]
    )

    # The first collision occurs at roughly 0.09 s. Remaining balls are static
    # and spaced far enough apart to keep scene generation independent of RNG.
    positions_norm = [
        (0.25, 0.50), (0.40, 0.50),
        (0.16, 0.18), (0.32, 0.18), (0.48, 0.18), (0.64, 0.18), (0.80, 0.18),
        (0.16, 0.34), (0.32, 0.34), (0.48, 0.34), (0.64, 0.34), (0.80, 0.34),
        (0.18, 0.78), (0.38, 0.78), (0.58, 0.78), (0.78, 0.78),
    ]
    for index, position in enumerate(positions_norm):
        velocity = (480.0, 0.0) if index == 0 else (0.0, 0.0)
        table.add_ball(tuple(v * FRAME_SIZE for v in position), velocity, radius_px)

    _, positions, _, collisions = simulate_billiard_game(table, DURATION, DT)
    positions_np = np.asarray(positions, dtype=np.float32)
    image_np = render_billiard_frame(positions_np[0], [radius_px] * N_BALLS, FRAME_SIZE, border)
    image = rearrange(torch.from_numpy(image_np).float().div(127.5).sub(1), "h w c -> 1 c h w")
    # Match training: T input states are pos[:-1], with the final simulated state
    # serving only as the target of the last training flow.
    truth = torch.from_numpy(positions_np[:-1] / FRAME_SIZE).unsqueeze(0)
    ts = torch.arange(truth.shape[1], dtype=torch.float32).unsqueeze(0)
    return image, truth, ts, collisions


def load_model(
    kind: str,
    checkpoint_path: Path,
    device: torch.device,
    physics_max_abs: float | None = None,
    physics_strength: float | None = None,
    physics_kinematics_mode: str = "long",
    collision_long_history_distance: float = 0.04,
    collision_long_history_temperature: float = 0.008,
):
    if kind.startswith("physics_bias"):
        constructor = MyriadStepByStep_Large_Billiard_PhysicsBias
    elif kind.startswith("long_history_bias"):
        constructor = MyriadStepByStep_Large_Billiard_LongHistoryBias
    else:
        constructor = MyriadStepByStep_Large_Billiard
    start = time.perf_counter()
    model = constructor()
    checkpoint = torch.load(checkpoint_path, weights_only=False, mmap=True, map_location="cpu")
    # Transformers 5.x wraps the DINOv3 encoder in one additional ``model``
    # module compared with the version used to create these checkpoints.
    state_dict = {
        key.replace(
            "image_embedder.model.model.layer.",
            "image_embedder.model.model.model.layer.",
        ): value
        for key, value in checkpoint["model"].items()
    }
    physics_input_key = "transformer.physics_bias_generator.mlp.0.weight"
    if physics_input_key in state_dict:
        source = state_dict[physics_input_key]
        target = model.state_dict()[physics_input_key]
        if source.shape[:-1] == target.shape[:-1] and source.shape[-1] < target.shape[-1]:
            expanded = source.new_zeros(target.shape)
            expanded[..., :source.shape[-1]] = source
            state_dict[physics_input_key] = expanded
    model.load_state_dict(state_dict, strict=True)
    if kind.startswith("physics_bias") and physics_max_abs is not None:
        model.transformer.physics_bias_generator.max_abs_bias = physics_max_abs
    if kind.startswith("physics_bias") and physics_strength is not None:
        with torch.no_grad():
            model.transformer.physics_bias_generator.layer_head_scales.mul_(physics_strength)
    if kind.startswith("physics_bias"):
        model.transformer.physics_bias_generator.set_kinematics_mode(
            physics_kinematics_mode,
            collision_distance=collision_long_history_distance,
            collision_temperature=collision_long_history_temperature,
        )
    if kind == "physics_bias_disabled":
        model.transformer.use_physics_bias = False
    if kind == "long_history_bias_disabled":
        model.transformer.use_long_history_bias = False
    step = int(checkpoint.get("step", -1))
    del checkpoint
    model.requires_grad_(False).eval().to(device)
    torch.cuda.synchronize()
    return model, step, time.perf_counter() - start


def evaluate_one(
    kind: str,
    checkpoint_path: Path,
    image_cpu: torch.Tensor,
    truth_cpu: torch.Tensor,
    ts_cpu: torch.Tensor,
    device: torch.device,
    embed_repeats: int,
    rollout_warmup: int,
    rollout_repeats: int,
    physics_max_abs: float | None = None,
    physics_strength: float | None = None,
    oracle_history: bool = False,
    physics_kinematics_mode: str = "long",
    collision_long_history_distance: float = 0.04,
    collision_long_history_temperature: float = 0.008,
) -> tuple[dict, np.ndarray]:
    torch.cuda.empty_cache()
    model, checkpoint_step, load_seconds = load_model(
        kind,
        checkpoint_path,
        device,
        physics_max_abs=physics_max_abs,
        physics_strength=physics_strength,
        physics_kinematics_mode=physics_kinematics_mode,
        collision_long_history_distance=collision_long_history_distance,
        collision_long_history_temperature=collision_long_history_temperature,
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    image = image_cpu.to(device)
    truth = truth_cpu.to(device)
    ts = ts_cpu.to(device)
    given_pos = truth[:, :GIVEN_STEPS].reshape(1, GIVEN_STEPS * N_BALLS, 2)
    camera_static = torch.ones(1, dtype=torch.bool, device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embed_times = timed_cuda(lambda: model.embed_image(image), warmup=2, repeats=embed_repeats)
        d_img = {key: value.clone() if torch.is_tensor(value) else value for key, value in model.embed_image(image).items()}

        holder: dict[str, torch.Tensor] = {}

        def rollout() -> None:
            holder["prediction"] = model.predict_simulate(
                n_traj=N_BALLS,
                ts=ts,
                given_pos=given_pos,
                camera_static=camera_static,
                d_img=d_img,
                verbose=False,
            )

        seed_everything(SEED)
        # Accuracy prediction uses the same RF sampling seed for both models.
        rollout()
        prediction = holder["prediction"].float()
        oracle_prediction = None
        if oracle_history:
            seed_everything(SEED)
            oracle_prediction = model.predict_simulate(
                n_traj=N_BALLS,
                ts=ts,
                given_pos=given_pos,
                camera_static=camera_static,
                d_img=d_img,
                verbose=False,
                teacher_forcing_pos=truth,
            ).float()
        seed_everything(SEED + 1)
        rollout_times = timed_cuda(rollout, warmup=rollout_warmup, repeats=rollout_repeats)

    truth_region = truth[:, GIVEN_STEPS:]

    def accuracy_metrics(candidate: torch.Tensor) -> dict:
        epe_px = torch.linalg.vector_norm(
            candidate[:, GIVEN_STEPS:] - truth_region, dim=-1
        ) * FRAME_SIZE
        all_errors = epe_px.flatten()
        return {
            "mean_epe_px": float(all_errors.mean()),
            "median_epe_px": float(all_errors.median()),
            "p95_epe_px": float(torch.quantile(all_errors, 0.95)),
            "final_step_mean_epe_px": float(epe_px[:, -1].mean()),
            "moving_ball_0_mean_epe_px": float(epe_px[:, :, 0].mean()),
            "collision_target_ball_1_mean_epe_px": float(epe_px[:, :, 1].mean()),
            "per_timestep_mean_epe_px": epe_px.mean(dim=(0, 2)).cpu().tolist(),
        }

    accuracy = accuracy_metrics(prediction)
    rollout_summary = summarize_ms(rollout_times)
    predicted_tokens = (truth.shape[1] - GIVEN_STEPS) * N_BALLS
    rollout_summary["predicted_motion_tokens"] = predicted_tokens
    rollout_summary["motion_tokens_per_second"] = predicted_tokens / (float(rollout_summary["mean_ms"]) / 1000)

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameters": params,
        "accuracy": accuracy,
        "efficiency": {
            "model_load_seconds": load_seconds,
            "image_embedding": summarize_ms(embed_times),
            "full_48_step_rollout": rollout_summary,
            "peak_allocated_vram_gib_including_model": torch.cuda.max_memory_allocated(device.index) / 1024**3,
        },
    }
    if oracle_prediction is not None:
        result["oracle_history_accuracy"] = accuracy_metrics(oracle_prediction)
    prediction_np = prediction.cpu().numpy()
    del model, image, truth, ts, given_pos, d_img, prediction, oracle_prediction, holder
    gc.collect()
    torch.cuda.empty_cache()
    return result, prediction_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dino-path",
        type=Path,
        default=Path(os.environ.get("MYRIAD_DINO_PATH", DEFAULT_DINO_PATH)),
        help="Local DINOv3 snapshot; avoids access to the gated Hugging Face repository.",
    )
    parser.add_argument("--embed-repeats", type=int, default=10)
    parser.add_argument("--rollout-warmup", type=int, default=0)
    parser.add_argument("--rollout-repeats", type=int, default=2)
    parser.add_argument(
        "--physics-max-abs",
        type=float,
        default=None,
        help="Override the checkpoint model's runtime physics-bias bound.",
    )
    parser.add_argument(
        "--physics-strength",
        type=float,
        default=None,
        help="Multiply all learned layer/head physics scales at evaluation time.",
    )
    parser.add_argument(
        "--oracle-history",
        action="store_true",
        help="Also measure predictions while feeding ground-truth history.",
    )
    parser.add_argument(
        "--physics-kinematics-mode",
        choices=("long", "short", "collision-gated", "collision-smooth"),
        default="long",
        help="Choose when the physics relation encoder uses long-history kinematics.",
    )
    parser.add_argument(
        "--collision-long-history-distance",
        type=float,
        default=0.04,
        help="Normalized absolute surface-distance window for collision-gated mode.",
    )
    parser.add_argument(
        "--collision-long-history-temperature",
        type=float,
        default=0.008,
        help="Sigmoid transition width for collision-smooth mode.",
    )
    args = parser.parse_args()

    if not (args.dino_path / "config.json").is_file():
        parser.error(f"DINOv3 snapshot is missing config.json: {args.dino_path}")
    os.environ["MYRIAD_DINO_PATH"] = str(args.dino_path)

    seed_everything(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.init()
    image, truth, ts, collisions = fixed_scene()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_u8 = rearrange(((image[0] + 1) * 127.5).round().byte(), "c h w -> h w c").numpy()
    Image.fromarray(image_u8).save(args.output_dir / "fixed_scene.png")

    torch.cuda.reset_peak_memory_stats(device.index)
    original, original_prediction = evaluate_one(
        "original", args.original, image, truth, ts, device,
        args.embed_repeats, args.rollout_warmup, args.rollout_repeats,
        oracle_history=args.oracle_history,
    )
    torch.cuda.reset_peak_memory_stats(device.index)
    physics, physics_prediction = evaluate_one(
        "physics_bias", args.physics, image, truth, ts, device,
        args.embed_repeats, args.rollout_warmup, args.rollout_repeats,
        physics_max_abs=args.physics_max_abs,
        physics_strength=args.physics_strength,
        physics_kinematics_mode=args.physics_kinematics_mode,
        collision_long_history_distance=args.collision_long_history_distance,
        collision_long_history_temperature=args.collision_long_history_temperature,
        oracle_history=args.oracle_history,
    )
    torch.cuda.reset_peak_memory_stats(device.index)
    physics_disabled, physics_disabled_prediction = evaluate_one(
        "physics_bias_disabled", args.physics, image, truth, ts, device,
        args.embed_repeats, args.rollout_warmup, args.rollout_repeats,
        physics_kinematics_mode=args.physics_kinematics_mode,
        collision_long_history_distance=args.collision_long_history_distance,
        collision_long_history_temperature=args.collision_long_history_temperature,
        oracle_history=args.oracle_history,
    )

    original_epe = original["accuracy"]["mean_epe_px"]
    physics_epe = physics["accuracy"]["mean_epe_px"]
    original_ms = original["efficiency"]["full_48_step_rollout"]["mean_ms"]
    physics_ms = physics["efficiency"]["full_48_step_rollout"]["mean_ms"]
    comparison = {
        "mean_epe_change_px": physics_epe - original_epe,
        "mean_epe_change_percent": (physics_epe / original_epe - 1) * 100,
        "rollout_latency_change_ms": physics_ms - original_ms,
        "rollout_latency_change_percent": (physics_ms / original_ms - 1) * 100,
        "rollout_speed_ratio_original_over_physics": original_ms / physics_ms,
        "extra_parameters": physics["parameters"] - original["parameters"],
    }
    metrics = {
        "reproducibility": {
            "seed": SEED,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "autocast": "bfloat16",
            "compile": False,
        },
        "scenario": {
            "frame_size": FRAME_SIZE,
            "balls": N_BALLS,
            "radius_normalized": RADIUS_NORM,
            "dt_seconds": DT,
            "duration_seconds": DURATION,
            "trajectory_steps": int(truth.shape[1]),
            "given_steps": GIVEN_STEPS,
            "predicted_steps": int(truth.shape[1] - GIVEN_STEPS),
            "first_collision_step": next((i for i, value in enumerate(collisions) if value), None),
            "accuracy_reference": "deterministic rigid-body billiards simulator ground truth",
        },
        "original": original,
        "physics_bias": physics,
        "physics_bias_disabled": physics_disabled,
        "comparison_physics_minus_original": comparison,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        truth=truth.numpy(),
        original=original_prediction,
        physics_bias=physics_prediction,
        physics_bias_disabled=physics_disabled_prediction,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
