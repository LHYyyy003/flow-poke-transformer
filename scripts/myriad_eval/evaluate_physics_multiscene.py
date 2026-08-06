#!/usr/bin/env python3
"""Evaluate billiards physics-bias checkpoints on deterministic scene types."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import billiards
import numpy as np
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from myriad.data_billiards import (
    expand_collision_window,
    render_billiard_frame,
    simulate_billiard_game,
)
from scripts.myriad_eval.compare_billiard_models import load_model, seed_everything


FRAME_SIZE = 512
DT = 0.01
DURATION = 0.5
N_BALLS = 16
RADIUS_NORM = 0.0333
GIVEN_STEPS = 2
COLLISION_WINDOW_STEPS = 10
SEED = 20260805
DEFAULT_DINO_PATH = Path(
    "/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master"
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    description: str
    first_position: tuple[float, float]
    first_velocity: tuple[float, float]
    second_position: tuple[float, float]
    expected_event: str


@dataclass
class Scene:
    spec: SceneSpec
    image: torch.Tensor
    truth: torch.Tensor
    collision_state_mask: np.ndarray
    ball_collision_steps: list[int]
    any_collision_steps: list[int]


SCENE_SPECS = (
    SceneSpec("head_on", "moving ball hits a stationary ball head-on", (0.25, 0.50), (480.0, 0.0), (0.40, 0.50), "ball"),
    SceneSpec("oblique", "moving ball hits a stationary ball at an angle", (0.25, 0.46), (480.0, 40.0), (0.40, 0.50), "ball"),
    SceneSpec("grazing", "moving ball barely clips a stationary ball", (0.25, 0.44), (480.0, 0.0), (0.40, 0.50), "ball"),
    SceneSpec("no_collision", "moving ball crosses open space without impact", (0.25, 0.42), (360.0, 0.0), (0.40, 0.64), "none"),
    SceneSpec("wall_bounce", "moving ball hits only the right wall", (0.75, 0.50), (480.0, 0.0), (0.30, 0.65), "wall"),
)


STATIC_POSITIONS = (
    (0.15, 0.18), (0.30, 0.18), (0.45, 0.18), (0.60, 0.18), (0.75, 0.18),
    (0.85, 0.30), (0.15, 0.82), (0.30, 0.82), (0.45, 0.82), (0.60, 0.82),
    (0.75, 0.82), (0.85, 0.70), (0.15, 0.66), (0.85, 0.58),
)


def build_scene(spec: SceneSpec) -> Scene:
    border = [35, 35, 35, 35]
    radius_px = RADIUS_NORM * FRAME_SIZE
    table = billiards.Billiard(
        obstacles=[
            billiards.InfiniteWall((0, border[0]), (FRAME_SIZE - border[1], border[0])),
            billiards.InfiniteWall((FRAME_SIZE - border[1], border[0]), (FRAME_SIZE - border[1], FRAME_SIZE - border[2])),
            billiards.InfiniteWall((FRAME_SIZE - border[1], FRAME_SIZE - border[2]), (border[3], FRAME_SIZE - border[2])),
            billiards.InfiniteWall((border[3], FRAME_SIZE - border[2]), (border[3], border[0])),
        ]
    )
    positions = (spec.first_position, spec.second_position) + STATIC_POSITIONS
    for index, position in enumerate(positions):
        velocity = spec.first_velocity if index == 0 else (0.0, 0.0)
        table.add_ball(tuple(value * FRAME_SIZE for value in position), velocity, radius_px)

    _, positions_px, _, collisions, ball_masks = simulate_billiard_game(
        table, DURATION, DT, return_ball_collision_mask=True
    )
    positions_np = np.asarray(positions_px, dtype=np.float32)
    ball_masks_np = np.asarray(ball_masks, dtype=bool)
    flow_collision_mask = expand_collision_window(ball_masks_np[1:], COLLISION_WINDOW_STEPS)
    state_collision_mask = np.zeros((positions_np.shape[0] - 1, N_BALLS), dtype=bool)
    state_collision_mask[1:] = flow_collision_mask[:-1]
    ball_collision_steps = np.flatnonzero(ball_masks_np.any(axis=1)).astype(int).tolist()
    any_collision_steps = np.flatnonzero(np.asarray(collisions, dtype=bool)).astype(int).tolist()

    if spec.expected_event == "ball" and not ball_collision_steps:
        raise RuntimeError(f"Scene {spec.name} did not produce the expected ball collision")
    if spec.expected_event == "none" and any_collision_steps:
        raise RuntimeError(f"Scene {spec.name} unexpectedly produced a collision")
    if spec.expected_event == "wall" and (not any_collision_steps or ball_collision_steps):
        raise RuntimeError(f"Scene {spec.name} did not produce an isolated wall collision")

    image_np = render_billiard_frame(positions_np[0], [radius_px] * N_BALLS, FRAME_SIZE, border)
    image = rearrange(torch.from_numpy(image_np).float().div(127.5).sub(1), "h w c -> 1 c h w")
    truth = torch.from_numpy(positions_np[:-1] / FRAME_SIZE).unsqueeze(0)
    return Scene(spec, image, truth, state_collision_mask, ball_collision_steps, any_collision_steps)


def accuracy(prediction: torch.Tensor, scene: Scene) -> dict[str, float | int | None]:
    error = torch.linalg.vector_norm(prediction[:, GIVEN_STEPS:] - scene.truth[:, GIVEN_STEPS:], dim=-1)
    error = error.mul(FRAME_SIZE).squeeze(0).cpu().numpy()
    collision_mask = scene.collision_state_mask[GIVEN_STEPS:]
    noncollision_mask = ~collision_mask

    def masked_mean(mask: np.ndarray) -> float | None:
        return float(error[mask].mean()) if mask.any() else None

    return {
        "mean_epe_px": float(error.mean()),
        "median_epe_px": float(np.median(error)),
        "p95_epe_px": float(np.quantile(error, 0.95)),
        "final_step_mean_epe_px": float(error[-1].mean()),
        "moving_ball_0_mean_epe_px": float(error[:, 0].mean()),
        "target_ball_1_mean_epe_px": float(error[:, 1].mean()),
        "collision_window_mean_epe_px": masked_mean(collision_mask),
        "noncollision_mean_epe_px": masked_mean(noncollision_mask),
        "collision_window_tokens": int(collision_mask.sum()),
        "noncollision_tokens": int(noncollision_mask.sum()),
    }


def evaluate_variant(
    name: str,
    kind: str,
    checkpoint: Path,
    mode: str,
    scenes: list[Scene],
    device: torch.device,
) -> dict:
    print(f"Evaluating {name} ({mode}) from {checkpoint}", flush=True)
    model, step, load_seconds = load_model(
        kind, checkpoint, device, physics_kinematics_mode=mode,
        collision_long_history_distance=0.04, collision_long_history_temperature=0.008,
    )
    per_scene = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for scene_index, scene in enumerate(scenes):
            print(f"  scene {scene_index + 1}/{len(scenes)}: {scene.spec.name}", flush=True)
            image = scene.image.to(device)
            truth = scene.truth.to(device)
            ts = torch.arange(truth.shape[1], dtype=torch.float32, device=device).unsqueeze(0)
            given = truth[:, :GIVEN_STEPS].reshape(1, GIVEN_STEPS * N_BALLS, 2)
            d_img = model.embed_image(image)
            seed_everything(SEED + scene_index)
            prediction = model.predict_simulate(
                n_traj=N_BALLS,
                ts=ts,
                given_pos=given,
                camera_static=torch.ones(1, dtype=torch.bool, device=device),
                d_img=d_img,
                verbose=False,
            ).float().cpu()
            per_scene[scene.spec.name] = accuracy(prediction, scene)
            del image, truth, ts, given, d_img, prediction
            torch.cuda.empty_cache()

    aggregate = {}
    scalar_keys = ("mean_epe_px", "p95_epe_px", "final_step_mean_epe_px")
    for key in scalar_keys:
        values = np.asarray([metrics[key] for metrics in per_scene.values()], dtype=np.float64)
        aggregate[f"macro_{key}"] = float(values.mean())
        aggregate[f"worst_scene_{key}"] = float(values.max())
    aggregate["macro_collision_window_mean_epe_px"] = float(np.mean([
        metrics["collision_window_mean_epe_px"]
        for metrics in per_scene.values() if metrics["collision_window_mean_epe_px"] is not None
    ]))
    aggregate["macro_noncollision_mean_epe_px"] = float(np.mean([
        metrics["noncollision_mean_epe_px"] for metrics in per_scene.values()
    ]))

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "kinematics_mode": mode,
        "load_seconds": load_seconds,
        "per_scene": per_scene,
        "aggregate": aggregate,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--bias-only", type=Path, required=True)
    parser.add_argument("--long-500", type=Path, required=True)
    parser.add_argument("--long-1000", type=Path, required=True)
    parser.add_argument("--long-1500", type=Path, required=True)
    parser.add_argument("--smooth-500", type=Path)
    parser.add_argument("--smooth-1500", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO_PATH)
    parser.add_argument(
        "--only-long-short-ablation",
        action="store_true",
        help="Evaluate the three long-history checkpoints with runtime long history disabled.",
    )
    parser.add_argument(
        "--only-smooth",
        action="store_true",
        help="Evaluate only the collision-smooth 500- and 1500-step checkpoints.",
    )
    parser.add_argument(
        "--only-smooth-disabled",
        action="store_true",
        help="Evaluate collision-smooth checkpoints with physics attention bias disabled.",
    )
    args = parser.parse_args()
    for path in (args.original, args.bias_only, args.long_500, args.long_1000, args.long_1500):
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")
    if not (args.dino_path / "config.json").is_file():
        parser.error(f"Missing DINOv3 snapshot: {args.dino_path}")

    os.environ["MYRIAD_DINO_PATH"] = str(args.dino_path)
    scenes = [build_scene(spec) for spec in SCENE_SPECS]
    device = torch.device("cuda:0")
    torch.cuda.init()
    restricted_modes = sum((
        args.only_long_short_ablation,
        args.only_smooth,
        args.only_smooth_disabled,
    ))
    if restricted_modes > 1:
        parser.error("Choose at most one restricted evaluation mode")
    if args.only_smooth or args.only_smooth_disabled:
        if args.smooth_500 is None or args.smooth_1500 is None:
            parser.error("Smooth evaluation requires --smooth-500 and --smooth-1500")
        if args.only_smooth_disabled:
            variants = (
                ("collision_smooth_500_bias_disabled", "physics_bias_disabled", args.smooth_500, "collision-smooth"),
                ("collision_smooth_1500_bias_disabled", "physics_bias_disabled", args.smooth_1500, "collision-smooth"),
            )
        else:
            variants = (
                ("collision_smooth_500", "physics_bias", args.smooth_500, "collision-smooth"),
                ("collision_smooth_1500", "physics_bias", args.smooth_1500, "collision-smooth"),
            )
    elif args.only_long_short_ablation:
        variants = (
            ("long_history_500_short_ablation", "physics_bias", args.long_500, "short"),
            ("long_history_1000_short_ablation", "physics_bias", args.long_1000, "short"),
            ("long_history_1500_short_ablation", "physics_bias", args.long_1500, "short"),
        )
    else:
        variants = (
            ("original", "original", args.original, "short"),
            ("bias_only_1000", "physics_bias", args.bias_only, "short"),
            ("long_history_500", "physics_bias", args.long_500, "long"),
            ("long_history_1000", "physics_bias", args.long_1000, "long"),
            ("long_history_1500", "physics_bias", args.long_1500, "long"),
        )
    results = {
        name: evaluate_variant(name, kind, checkpoint, mode, scenes, device)
        for name, kind, checkpoint, mode in variants
    }
    payload = {
        "reproducibility": {
            "seed": SEED,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "autocast": "bfloat16",
        },
        "protocol": {
            "frame_size": FRAME_SIZE,
            "dt_seconds": DT,
            "duration_seconds": DURATION,
            "given_steps": GIVEN_STEPS,
            "predicted_steps": int(scenes[0].truth.shape[1] - GIVEN_STEPS),
            "collision_window_steps": COLLISION_WINDOW_STEPS,
            "scenes": {
                scene.spec.name: {
                    "description": scene.spec.description,
                    "expected_event": scene.spec.expected_event,
                    "ball_collision_steps": scene.ball_collision_steps,
                    "any_collision_steps": scene.any_collision_steps,
                }
                for scene in scenes
            },
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
