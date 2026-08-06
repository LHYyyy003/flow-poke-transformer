#!/usr/bin/env python3
"""Render one dataset-custom scene with three MYRIAD billiards variants."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.myriad_eval.compare_billiard_models import load_model, seed_everything


DEFAULT_DINO_PATH = Path(
    "/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master"
)
GIVEN_STEPS = 2
SEED = 20260805


def load_custom_data_module():
    path = REPO_ROOT / "dataset-custom" / "data.py"
    spec = importlib.util.spec_from_file_location("dataset_custom_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import dataset generator: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_collision_sample(data_module, requested_index: int, wall_style: str):
    training_walls = wall_style == "training"
    simulation = data_module.SceneSimulationConfig(
        image_size=512,
        num_balls=6,
        radius=0.0333,
        dt=0.01,
        trajectory_steps=50,
        moving_probability=0.80,
        min_speed=0.35,
        max_speed=0.70,
        corner_jitter=0.0 if training_walls else 0.045,
        max_disk_obstacles=0 if training_walls else 1,
        disk_probability=0.0 if training_walls else 0.35,
        min_disk_radius=0.055,
        max_disk_radius=0.10,
        min_restitution=1.0 if training_walls else 0.72,
        max_restitution=1.0,
        mask_supersample=2,
    )
    episode = data_module.AdaptiveEpisodeConfig(fixed_context_trajectories=0)
    dataset = data_module.AdaptiveSceneEpisodeDataset(
        simulation=simulation,
        episode=episode,
        base_seed=51_000,
    )
    indices = [requested_index] if requested_index >= 0 else range(250)
    for index in indices:
        sample = dataset.sample_at(index)
        collision_steps = torch.where(sample["ball_collision_objects"].any(dim=1))[0]
        if requested_index >= 0 or collision_steps.numel() > 0:
            return dataset, sample, int(index), collision_steps.tolist()
    raise RuntimeError("No ball-ball collision found in the first 250 deterministic samples")


def predict(checkpoint: Path, kind: str, mode: str, image: torch.Tensor,
            truth: torch.Tensor, device: torch.device, dino_path: Path,
            strength: float | None = None) -> tuple[np.ndarray, int]:
    os.environ["MYRIAD_DINO_PATH"] = str(dino_path)
    model, step, _ = load_model(
        kind,
        checkpoint,
        device,
        physics_strength=strength,
        physics_kinematics_mode=mode,
        collision_long_history_distance=0.04,
    )
    image_device = image.to(device)
    truth_device = truth.to(device)
    ts = torch.arange(truth.size(1), dtype=torch.float32, device=device).unsqueeze(0)
    given = truth_device[:, :GIVEN_STEPS].reshape(1, -1, 2)
    camera_static = torch.ones(1, dtype=torch.bool, device=device)
    seed_everything(SEED)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        d_img = model.embed_image(image_device)
        prediction = model.predict_simulate(
            n_traj=truth.size(2),
            ts=ts,
            given_pos=given,
            camera_static=camera_static,
            d_img=d_img,
            verbose=False,
        ).float()
    result = prediction.cpu().numpy()[0]
    del model, image_device, truth_device, ts, given, d_img, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result, step


def render_frames(data_module, positions: np.ndarray, sample: dict[str, torch.Tensor],
                  label: str, truth: np.ndarray | None = None) -> np.ndarray:
    frames = []
    for frame_index, frame_positions in enumerate(positions):
        rgb = data_module.render_scene(
            frame_positions,
            sample["clean_mask"],
            sample["object_colors"],
            radius=0.0333,
            image_size=512,
        ).permute(1, 2, 0).mul(255).byte().numpy()
        canvas = rgb.copy()
        cv2.rectangle(canvas, (0, 0), (511, 46), (20, 23, 27), thickness=-1)
        cv2.putText(canvas, label, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (245, 245, 245), 1, cv2.LINE_AA)
        subtitle = f"frame {frame_index:02d}"
        if truth is not None and frame_index >= GIVEN_STEPS:
            epe = np.linalg.norm(frame_positions - truth[frame_index], axis=-1).mean() * 512
            subtitle += f"   frame EPE {epe:.3f}px"
        cv2.putText(canvas, subtitle, (12, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (215, 220, 225), 1, cv2.LINE_AA)
        frames.append(canvas)
    return np.stack(frames)


def write_video(path: Path, rgb_frames: np.ndarray, fps: float) -> None:
    height, width = rgb_frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    try:
        for frame in rgb_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def accuracy(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(prediction[GIVEN_STEPS:] - truth[GIVEN_STEPS:], axis=-1) * 512
    return {
        "mean_epe_px": float(errors.mean()),
        "p95_epe_px": float(np.quantile(errors, 0.95)),
        "final_step_mean_epe_px": float(errors[-1].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--bias-only", type=Path, required=True)
    parser.add_argument("--bias-long", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument(
        "--wall-style", choices=("training", "adaptive"), default="training",
        help="Use the fixed rectangular training table by default; adaptive enables OOD walls.",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--bias-long-strength", type=float, default=0.25)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO_PATH)
    args = parser.parse_args()
    if not (args.dino_path / "config.json").is_file():
        parser.error(f"Missing local DINOv3 snapshot: {args.dino_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_module = load_custom_data_module()
    _, sample, sample_index, collision_steps = choose_collision_sample(
        data_module, args.sample_index, args.wall_style
    )
    truth_np = sample["positions"][:-1].numpy()
    truth = torch.from_numpy(truth_np).unsqueeze(0)
    image = data_module.render_scene(
        truth_np[0], sample["clean_mask"], sample["object_colors"], 0.0333, 512
    ).unsqueeze(0).mul(2).sub(1)

    device = torch.device("cuda:0")
    torch.cuda.init()
    predictions = {}
    steps = {}
    predictions["original"], steps["original"] = predict(
        args.original, "original", "short", image, truth, device, args.dino_path
    )
    predictions["bias_only"], steps["bias_only"] = predict(
        args.bias_only, "physics_bias", "short", image, truth, device, args.dino_path
    )
    predictions["bias_long"], steps["bias_long"] = predict(
        args.bias_long, "physics_bias", "collision-smooth", image, truth, device,
        args.dino_path, strength=args.bias_long_strength,
    )

    frame_sets = {
        "ground_truth": render_frames(data_module, truth_np, sample, "GROUND TRUTH"),
        "original": render_frames(data_module, predictions["original"], sample, "ORIGINAL", truth_np),
        "bias_only": render_frames(data_module, predictions["bias_only"], sample, "BIAS ONLY", truth_np),
        "bias_long": render_frames(
            data_module, predictions["bias_long"], sample,
            f"BIAS + SMOOTH LONG HISTORY (x{args.bias_long_strength:g})", truth_np,
        ),
    }
    for name, frames in frame_sets.items():
        write_video(args.output_dir / f"{name}.mp4", frames, args.fps)
    comparison = np.concatenate(
        [frame_sets[name] for name in ("ground_truth", "original", "bias_only", "bias_long")],
        axis=2,
    )
    write_video(args.output_dir / "comparison.mp4", comparison, args.fps)

    metrics = {
        "dataset_custom_sample_index": sample_index,
        "wall_style": args.wall_style,
        "wall_configuration": {
            "corner_jitter": 0.0 if args.wall_style == "training" else 0.045,
            "disk_obstacles": 0 if args.wall_style == "training" else "sampled_0_or_1",
            "restitution": 1.0 if args.wall_style == "training" else "sampled_0.72_to_1.0",
        },
        "ball_collision_steps": collision_steps,
        "given_steps": GIVEN_STEPS,
        "predicted_steps": int(truth_np.shape[0] - GIVEN_STEPS),
        "checkpoints": {
            "original": {"path": str(args.original), "step": steps["original"]},
            "bias_only": {"path": str(args.bias_only), "step": steps["bias_only"], "kinematics": "short"},
            "bias_long": {
                "path": str(args.bias_long), "step": steps["bias_long"],
                "kinematics": "collision-smooth", "collision_distance": 0.04,
                "collision_temperature": 0.008, "strength": args.bias_long_strength,
            },
        },
        "accuracy": {name: accuracy(value, truth_np) for name, value in predictions.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "trajectories.npz", truth=truth_np, **predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
