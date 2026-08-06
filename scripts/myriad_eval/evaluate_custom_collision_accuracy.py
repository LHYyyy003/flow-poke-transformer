#!/usr/bin/env python3
"""Average ball-ball and ball-wall collision accuracy over custom scenes."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.myriad_eval.compare_billiard_models import load_model, seed_everything


FRAME_SIZE = 512
GIVEN_STEPS = 2
SEED = 20260805
RADIUS = 0.0333


def load_custom_data_module():
    path = REPO_ROOT / "dataset-custom" / "data.py"
    spec = importlib.util.spec_from_file_location("dataset_custom_collision_accuracy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import dataset generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_dataset(data_module):
    simulation = data_module.SceneSimulationConfig(
        image_size=512, num_balls=6, radius=RADIUS, dt=0.01,
        trajectory_steps=50, moving_probability=0.80, min_speed=0.35,
        max_speed=0.70, corner_jitter=0.0, max_disk_obstacles=0,
        disk_probability=0.0, min_disk_radius=0.055, max_disk_radius=0.10,
        min_restitution=1.0, max_restitution=1.0, mask_supersample=2,
    )
    episode = data_module.AdaptiveEpisodeConfig(fixed_context_trajectories=0)
    return data_module.AdaptiveSceneEpisodeDataset(
        simulation=simulation, episode=episode, base_seed=51_000,
    )


def collect_scenes(dataset, count: int, search_limit: int):
    ball, wall = [], []
    for index in range(search_limit):
        sample = dataset.sample_at(index)
        has_ball = bool(sample["ball_collision_objects"].any())
        has_wall = bool(sample["obstacle_collision_objects"].any())
        if has_ball and len(ball) < count:
            ball.append((index, sample))
        if has_wall and len(wall) < count:
            wall.append((index, sample))
        if len(ball) >= count and len(wall) >= count:
            break
    if len(ball) < count or len(wall) < count:
        raise RuntimeError(
            f"Only found ball={len(ball)} wall={len(wall)} scenes; increase search_limit"
        )
    return {"ball_ball": ball, "ball_wall": wall}


def predict(model, data_module, sample, device):
    truth = sample["positions"][:-1].numpy().astype(np.float32)
    image = data_module.render_scene(
        truth[0], sample["clean_mask"], sample["object_colors"], RADIUS, FRAME_SIZE,
    ).unsqueeze(0).mul(2).sub(1).to(device)
    truth_device = torch.from_numpy(truth).unsqueeze(0).to(device)
    ts = torch.arange(truth.shape[0], dtype=torch.float32, device=device).unsqueeze(0)
    given = truth_device[:, :GIVEN_STEPS].reshape(1, GIVEN_STEPS * truth.shape[1], 2)
    camera_static = torch.ones(1, dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        d_img = model.embed_image(image)
        seed_everything(SEED)
        pred = model.predict_simulate(
            n_traj=truth.shape[1], ts=ts, given_pos=given,
            camera_static=camera_static, d_img=d_img, verbose=False,
        ).float().cpu().numpy()[0]
    del image, truth_device, ts, given, d_img
    return truth, pred


def point_segment_distance(points, start, end):
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return np.linalg.norm(points - start, axis=-1)
    u = np.clip(((points - start) * segment).sum(axis=-1) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (start + u[..., None] * segment), axis=-1)


def predicted_contact_events(positions, vertices):
    ball_events = np.zeros(positions.shape[0], dtype=bool)
    wall_events = np.zeros(positions.shape[0], dtype=bool)
    for t, points in enumerate(positions):
        distances = []
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                if np.linalg.norm(points[i] - points[j]) <= 2 * RADIUS * 1.15:
                    ball_events[t] = True
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            distances.append(point_segment_distance(points, start, end))
        if distances and np.min(np.stack(distances)) <= RADIUS * 1.15:
            wall_events[t] = True
    return ball_events, wall_events


def event_stats(gt_events, pred_events, tolerance=2):
    gt_steps = np.flatnonzero(gt_events)
    pred_steps = np.flatnonzero(pred_events)
    if len(gt_steps) == 0:
        return {"recall_pct": None, "precision_pct": None, "f1_pct": None}
    matched_gt = sum(np.any(np.abs(pred_steps - step) <= tolerance) for step in gt_steps)
    matched_pred = sum(np.any(np.abs(gt_steps - step) <= tolerance) for step in pred_steps)
    recall = matched_gt / len(gt_steps)
    precision = matched_pred / len(pred_steps) if len(pred_steps) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "recall_pct": float(100 * recall),
        "precision_pct": float(100 * precision),
        "f1_pct": float(100 * f1),
        "gt_event_steps": gt_steps.tolist(),
        "pred_event_steps": pred_steps.tolist(),
    }


def summarize_scene(model, data_module, item, device, category):
    index, sample = item
    truth, prediction = predict(model, data_module, sample, device)
    errors = np.linalg.norm(prediction - truth, axis=-1) * FRAME_SIZE
    if category == "ball_ball":
        mask = sample["ball_collision_objects"].numpy().astype(bool)
    else:
        mask = sample["obstacle_collision_objects"].numpy().astype(bool)
    mask[:GIVEN_STEPS] = False
    values = errors[mask]
    vertices = sample["scene_vertices"].numpy()
    pred_ball, pred_wall = predicted_contact_events(prediction, vertices)
    gt_ball = sample["ball_collision_objects"].numpy().any(axis=1)
    gt_wall = sample["obstacle_collision_objects"].numpy().any(axis=1)
    event = event_stats(gt_ball if category == "ball_ball" else gt_wall,
                        pred_ball if category == "ball_ball" else pred_wall)
    return {
        "scene_index": index,
        "category": category,
        "collision_tokens": int(mask.sum()),
        "mean_epe_px": float(values.mean()) if values.size else None,
        "median_epe_px": float(np.median(values)) if values.size else None,
        "accuracy_le_1px_pct": float(100 * np.mean(values <= 1.0)) if values.size else None,
        "accuracy_le_2px_pct": float(100 * np.mean(values <= 2.0)) if values.size else None,
        "accuracy_le_5px_pct": float(100 * np.mean(values <= 5.0)) if values.size else None,
        "event": event,
    }


def aggregate(rows):
    result = {"scene_count": len(rows)}
    for key in ("mean_epe_px", "accuracy_le_1px_pct", "accuracy_le_2px_pct", "accuracy_le_5px_pct"):
        vals = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(vals)) if vals else None
    for key in ("recall_pct", "precision_pct", "f1_pct"):
        vals = [row["event"][key] for row in rows if row["event"][key] is not None]
        result[f"event_{key}"] = float(np.mean(vals)) if vals else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--physics-500", type=Path, required=True)
    parser.add_argument("--long-history-500", type=Path, required=True)
    parser.add_argument(
        "--gated-long-history", type=Path, action="append", default=None,
        help="Optional trained long-history checkpoint with cross-track gating; repeat for multiple steps",
    )
    parser.add_argument(
        "--only-gated", action="store_true",
        help="Evaluate only the repeated --gated-long-history checkpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, required=True)
    parser.add_argument("--scenes-per-category", type=int, default=10)
    parser.add_argument("--search-limit", type=int, default=250)
    args = parser.parse_args()
    os.environ["MYRIAD_DINO_PATH"] = str(args.dino_path)
    data_module = load_custom_data_module()
    dataset = make_dataset(data_module)
    scenes = collect_scenes(dataset, args.scenes_per_category, args.search_limit)
    device = torch.device("cuda:0")
    torch.cuda.init()
    models = {}
    if not args.only_gated:
        models["original"] = load_model("original", args.original, device)[0]
        models["physics_only_500_strength025"] = load_model(
            "physics_bias", args.physics_500, device, physics_strength=0.25,
            physics_kinematics_mode="short",
        )[0]
        models["long_history_only_500"] = load_model(
            "long_history_bias", args.long_history_500, device,
            physics_kinematics_mode="temporal-only",
        )[0]
        # The repository's trained collision-smooth checkpoint is the production
        # combined variant: physical relation features plus causal long-history
        # kinematics inside the same bias module.
        models["combined_500"] = load_model(
            "physics_bias", args.physics_500, device, physics_strength=0.25,
            physics_kinematics_mode="collision-smooth",
        )[0]
    if args.gated_long_history is not None:
        for checkpoint in args.gated_long_history:
            step = checkpoint.stem.rsplit("_", 1)[-1].lstrip("0") or "0"
            models[f"long_history_gated_{step}"] = load_model(
                "long_history_bias", checkpoint, device,
                physics_kinematics_mode="temporal-only",
            )[0]
    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for category, items in scenes.items():
            print(f"Evaluating {model_name} {category} ({len(items)} scenes)", flush=True)
            rows = [summarize_scene(model, data_module, item, device, category) for item in items]
            results[model_name][category] = {
                "aggregate": aggregate(rows), "per_scene": rows,
            }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "protocol": {
            "scene_source": "dataset-custom training walls",
            "scenes_per_category": args.scenes_per_category,
            "scene_indices": {k: [item[0] for item in v] for k, v in scenes.items()},
            "given_steps": GIVEN_STEPS,
            "event_tolerance_steps": 2,
            "position_accuracy_thresholds_px": [1.0, 2.0, 5.0],
            "physics_strength": 0.25,
            "combined_definition": "physics checkpoint with collision-smooth causal long-history kinematics",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
