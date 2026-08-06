"""Roll out one user-controlled moving ball with normal V4 context10.

The scene, noisy map prior, and same-scene context trajectories are generated
exactly as in the standard V4 inference script.  Only the query initial state
is replaced: every ball starts at rest except ``--ball-index``, whose physical
velocity is supplied by ``--velocity-x`` and ``--velocity-y``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import billiards
import cv2
import mediapy
import numpy as np
import torch
from PIL import Image

from ..adaptive_scene_billiards.data import (
    RestitutionDisk,
    RestitutionWall,
    _event_masks,
)
from ..adaptive_scene_billiards.inference import _annotate, _render_rollout
from .data import AdaptiveSceneV4EpisodeDataset, SceneSimulationV4Config
from .experiment import (
    load_checkpoint_v4,
    move_batch,
    resolve_device,
    scene_inputs_v4,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/adaptive_scene_billiards/v4_single_ball_inference"
        ),
    )
    parser.add_argument(
        "--ball-index",
        type=int,
        default=0,
        help="zero-based index of the only ball with nonzero initial velocity",
    )
    parser.add_argument(
        "--velocity-x",
        type=float,
        default=0.30,
        help="initial x velocity in normalized scene coordinates per second",
    )
    parser.add_argument(
        "--velocity-y",
        type=float,
        default=0.00,
        help="initial y velocity in normalized scene coordinates per second",
    )
    parser.add_argument("--context-trajectories", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument(
        "--candidate-samples",
        type=int,
        default=64,
        help=(
            "number of normally generated context10 scenes searched for the "
            "most collision-rich controlled rollout"
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help=(
            "use this candidate directly instead of selecting the most "
            "collision-rich controlled rollout"
        ),
    )
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    return parser


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def simulate_controlled_rollout_v4(
    simulation: SceneSimulationV4Config,
    sample: dict[str, torch.Tensor],
    initial_velocities: np.ndarray,
    steps: int,
    initial_positions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run exact physics from the sample positions and supplied velocities."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if initial_positions is None:
        initial_positions = _numpy(sample["positions"])[0]
    initial_positions = np.asarray(initial_positions, dtype=np.float64).copy()
    initial_velocities = np.asarray(initial_velocities, dtype=np.float64)
    expected_shape = (simulation.num_balls, 2)
    if initial_positions.shape != expected_shape:
        raise ValueError(
            f"sample initial positions must have shape {expected_shape}"
        )
    if initial_velocities.shape != expected_shape:
        raise ValueError(f"initial_velocities must have shape {expected_shape}")
    if not np.isfinite(initial_velocities).all():
        raise ValueError("initial velocities must be finite")

    vertices = _numpy(sample["scene_vertices"]).astype(np.float64, copy=False)
    restitution = float(_numpy(sample["scene_restitution"]))
    obstacles = [
        RestitutionWall(start, end, restitution)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    ]
    for disk, exists in zip(
        _numpy(sample["scene_disks"]),
        _numpy(sample["scene_disk_exists"]),
        strict=True,
    ):
        if bool(exists):
            x, y, radius = (float(value) for value in disk)
            obstacles.append(
                RestitutionDisk((x, y), radius, restitution)
            )

    simulator = billiards.Billiard(obstacles=obstacles)
    for position, velocity in zip(
        initial_positions,
        initial_velocities,
        strict=True,
    ):
        simulator.add_ball(
            tuple(position),
            tuple(velocity),
            radius=simulation.radius,
            mass=1.0,
        )

    positions = [simulator.balls_position.copy()]
    velocities = [simulator.balls_velocity.copy()]
    collision_objects = []
    ball_collision_objects = []
    obstacle_collision_objects = []
    for step in range(steps):
        events = simulator.evolve((step + 1) * simulation.dt)
        any_mask, ball_mask, obstacle_mask = _event_masks(
            events,
            simulation.num_balls,
        )
        collision_objects.append(any_mask)
        ball_collision_objects.append(ball_mask)
        obstacle_collision_objects.append(obstacle_mask)
        positions.append(simulator.balls_position.copy())
        velocities.append(simulator.balls_velocity.copy())

    positions_array = np.asarray(positions, dtype=np.float32)
    return {
        "positions": positions_array,
        "velocities": np.asarray(velocities, dtype=np.float32),
        "flows": positions_array[1:] - positions_array[:-1],
        "collision_objects": np.asarray(collision_objects, dtype=np.bool_),
        "ball_collision_objects": np.asarray(
            ball_collision_objects,
            dtype=np.bool_,
        ),
        "obstacle_collision_objects": np.asarray(
            obstacle_collision_objects,
            dtype=np.bool_,
        ),
    }


def _collision_score(reference: dict[str, np.ndarray]) -> tuple[int, int, int]:
    obstacle = reference["obstacle_collision_objects"]
    ball = reference["ball_collision_objects"]
    return (
        int(obstacle.any()) + int(ball.any()),
        int(obstacle.sum()) + int(ball.sum()),
        int(ball.sum()),
    )


def _choose_controlled_sample(
    dataset: AdaptiveSceneV4EpisodeDataset,
    simulation: SceneSimulationV4Config,
    initial_velocities: np.ndarray,
    steps: int,
    sample_index: int | None,
) -> tuple[int, dict[str, torch.Tensor], dict[str, np.ndarray], tuple[int, int, int]]:
    if sample_index is not None:
        if not 0 <= sample_index < len(dataset):
            raise ValueError("sample-index is outside candidate-samples")
        sample = dataset[sample_index]
        reference = simulate_controlled_rollout_v4(
            simulation,
            sample,
            initial_velocities,
            steps,
        )
        return sample_index, sample, reference, _collision_score(reference)

    best_index = 0
    best_sample = dataset[0]
    best_reference = simulate_controlled_rollout_v4(
        simulation,
        best_sample,
        initial_velocities,
        steps,
    )
    best_score = _collision_score(best_reference)
    for index in range(1, len(dataset)):
        sample = dataset[index]
        reference = simulate_controlled_rollout_v4(
            simulation,
            sample,
            initial_velocities,
            steps,
        )
        score = _collision_score(reference)
        if score > best_score:
            best_index = index
            best_sample = sample
            best_reference = reference
            best_score = score
    return best_index, best_sample, best_reference, best_score


def _save_scene_belief(
    path: Path,
    sample: dict[str, torch.Tensor],
    belief: dict[str, torch.Tensor],
    sdf_scale: float,
    image_size: int,
) -> None:
    arrays = [
        sample["clean_mask"][0].numpy(),
        sample["noisy_mask"][0].numpy(),
        belief["occupancy_map"][0, 0].float().cpu().numpy(),
        (
            belief["sdf_map"][0, 0].float().cpu().numpy() + sdf_scale
        )
        / (2 * sdf_scale),
        belief["uncertainty_map"][0, 0].float().cpu().numpy(),
        1
        - np.exp(
            -belief["contact_density"][0, 0].float().cpu().numpy()
        ),
        belief["free_evidence"][0, 0].float().cpu().numpy(),
        (
            belief["geometry_wall_evidence"][0, 0].float().cpu().numpy()
            + 1
        )
        / 2,
        belief["wall_evidence_weight"][0, 0].float().cpu().numpy(),
    ]
    panels = []
    for array in arrays:
        resized = cv2.resize(
            array,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        panel = np.repeat(np.clip(resized[..., None], 0, 1), 3, axis=-1)
        panels.append(np.rint(panel * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if args.candidate_samples < 1 or args.render_size < 64:
        raise ValueError("candidate-samples must be positive and render-size >= 64")
    if args.rollout_steps < 1:
        raise ValueError("rollout-steps must be positive")
    if not np.isfinite([args.velocity_x, args.velocity_y]).all():
        raise ValueError("velocity components must be finite")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, simulation, episode, checkpoint = load_checkpoint_v4(
        args.checkpoint,
        device,
    )
    model.eval()
    if not 0 <= args.ball_index < simulation.num_balls:
        raise ValueError("ball-index is outside the checkpoint ball range")
    if not 0 <= args.context_trajectories <= episode.max_context_trajectories:
        raise ValueError("context trajectory count is outside checkpoint range")

    steps = min(args.rollout_steps, simulation.trajectory_steps)
    initial_velocities = np.zeros(
        (simulation.num_balls, 2),
        dtype=np.float32,
    )
    initial_velocities[args.ball_index] = (args.velocity_x, args.velocity_y)
    controlled_speed = float(np.linalg.norm(initial_velocities[args.ball_index]))
    if controlled_speed > simulation.max_speed:
        print(
            "warning: controlled speed "
            f"{controlled_speed:.4f} exceeds training max_speed="
            f"{simulation.max_speed:.4f}; rollout is out of distribution"
        )

    dataset = AdaptiveSceneV4EpisodeDataset(
        simulation,
        replace(
            episode,
            min_context_trajectories=0,
            fixed_context_trajectories=args.context_trajectories,
        ),
        args.candidate_samples,
        13_000_000 + args.seed * args.candidate_samples,
    )
    sample_index, sample, reference_np, collision_score = (
        _choose_controlled_sample(
            dataset,
            simulation,
            initial_velocities,
            steps,
            args.sample_index,
        )
    )
    batch = move_batch(
        {
            key: value.unsqueeze(0) if value.ndim else value.reshape(1)
            for key, value in sample.items()
        },
        device,
    )
    belief = model.encode_scene(**scene_inputs_v4(batch))
    initial_positions = torch.from_numpy(reference_np["positions"][0]).to(
        device=device,
    )[None]
    velocity_tensor = torch.from_numpy(initial_velocities).to(device=device)[None]
    rollout = model.rollout(
        initial_positions,
        velocity_tensor,
        belief,
        steps,
    )

    prediction = rollout["positions"][0].float().cpu()
    predicted_velocities = rollout["velocities"][0].float().cpu()
    reference = torch.from_numpy(reference_np["positions"])
    reference_velocities = torch.from_numpy(reference_np["velocities"])
    position_epe = (prediction - reference).norm(dim=-1)
    velocity_epe = (predicted_velocities - reference_velocities).norm(dim=-1)
    initially_stationary = torch.ones(simulation.num_balls, dtype=torch.bool)
    initially_stationary[args.ball_index] = False

    predicted_video = _render_rollout(
        prediction,
        sample["scene_vertices"],
        sample["scene_disks"],
        sample["scene_disk_exists"],
        sample["object_colors"],
        simulation.radius,
        args.render_size,
    )
    reference_video = _render_rollout(
        reference,
        sample["scene_vertices"],
        sample["scene_disks"],
        sample["scene_disk_exists"],
        sample["object_colors"],
        simulation.radius,
        args.render_size,
    )
    comparison = []
    for frame_index, (predicted_frame, reference_frame) in enumerate(
        zip(predicted_video, reference_video, strict=True)
    ):
        comparison.append(
            np.concatenate(
                [
                    _annotate(
                        predicted_frame,
                        "V4 MODEL",
                        f"t={frame_index * simulation.dt:.2f}s "
                        f"EPE={position_epe[frame_index].mean():.4f}",
                    ),
                    _annotate(
                        reference_frame,
                        "PHYSICS",
                        f"ball={args.ball_index} "
                        f"v=({args.velocity_x:.3f},{args.velocity_y:.3f})",
                    ),
                ],
                axis=1,
            )
        )
    comparison_video = np.stack(comparison)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fps = max(1, round(1 / simulation.dt))
    mediapy.write_video(
        str(args.out_dir / "comparison.mp4"),
        comparison_video,
        fps=fps,
    )
    mediapy.write_video(
        str(args.out_dir / "model.mp4"),
        predicted_video,
        fps=fps,
    )
    mediapy.write_video(
        str(args.out_dir / "physics.mp4"),
        reference_video,
        fps=fps,
    )
    Image.fromarray(comparison_video[-1]).save(
        args.out_dir / "final_comparison.png"
    )
    _save_scene_belief(
        args.out_dir / "scene_belief_v4.png",
        sample,
        belief,
        model.config.sdf_scale,
        simulation.image_size,
    )

    torch.save(
        {
            "initial_velocities": torch.from_numpy(initial_velocities),
            "prediction_positions": prediction,
            "physics_positions": reference,
            "prediction_velocities": predicted_velocities,
            "physics_velocities": reference_velocities,
            "position_epe": position_epe,
            "velocity_epe": velocity_epe,
            "physics_collision_objects": torch.from_numpy(
                reference_np["collision_objects"]
            ),
            "physics_ball_collision_objects": torch.from_numpy(
                reference_np["ball_collision_objects"]
            ),
            "physics_obstacle_collision_objects": torch.from_numpy(
                reference_np["obstacle_collision_objects"]
            ),
            "scene_occupancy": belief["occupancy_map"].float().cpu(),
            "scene_sdf": belief["sdf_map"].float().cpu(),
            "scene_normals": belief["normal_map"].float().cpu(),
            "scene_uncertainty": belief["uncertainty_map"].float().cpu(),
        },
        args.out_dir / "rollouts.pt",
    )

    metadata = {
        "format": "adaptive_scene_billiards_v4_single_ball",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "candidate_index": sample_index,
        "collision_score": list(collision_score),
        "wall_collision_events": int(
            reference_np["obstacle_collision_objects"].sum()
        ),
        "ball_collision_events": int(
            reference_np["ball_collision_objects"].sum() // 2
        ),
        "context_trajectories": args.context_trajectories,
        "inferred_wall_contacts": int(
            (sample["context_contact_confidence"] > 0).sum()
        ),
        "controlled_ball_index": args.ball_index,
        "initial_velocity": [args.velocity_x, args.velocity_y],
        "initial_speed": controlled_speed,
        "training_max_speed": simulation.max_speed,
        "rollout_steps": steps,
        "position_ade": float(position_epe[1:].mean()),
        "position_fde": float(position_epe[-1].mean()),
        "velocity_epe": float(velocity_epe[1:].mean()),
        "position_ade_per_object": position_epe[1:].mean(dim=0).tolist(),
        "position_fde_per_object": position_epe[-1].tolist(),
        "velocity_epe_per_object": velocity_epe[1:].mean(dim=0).tolist(),
        "initially_stationary_position_ade": float(
            position_epe[1:, initially_stationary].mean()
        ),
        "initially_stationary_position_fde": float(
            position_epe[-1, initially_stationary].mean()
        ),
    }
    with (args.out_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"saved controlled V4 rollout to {args.out_dir}")


if __name__ == "__main__":
    main()
