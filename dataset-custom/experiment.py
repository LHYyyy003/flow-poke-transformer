"""Training/evaluation utilities for adaptive scene billiards V4."""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ..adaptive_scene_billiards.experiment import (
    move_batch,
    resolve_device,
    save_json,
    seed_everything,
)
from .data import AdaptiveV4EpisodeConfig, SceneSimulationV4Config
from .model import AdaptiveDynamicsV4Config, AdaptiveSceneDynamicsModelV4


CONTEXT_KEYS_V4 = (
    "context_positions",
    "context_velocities",
    "context_displacements",
    "context_next_velocities",
    "context_neighbor_positions",
    "context_neighbor_velocities",
    "context_neighbor_distances",
    "context_times",
    "context_trajectory_ids",
    "context_contact_points",
    "context_contact_normals",
    "context_contact_confidence",
    "context_free_evidence",
    "context_confidence",
    "context_valid",
)

EVALUATION_BATCH_KEYS_V4 = tuple(
    dict.fromkeys(
        (
            *CONTEXT_KEYS_V4,
            "noisy_mask",
            "mask_confidence",
            "clean_mask",
            "clean_sdf",
            "clean_normals",
            "positions",
            "velocities",
            "flows",
            "collision_objects",
            "obstacle_collision_objects",
            "ball_collision_objects",
            "context_count",
        )
    )
)


def scene_inputs_v4(
    batch: dict[str, torch.Tensor],
    *,
    zero_context: bool = False,
    clean_prior: bool = False,
) -> dict[str, torch.Tensor]:
    result = {key: batch[key] for key in CONTEXT_KEYS_V4}
    result["noisy_mask"] = batch["clean_mask"] if clean_prior else batch["noisy_mask"]
    result["mask_confidence"] = (
        torch.ones_like(batch["clean_mask"])
        if clean_prior
        else batch["mask_confidence"]
    )
    if zero_context:
        result["context_valid"] = torch.zeros_like(batch["context_valid"])
        result["context_confidence"] = torch.zeros_like(batch["context_confidence"])
        result["context_contact_confidence"] = torch.zeros_like(
            batch["context_contact_confidence"]
        )
        result["context_free_evidence"] = torch.zeros_like(
            batch["context_free_evidence"]
        )
    return result


def sample_query_windows_v4(
    batch: dict[str, torch.Tensor],
    rollout_horizon: int,
    wall_probability: float = 0.50,
    ball_probability: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Stratify windows into wall, ball-ball, and random dynamics cases."""

    displacements = batch["flows"]
    batch_size, num_steps, _, _ = displacements.shape
    if not 1 <= rollout_horizon <= num_steps:
        raise ValueError("rollout_horizon must be in [1, num_steps]")
    if wall_probability < 0 or ball_probability < 0 or wall_probability + ball_probability > 1:
        raise ValueError("invalid stratified sampling probabilities")
    num_starts = num_steps - rollout_horizon + 1
    wall_windows = batch["obstacle_collision_objects"].unfold(
        1, rollout_horizon, 1
    ).any(dim=(-1, -2))
    ball_windows = batch["ball_collision_objects"].unfold(
        1, rollout_horizon, 1
    ).any(dim=(-1, -2))
    random_times = torch.randint(0, num_starts, (batch_size,), device=displacements.device)

    def select(mask: torch.Tensor) -> torch.Tensor:
        scores = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1)
        return scores.argmax(dim=-1)

    draw = torch.rand(batch_size, device=displacements.device)
    use_wall = (draw < wall_probability) & wall_windows.any(dim=-1)
    use_ball = (
        (draw >= wall_probability)
        & (draw < wall_probability + ball_probability)
        & ball_windows.any(dim=-1)
    )
    times = torch.where(use_wall, select(wall_windows), random_times)
    times = torch.where(use_ball, select(ball_windows), times)
    batch_indices = torch.arange(batch_size, device=displacements.device)
    future_times = times[:, None] + torch.arange(
        rollout_horizon, device=displacements.device
    )[None]
    next_times = future_times + 1
    return {
        "positions": batch["positions"][batch_indices, times],
        "velocities": batch["velocities"][batch_indices, times],
        "exists": batch["exists"][batch_indices, times],
        "future_positions": batch["positions"][batch_indices[:, None], next_times],
        "future_displacements": displacements[batch_indices[:, None], future_times],
        "future_velocities": batch["velocities"][batch_indices[:, None], next_times],
        "future_collision_objects": batch["collision_objects"][
            batch_indices[:, None], future_times
        ],
        "future_wall_objects": batch["obstacle_collision_objects"][
            batch_indices[:, None], future_times
        ],
        "future_ball_objects": batch["ball_collision_objects"][
            batch_indices[:, None], future_times
        ],
        "times": times,
    }


def _field_targets(
    batch: dict[str, torch.Tensor],
    belief: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = belief["sdf_map"].shape[-2:]
    occupancy = F.interpolate(batch["clean_mask"], size=size, mode="area")
    sdf = F.interpolate(
        batch["clean_sdf"],
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    normals = F.interpolate(
        batch["clean_normals"],
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    normals = F.normalize(normals, dim=1, eps=1e-6)
    return occupancy, sdf, normals


def inject_oracle_geometry_v4(
    belief: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Route exact raster geometry through the dynamics' explicit-map API.

    This is used only for the Phase-A dynamics upper bound and oracle
    validation.  Learned latents remain in the diagnostic belief dictionary,
    but the dynamics cannot read them; adaptive phases use the same four-map
    interface with predicted geometry.
    """

    occupancy, sdf, normals = _field_targets(batch, belief)
    result = belief.copy()
    logits = torch.logit(occupancy.clamp(1e-4, 1 - 1e-4))
    result.update(
        occupancy_logits_map=logits,
        occupancy_map=occupancy,
        sdf_map=sdf,
        normal_map=normals,
        uncertainty_map=torch.zeros_like(sdf),
        occupancy_logits=logits.flatten(2).transpose(1, 2),
        occupancy=occupancy.flatten(2).transpose(1, 2),
        sdf=sdf.flatten(2).transpose(1, 2),
        uncertainty=torch.zeros_like(sdf).flatten(2).transpose(1, 2),
    )
    return result


def scene_field_loss_v4(
    model: AdaptiveSceneDynamicsModelV4,
    belief: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    occupancy_weight: float = 0.20,
    sdf_weight: float = 1.0,
    normal_weight: float = 0.20,
    uncertainty_weight: float = 0.02,
    occupancy_consistency_weight: float = 0.10,
    normal_consistency_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    occupancy, target_sdf, target_normals = _field_targets(batch, belief)
    occupancy_loss = F.binary_cross_entropy_with_logits(
        belief["occupancy_logits_map"].float(),
        occupancy.float(),
    )
    sdf_error = F.smooth_l1_loss(
        belief["sdf_map"],
        target_sdf,
        reduction="none",
        beta=0.005,
    )
    near_boundary = target_sdf.abs() < 3 * model.config.object_radius
    boundary_weights = 1 + 4 * near_boundary.to(sdf_error.dtype)
    sdf_loss = (sdf_error * boundary_weights).sum() / boundary_weights.sum().clamp_min(1)
    cosine = (belief["normal_map"] * target_normals).sum(dim=1, keepdim=True)
    normal_valid = near_boundary & (target_normals.norm(dim=1, keepdim=True) > 0.5)
    normal_loss = ((1 - cosine) * normal_valid).sum() / normal_valid.sum().clamp_min(1)

    # Keep all explicit maps on the same zero level set.  SDF is the primary
    # geometric representation; detached SDF-derived targets prevent auxiliary
    # heads from distorting it merely to make the consistency terms easier.
    consistency_temperature = max(model.config.object_radius / 4, 1e-4)
    sdf_occupancy = torch.sigmoid(
        -belief["sdf_map"].detach() / consistency_temperature
    )
    occupancy_consistency = F.binary_cross_entropy_with_logits(
        belief["occupancy_logits_map"],
        sdf_occupancy,
    )
    padded_sdf = F.pad(belief["sdf_map"], (1, 1, 1, 1), mode="replicate")
    gradient_x = padded_sdf[:, :, 1:-1, 2:] - padded_sdf[:, :, 1:-1, :-2]
    gradient_y = padded_sdf[:, :, 2:, 1:-1] - padded_sdf[:, :, :-2, 1:-1]
    sdf_gradient = torch.cat([gradient_x, gradient_y], dim=1)
    gradient_magnitude = sdf_gradient.norm(dim=1, keepdim=True)
    sdf_normals = F.normalize(sdf_gradient, dim=1, eps=1e-6).detach()
    consistency_valid = near_boundary & (gradient_magnitude.detach() > 1e-5)
    normal_consistency = (
        (1 - (belief["normal_map"] * sdf_normals).sum(dim=1, keepdim=True))
        * consistency_valid
    ).sum() / consistency_valid.sum().clamp_min(1)
    absolute_sdf_error = (belief["sdf_map"] - target_sdf).abs()
    sigma = 0.001 + model.config.sdf_scale * belief["uncertainty_map"]
    uncertainty_nll = (
        (absolute_sdf_error / sigma + sigma.log()) * boundary_weights
    ).sum() / boundary_weights.sum().clamp_min(1)
    loss = (
        occupancy_weight * occupancy_loss
        + sdf_weight * sdf_loss
        + normal_weight * normal_loss
        + uncertainty_weight * uncertainty_nll
        + occupancy_consistency_weight * occupancy_consistency
        + normal_consistency_weight * normal_consistency
    )
    return loss, {
        "field_loss": loss.detach(),
        "occupancy_bce": occupancy_loss.detach(),
        "sdf_smooth_l1": sdf_loss.detach(),
        "normal_cosine_loss": normal_loss.detach(),
        "uncertainty_nll": uncertainty_nll.detach(),
        "occupancy_sdf_consistency": occupancy_consistency.detach(),
        "normal_sdf_consistency": normal_consistency.detach(),
    }


def context_evidence_loss_v4(
    model: AdaptiveSceneDynamicsModelV4,
    belief: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Make predicted geometry agree with trajectory-derived wall/free evidence."""

    batch_size, num_tokens, _ = batch["context_contact_points"].shape
    contact_grid = batch["context_contact_points"].mul(2).sub(1).reshape(
        batch_size, num_tokens, 1, 2
    )
    sampled_sdf = F.grid_sample(
        belief["sdf_map"],
        contact_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(1).squeeze(-1)
    sampled_normals = F.grid_sample(
        belief["normal_map"],
        contact_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(-1).transpose(1, 2)
    contact_weights = (
        batch["context_contact_confidence"]
        * batch["context_valid"].to(sampled_sdf.dtype)
    )
    contact_denominator = contact_weights.sum().clamp_min(1)
    contact_sdf_loss = (
        sampled_sdf.abs() / model.config.object_radius * contact_weights
    ).sum() / contact_denominator
    normal_cosine = (
        F.normalize(sampled_normals, dim=-1, eps=1e-6)
        * batch["context_contact_normals"]
    ).sum(dim=-1)
    contact_normal_loss = (
        (1 - normal_cosine) * contact_weights
    ).sum() / contact_denominator

    free_evidence = F.interpolate(
        batch["context_free_evidence"],
        size=belief["sdf_map"].shape[-2:],
        mode="area",
    )
    free_denominator = free_evidence.sum().clamp_min(1)
    free_occupancy_loss = (
        belief["occupancy_map"] * free_evidence
    ).sum() / free_denominator
    free_clearance_loss = (
        F.relu(0.5 * model.config.object_radius - belief["sdf_map"])
        / model.config.object_radius
        * free_evidence
    ).sum() / free_denominator
    loss = (
        contact_sdf_loss
        + 0.25 * contact_normal_loss
        + 0.25 * free_occupancy_loss
        + 0.25 * free_clearance_loss
    )
    return loss, {
        "context_evidence_loss": loss.detach(),
        "contact_sdf_loss": contact_sdf_loss.detach(),
        "contact_normal_loss": contact_normal_loss.detach(),
        "free_occupancy_loss": free_occupancy_loss.detach(),
        "free_clearance_loss": free_clearance_loss.detach(),
    }


def repeat_scene_belief_v4(
    belief: dict[str, torch.Tensor],
    repeats: int,
) -> dict[str, torch.Tensor]:
    dynamics_keys = (
        "occupancy_map",
        "sdf_map",
        "normal_map",
        "uncertainty_map",
    )
    if repeats == 1:
        return {key: belief[key] for key in dynamics_keys}
    return {
        key: belief[key][:, None]
        .expand(-1, repeats, *belief[key].shape[1:])
        .reshape(-1, *belief[key].shape[1:])
        for key in dynamics_keys
    }


@torch.inference_mode()
def evaluate_dynamics_v4(
    model: AdaptiveSceneDynamicsModelV4,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int = 4,
    rollout_steps: int = 32,
    one_step_time_chunk: int = 4,
    clean_prior: bool = False,
    zero_context: bool = False,
    oracle_geometry: bool = False,
) -> dict[str, float]:
    model.eval()
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    names = (
        "displacement_epe",
        "velocity_epe",
        "wall_displacement_epe",
        "wall_velocity_epe",
        "ball_displacement_epe",
        "ball_velocity_epe",
        "free_displacement_epe",
        "free_velocity_epe",
        "stationary_displacement_epe",
        "stationary_velocity_epe",
        "rollout_ade",
        "rollout_fde",
        "stationary_rollout_ade",
        "stationary_rollout_fde",
        "sdf_l1",
        "normal_cosine_error",
        "occupancy_l1",
        "context_count",
    )
    sums: dict[str, torch.Tensor | None] = {name: None for name in names}
    counts = {name: 0 for name in names}

    def accumulate(name: str, values: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            values = values[mask]
        if values.numel():
            value_sum = values.float().sum()
            current_sum = sums[name]
            sums[name] = (
                value_sum if current_sum is None else current_sum + value_sum
            )
            counts[name] += values.numel()

    for batch_number, raw_batch in enumerate(batches):
        if batch_number >= max_batches:
            break
        batch = move_batch(
            {key: raw_batch[key] for key in EVALUATION_BATCH_KEYS_V4},
            device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            belief = model.encode_scene(
                **scene_inputs_v4(
                    batch,
                    zero_context=zero_context,
                    clean_prior=clean_prior,
                )
            )
        if oracle_geometry:
            belief = inject_oracle_geometry_v4(belief, batch)
        batch_size, num_steps, num_objects, _ = batch["flows"].shape
        for start in range(0, num_steps, one_step_time_chunk):
            end = min(start + one_step_time_chunk, num_steps)
            chunk = end - start
            positions = batch["positions"][:, start:end].reshape(-1, num_objects, 2)
            velocities = batch["velocities"][:, start:end].reshape(-1, num_objects, 2)
            target_displacement = batch["flows"][:, start:end].reshape(-1, num_objects, 2)
            target_velocity = batch["velocities"][:, start + 1 : end + 1].reshape(
                -1, num_objects, 2
            )
            repeated_belief = repeat_scene_belief_v4(belief, chunk)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                prediction = model(
                    positions,
                    velocities,
                    repeated_belief,
                )
            displacement_error = (
                prediction["displacement"].float() - target_displacement.float()
            ).norm(dim=-1)
            velocity_error = (
                prediction["next_velocity"].float() - target_velocity.float()
            ).norm(dim=-1)
            wall = batch["obstacle_collision_objects"][:, start:end].reshape(
                -1, num_objects
            )
            ball = batch["ball_collision_objects"][:, start:end].reshape(
                -1, num_objects
            )
            free = ~batch["collision_objects"][:, start:end].reshape(-1, num_objects)
            stationary = (
                free
                & (velocities.norm(dim=-1) <= 1e-8)
                & (target_velocity.norm(dim=-1) <= 1e-8)
            )
            accumulate("displacement_epe", displacement_error)
            accumulate("velocity_epe", velocity_error)
            for prefix, mask in (("wall", wall), ("ball", ball), ("free", free)):
                accumulate(f"{prefix}_displacement_epe", displacement_error, mask)
                accumulate(f"{prefix}_velocity_epe", velocity_error, mask)
            accumulate(
                "stationary_displacement_epe",
                displacement_error,
                stationary,
            )
            accumulate("stationary_velocity_epe", velocity_error, stationary)

        steps = min(rollout_steps, num_steps)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            rollout = model.rollout(
                batch["positions"][:, 0],
                batch["velocities"][:, 0],
                belief,
                steps,
            )
        position_error = (
            rollout["positions"][:, 1:].float()
            - batch["positions"][:, 1 : steps + 1].float()
        ).norm(dim=-1)
        accumulate("rollout_ade", position_error)
        accumulate("rollout_fde", position_error[:, -1])
        stationary_rollout = (
            (batch["velocities"][:, : steps + 1].norm(dim=-1).amax(dim=1) <= 1e-8)
            & ~batch["collision_objects"][:, :steps].any(dim=1)
            & (
                (
                    batch["positions"][:, : steps + 1]
                    - batch["positions"][:, :1]
                )
                .norm(dim=-1)
                .amax(dim=1)
                <= 1e-8
            )
        )
        accumulate(
            "stationary_rollout_ade",
            position_error,
            stationary_rollout[:, None].expand_as(position_error),
        )
        accumulate(
            "stationary_rollout_fde",
            position_error[:, -1],
            stationary_rollout,
        )
        target_occupancy, target_sdf, target_normals = _field_targets(batch, belief)
        accumulate("occupancy_l1", (belief["occupancy_map"] - target_occupancy).abs())
        accumulate("sdf_l1", (belief["sdf_map"] - target_sdf).abs())
        near_boundary = target_sdf.abs() < 3 * model.config.object_radius
        normal_error = 1 - (belief["normal_map"] * target_normals).sum(1, keepdim=True)
        accumulate("normal_cosine_error", normal_error, near_boundary)
        accumulate("context_count", batch["context_count"].float())
    result = {
        name: (
            float(sums[name]) / counts[name]
            if sums[name] is not None and counts[name]
            else 0.0
        )
        for name in names
    }
    result["stationary_interval_count"] = float(
        counts["stationary_velocity_epe"]
    )
    result["stationary_rollout_object_count"] = float(
        counts["stationary_rollout_fde"]
    )
    return result


def save_checkpoint_v4(
    path: Path,
    model: AdaptiveSceneDynamicsModelV4,
    optimizer: torch.optim.Optimizer,
    step: int,
    simulation: SceneSimulationV4Config,
    episode: AdaptiveV4EpisodeConfig,
    *,
    metrics: dict[str, float] | None = None,
    training_state: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "adaptive_scene_billiards_v4",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "dynamics_config": asdict(model.config),
        "simulation_config": asdict(simulation),
        "episode_config": asdict(episode),
        "metrics": metrics or {},
        "training_state": training_state or {},
        "training_config": training_config or {},
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # File fsync + replace still provides an intact checkpoint on
            # platforms/filesystems that do not support directory fsync.
            pass
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint_v4(
    path: Path,
    device: torch.device,
) -> tuple[
    AdaptiveSceneDynamicsModelV4,
    SceneSimulationV4Config,
    AdaptiveV4EpisodeConfig,
    dict[str, Any],
]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "adaptive_scene_billiards_v4":
        raise ValueError("checkpoint is not an adaptive scene billiards V4 checkpoint")
    model = AdaptiveSceneDynamicsModelV4(
        AdaptiveDynamicsV4Config(**checkpoint["dynamics_config"])
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    episode_values = checkpoint["episode_config"].copy()
    episode_values["context_choices"] = tuple(episode_values["context_choices"])
    episode_values["context_choice_weights"] = tuple(
        episode_values["context_choice_weights"]
    )
    return (
        model,
        SceneSimulationV4Config(**checkpoint["simulation_config"]),
        AdaptiveV4EpisodeConfig(**episode_values),
        checkpoint,
    )


__all__ = [
    "evaluate_dynamics_v4",
    "context_evidence_loss_v4",
    "inject_oracle_geometry_v4",
    "load_checkpoint_v4",
    "move_batch",
    "resolve_device",
    "sample_query_windows_v4",
    "save_checkpoint_v4",
    "save_json",
    "scene_field_loss_v4",
    "scene_inputs_v4",
    "seed_everything",
]
