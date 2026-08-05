#!/usr/bin/env python3
"""Test whether the physics-bias path is connected and optimizable.

This is deliberately a tiny, deterministic teacher/student experiment.  It
does not claim that the real billiards objective is good; it isolates the
module and its optimization path from the dataset and rollout objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from myriad.model import FusedTransformer


def make_transformer(use_physics_bias: bool) -> FusedTransformer:
    model = FusedTransformer(
        width=48,
        depth=4,
        aux_feat_dim=32,
        d_head=16,
        out_mlp_depth=1,
        ff_expand=2,
        track_id_embedding=False,
        use_physics_bias=use_physics_bias,
        physics_bias_hidden_dim=32,
        physics_bias_depth=2,
        physics_bias_num_layers=4,
        physics_bias_max_abs=0.5,
    )
    # Transformer residual branches are zero-initialized in the real model.
    # Give this isolated experiment a non-zero, deterministic readout so an
    # attention-score change is observable without training the frozen base.
    with torch.no_grad():
        for layer in model.mid_level:
            nn.init.normal_(layer.out_proj.weight, std=0.02)
    return model


def collision_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """A small deterministic sequence containing two approaching tracks."""
    time_steps = 4
    tracks = 2
    times = torch.arange(time_steps, dtype=torch.float32)
    ball_0 = torch.stack((0.20 + 0.08 * times, torch.full_like(times, 0.50)), dim=-1)
    ball_1 = torch.stack((0.68 - 0.05 * times, torch.full_like(times, 0.50)), dim=-1)
    pos = torch.stack((ball_0, ball_1), dim=1).reshape(1, time_steps * tracks, 2)
    pos = pos.expand(batch_size, -1, -1).contiguous()
    time = times[:, None].expand(-1, tracks).reshape(1, -1).expand(batch_size, -1)
    track_id = (
        torch.arange(tracks).expand(time_steps, -1).reshape(1, -1).expand(batch_size, -1)
    )

    generator = torch.Generator().manual_seed(20260805)
    motion = torch.randn(batch_size, time_steps * tracks, 2, generator=generator)
    cross = torch.randn(batch_size, 4, 32, generator=generator)
    pos_cross = torch.rand(batch_size, 2, 2, 2, generator=generator)
    id_table = torch.randn(256, 48, generator=generator)
    return {
        "x": motion,
        "x_cross": cross,
        "pos": pos,
        "pos_orig": pos[:, :tracks].repeat(1, time_steps, 1),
        "pos_cross": pos_cross,
        "is_query": torch.zeros(batch_size, time_steps * tracks, dtype=torch.bool),
        "track_id": track_id,
        "camera_static": torch.ones(batch_size, dtype=torch.bool),
        "time": time,
        "track_id_emb_table": id_table,
        "block_mask": None,
    }


def checkpoint_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefix = "transformer.physics_bias_generator."
    tensors = {
        key[len(prefix) :]: value.float()
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not tensors:
        raise ValueError(f"Checkpoint contains no physics-bias parameters: {path}")
    final_weight = tensors["mlp.4.weight"]
    final_bias = tensors["mlp.4.bias"]
    scales = tensors["layer_head_scales"]
    return {
        "path": str(path),
        "step": int(checkpoint.get("step", -1)),
        "final_weight_norm": float(final_weight.norm()),
        "final_bias_norm": float(final_bias.norm()),
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std()),
        "scale_max_deviation_from_one": float((scales - 1).abs().max()),
        "scale_max_deviation_from_new_initial_half": float((scales - 0.5).abs().max()),
    }


def diagnose(steps: int, learning_rate: float, checkpoint: Path | None) -> dict:
    torch.manual_seed(20260805)
    inputs = collision_inputs()

    plain = make_transformer(False).eval()
    student = make_transformer(True).eval()
    student.load_state_dict(plain.state_dict(), strict=False)
    teacher = copy.deepcopy(student).eval()
    assert student.physics_bias_generator is not None
    assert teacher.physics_bias_generator is not None

    with torch.no_grad():
        zero_output = student(**inputs)
        student.use_physics_bias = False
        disabled_output = student(**inputs)
        student.use_physics_bias = True

        nn.init.normal_(teacher.physics_bias_generator.mlp[-1].weight, std=0.30)
        nn.init.normal_(teacher.physics_bias_generator.mlp[-1].bias, std=0.10)
        target = teacher(**inputs)

    zero_equivalence_max_abs = float((zero_output - disabled_output).abs().max())
    forced_output_mean_abs_delta = float((target - zero_output).abs().mean())

    student.requires_grad_(False)
    student.physics_bias_generator.requires_grad_(True)
    optimizer = torch.optim.Adam(student.physics_bias_generator.parameters(), lr=learning_rate)

    def loss_value() -> torch.Tensor:
        return torch.nn.functional.mse_loss(student(**inputs), target)

    optimizer.zero_grad(set_to_none=True)
    initial_loss_tensor = loss_value()
    initial_loss_tensor.backward()
    first_step_grad_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.float().square().sum()
                for parameter in student.physics_bias_generator.parameters()
                if parameter.grad is not None
            )
        )
    )
    optimizer.step()
    losses = [float(initial_loss_tensor.detach())]

    for _ in range(1, steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        final_loss = float(loss_value())
        learned_output = student(**inputs)
    initial_loss = losses[0]
    reduction = 1.0 - final_loss / initial_loss if initial_loss > 0 else 0.0

    connected = (
        zero_equivalence_max_abs <= 1e-6
        and forced_output_mean_abs_delta > 1e-5
        and first_step_grad_norm > 0.0
    )
    optimizable = reduction >= 0.90
    return {
        "experiment": {
            "seed": 20260805,
            "steps": steps,
            "learning_rate": learning_rate,
            "trainable_parameters": sum(p.numel() for p in student.physics_bias_generator.parameters()),
        },
        "wiring": {
            "zero_bias_vs_disabled_max_abs": zero_equivalence_max_abs,
            "forced_bias_output_mean_abs_delta": forced_output_mean_abs_delta,
            "first_step_generator_grad_norm": first_step_grad_norm,
            "connected": connected,
        },
        "micro_overfit": {
            "initial_mse": initial_loss,
            "final_mse": final_loss,
            "loss_reduction_fraction": reduction,
            "final_output_mean_abs_error": float((learned_output - target).abs().mean()),
            "optimizable": optimizable,
            "loss_at_quarters": [
                losses[min(len(losses) - 1, i * len(losses) // 4)] for i in range(4)
            ],
        },
        "checkpoint": checkpoint_summary(checkpoint),
        "verdict": (
            "module_connected_and_optimizable"
            if connected and optimizable
            else "module_or_optimization_path_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")

    result = diagnose(args.steps, args.learning_rate, args.checkpoint)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if result["verdict"] != "module_connected_and_optimizable":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
