#!/usr/bin/env python3
"""Reproducible FPT inference/efficiency baseline on the bundled tennis-ball scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from einops import rearrange
from torchvision.utils import flow_to_image


SEED = 20260804
IMAGE_SIZE = 448
POKE_POS = (0.276, 0.721)
POKE_FLOW = (0.120, 0.000)
QUERY_POINTS = (
    POKE_POS,
    (0.220, 0.680),
    (0.320, 0.680),
    (0.220, 0.760),
    (0.320, 0.760),
)


def synchronize() -> None:
    torch.cuda.synchronize()


def timed_cuda(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def summarize_ms(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
        "repeats": len(values),
    }


def make_grid(size: int, device: torch.device) -> torch.Tensor:
    centers = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, size * size, 2)


def point_px(point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    return tuple(round(float(v) * (IMAGE_SIZE - 1)) for v in point)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    source_image = repo / "scripts/myriad_eval/qual_examples/ball_roll.jpg"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    image = Image.open(source_image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    image.save(args.output_dir / "input.png")
    image_np = np.asarray(image).copy()
    image_tensor = rearrange(torch.from_numpy(image_np), "h w c -> 1 c h w").float().div(127.5).sub(1).to(device)

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints/flow_poke_open_set_base.pt"
    checkpoint_sha256 = sha256_file(checkpoint)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    load_start = time.perf_counter()
    model = torch.hub.load(str(repo), "fpt_base", source="local").to(device).eval()
    synchronize()
    model_load_s = time.perf_counter() - load_start

    poke_pos = torch.tensor([[POKE_POS]], dtype=torch.float32, device=device)
    poke_flow = torch.tensor([[POKE_FLOW]], dtype=torch.float32, device=device)
    query_pos = torch.tensor([QUERY_POINTS], dtype=torch.float32, device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embed_times = timed_cuda(lambda: model.embed_image(image_tensor), warmup=3, repeats=20)
        d_img = {key: value.clone() for key, value in model.embed_image(image_tensor).items()}

        point_holder: dict[str, object] = {}

        def infer_points() -> None:
            point_holder["dist"] = model.predict_parallel(poke_pos, poke_flow, query_pos, True, d_img)

        point_times = timed_cuda(infer_points, warmup=5, repeats=50)
        dist = point_holder["dist"]
        predicted_flow = dist.mean.float().cpu().numpy()[0]
        component_probs = dist.mixture_distribution.probs.float().cpu().numpy()[0]
        component_means = dist.component_distribution.loc.float().cpu().numpy()[0]
        component_covariances = dist.component_distribution.covariance_matrix.float().cpu().numpy()[0]

        efficiency = {}
        dense_outputs = {}
        for resolution, repeats in ((32, 20), (64, 10)):
            grid = make_grid(resolution, device)
            holder: dict[str, object] = {}

            def infer_dense() -> None:
                holder["dist"] = model.predict_parallel(poke_pos, poke_flow, grid, True, d_img)

            times = timed_cuda(infer_dense, warmup=3, repeats=repeats)
            efficiency[f"dense_{resolution}x{resolution}"] = {
                **summarize_ms(times),
                "queries": resolution * resolution,
                "queries_per_second": (resolution * resolution) / (statistics.fmean(times) / 1000),
            }
            dense_outputs[resolution] = holder["dist"].mean.float().cpu()

    expected = np.repeat(np.asarray(POKE_FLOW, dtype=np.float32)[None], len(QUERY_POINTS), axis=0)
    errors = predicted_flow - expected
    epe_norm = np.linalg.norm(errors, axis=1)
    epe_px = epe_norm * IMAGE_SIZE
    expected_norm = np.linalg.norm(expected, axis=1)
    predicted_norm = np.linalg.norm(predicted_flow, axis=1)
    cosine = np.sum(predicted_flow * expected, axis=1) / np.maximum(predicted_norm * expected_norm, 1e-12)
    angle_deg = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    start = point_px(POKE_POS)
    target = point_px(np.asarray(POKE_POS) + np.asarray(POKE_FLOW))
    prediction = point_px(np.asarray(POKE_POS) + predicted_flow[0])
    draw.line((start, target), fill=(0, 220, 0), width=6)
    draw.ellipse((target[0] - 7, target[1] - 7, target[0] + 7, target[1] + 7), fill=(0, 220, 0))
    draw.line((start, prediction), fill=(255, 40, 40), width=4)
    draw.ellipse((prediction[0] - 6, prediction[1] - 6, prediction[0] + 6, prediction[1] + 6), fill=(255, 40, 40))
    draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill=(30, 100, 255))
    draw.text((10, 10), "blue=start  green=target  red=predicted mean", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    overlay.save(args.output_dir / "position_overlay.png")

    dense64 = rearrange(dense_outputs[64][0], "(h w) c -> c h w", h=64, w=64)
    dense_vis = rearrange(flow_to_image(dense64), "c h w -> h w c").numpy()
    Image.fromarray(dense_vis).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST).save(
        args.output_dir / "dense_flow_64.png"
    )

    peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3
    metrics = {
        "baseline": "ball_roll_v1",
        "reproducibility": {
            "seed": SEED,
            "repository_commit": commit,
            "checkpoint_sha256": checkpoint_sha256,
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "image_size": IMAGE_SIZE,
            "compile": False,
            "autocast": "bfloat16",
        },
        "scenario": {
            "source_image": str(source_image.relative_to(repo)),
            "poke_position_xy_normalized": POKE_POS,
            "poke_flow_xy_normalized": POKE_FLOW,
            "query_points_xy_normalized": QUERY_POINTS,
            "accuracy_reference": "rigid-translation proxy; not dataset ground truth",
        },
        "position": {
            "predicted_flow_xy_normalized": predicted_flow.tolist(),
            "expected_flow_xy_normalized": expected.tolist(),
            "epe_normalized": epe_norm.tolist(),
            "epe_pixels_at_448": epe_px.tolist(),
            "angle_error_degrees": angle_deg.tolist(),
            "action_point_epe_pixels": float(epe_px[0]),
            "all_ball_points_mean_epe_pixels": float(epe_px.mean()),
        },
        "efficiency": {
            "model_load_seconds_cached": model_load_s,
            "image_embedding": summarize_ms(embed_times),
            "five_point_parallel": {**summarize_ms(point_times), "queries": len(QUERY_POINTS)},
            **efficiency,
            "peak_allocated_vram_gib": peak_vram_gib,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        query_points=np.asarray(QUERY_POINTS),
        expected_flow=expected,
        predicted_flow=predicted_flow,
        component_probs=component_probs,
        component_means=component_means,
        component_covariances=component_covariances,
        dense_flow_32=dense_outputs[32].numpy(),
        dense_flow_64=dense_outputs[64].numpy(),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
