"""Procedural same-scene episodes for adaptive 2-D collision prediction.

The simulator owns exact vector geometry and collision parameters.  The model
never receives them: its scene prior is a persistently corrupted soft occupancy
mask, and its adaptation evidence consists only of previously observed object
transitions from the same scene.
"""

from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass
from typing import Any, Iterator

import billiards
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


@dataclass(frozen=True)
class SceneSimulationConfig:
    image_size: int = 96
    num_balls: int = 6
    radius: float = 0.035
    dt: float = 0.04
    trajectory_steps: int = 64
    moving_probability: float = 0.55
    min_speed: float = 0.18
    max_speed: float = 0.45
    corner_jitter: float = 0.045
    max_disk_obstacles: int = 1
    disk_probability: float = 0.35
    min_disk_radius: float = 0.055
    max_disk_radius: float = 0.10
    min_restitution: float = 0.72
    max_restitution: float = 1.0
    mask_supersample: int = 2

    def validate(self) -> None:
        if self.image_size < 32 or self.num_balls < 1:
            raise ValueError("image_size must be >= 32 and num_balls positive")
        if not 0 < self.radius < 0.2 or self.dt <= 0 or self.trajectory_steps < 3:
            raise ValueError("radius/dt must be positive and trajectory_steps >= 3")
        if not 0 <= self.moving_probability <= 1:
            raise ValueError("moving_probability must be in [0, 1]")
        if not 0 <= self.min_speed <= self.max_speed:
            raise ValueError("invalid speed range")
        if not 0 <= self.corner_jitter <= 0.12:
            raise ValueError("corner_jitter must be in [0, 0.12]")
        if self.max_disk_obstacles < 0 or not 0 <= self.disk_probability <= 1:
            raise ValueError("invalid disk obstacle configuration")
        if not 0 < self.min_disk_radius <= self.max_disk_radius < 0.25:
            raise ValueError("invalid disk radius range")
        if not 0 < self.min_restitution <= self.max_restitution <= 1:
            raise ValueError("restitution must lie in (0, 1]")
        if self.mask_supersample < 1:
            raise ValueError("mask_supersample must be positive")


@dataclass(frozen=True)
class AdaptiveEpisodeConfig:
    max_context_trajectories: int = 3
    fixed_context_trajectories: int | None = None
    max_context_tokens: int = 128
    mask_max_shift_px: float = 3.0
    mask_morphology_px: int = 2
    mask_hole_probability: float = 0.35
    mask_false_positive_probability: float = 0.25
    mask_noise_std: float = 0.04

    def validate(self) -> None:
        if self.max_context_trajectories < 0 or self.max_context_tokens < 1:
            raise ValueError("context counts must be non-negative/positive")
        if self.fixed_context_trajectories is not None and not (
            0 <= self.fixed_context_trajectories <= self.max_context_trajectories
        ):
            raise ValueError("fixed_context_trajectories is outside configured range")
        if self.mask_max_shift_px < 0 or self.mask_morphology_px < 0:
            raise ValueError("mask corruption magnitudes must be non-negative")
        for probability in (
            self.mask_hole_probability,
            self.mask_false_positive_probability,
        ):
            if not 0 <= probability <= 1:
                raise ValueError("mask corruption probabilities must be in [0, 1]")
        if self.mask_noise_std < 0:
            raise ValueError("mask_noise_std must be non-negative")


@dataclass(frozen=True)
class SceneSpec:
    vertices: np.ndarray
    disk_obstacles: tuple[tuple[float, float, float], ...]
    restitution: float


class RestitutionWall(billiards.InfiniteWall):
    """One-sided infinite wall with scene-specific normal restitution."""

    def __init__(self, start_point, end_point, restitution: float) -> None:
        super().__init__(start_point, end_point, inside="left")
        self.restitution = restitution

    def collide(self, pos, vel, radius):
        del pos, radius
        headway = -np.dot(vel, self._normal)
        if headway <= 0:
            return vel
        return vel + (1 + self.restitution) * headway * self._normal


class RestitutionDisk(billiards.Disk):
    """Circular static obstacle with scene-specific normal restitution."""

    def __init__(self, center, radius: float, restitution: float) -> None:
        super().__init__(center, radius)
        self.restitution = restitution

    def collide(self, pos, vel, radius):
        del radius
        normal = np.asarray(pos) - self.center
        normal /= np.linalg.norm(normal).clip(min=1e-12)
        headway = -np.dot(vel, normal)
        if headway <= 0:
            return vel
        return vel + (1 + self.restitution) * headway * normal


def make_object_colors(num_balls: int) -> np.ndarray:
    colors = []
    golden_ratio = 0.618033988749895
    for object_id in range(num_balls):
        hue = (0.02 + object_id * golden_ratio) % 1.0
        colors.append(colorsys.hsv_to_rgb(hue, 0.78, 0.92))
    return np.asarray(colors, dtype=np.float32)


def _polygon_normals(vertices: np.ndarray) -> np.ndarray:
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _is_valid_convex_polygon(vertices: np.ndarray) -> bool:
    edges = np.roll(vertices, -1, axis=0) - vertices
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool((cross > 0.02).all())


def sample_scene(config: SceneSimulationConfig, seed: int) -> SceneSpec:
    rng = np.random.default_rng(seed)
    base = np.asarray(
        [[0.07, 0.07], [0.93, 0.07], [0.93, 0.93], [0.07, 0.93]],
        dtype=np.float64,
    )
    for _ in range(1_000):
        vertices = base + rng.uniform(
            -config.corner_jitter,
            config.corner_jitter,
            size=(4, 2),
        )
        vertices = np.clip(vertices, 0.015, 0.985)
        if _is_valid_convex_polygon(vertices):
            break
    else:
        raise RuntimeError("failed to sample a convex arena")

    disks: list[tuple[float, float, float]] = []
    for _ in range(config.max_disk_obstacles):
        if rng.random() >= config.disk_probability:
            continue
        radius = float(rng.uniform(config.min_disk_radius, config.max_disk_radius))
        normals = _polygon_normals(vertices)
        for _ in range(1_000):
            center = rng.uniform(0.30, 0.70, size=2)
            wall_clearance = np.sum((center - vertices) * normals, axis=-1)
            separated = all(
                np.linalg.norm(center - np.asarray(existing[:2]))
                >= radius + existing[2] + config.radius * 2
                for existing in disks
            )
            if (wall_clearance > radius + config.radius * 2).all() and separated:
                disks.append((float(center[0]), float(center[1]), radius))
                break

    restitution = float(rng.uniform(config.min_restitution, config.max_restitution))
    return SceneSpec(vertices.astype(np.float32), tuple(disks), restitution)


def _point_is_free(
    point: np.ndarray,
    scene: SceneSpec,
    clearance: float,
) -> bool:
    normals = _polygon_normals(scene.vertices)
    wall_clearance = np.sum((point - scene.vertices) * normals, axis=-1)
    if not (wall_clearance >= clearance).all():
        return False
    return all(
        np.linalg.norm(point - np.asarray((x, y))) >= radius + clearance
        for x, y, radius in scene.disk_obstacles
    )


def _make_simulator(
    config: SceneSimulationConfig,
    scene: SceneSpec,
    rng: np.random.Generator,
) -> billiards.Billiard:
    obstacles: list[Any] = []
    for start, end in zip(scene.vertices, np.roll(scene.vertices, -1, axis=0)):
        obstacles.append(RestitutionWall(start, end, scene.restitution))
    obstacles.extend(
        RestitutionDisk((x, y), radius, scene.restitution)
        for x, y, radius in scene.disk_obstacles
    )
    simulator = billiards.Billiard(obstacles=obstacles)

    positions: list[np.ndarray] = []
    guaranteed_moving_id = int(rng.integers(config.num_balls))
    clearance = config.radius * 1.08
    for object_id in range(config.num_balls):
        for _ in range(20_000):
            position = rng.uniform(0.04, 0.96, size=2)
            separated = all(
                np.linalg.norm(position - other) >= 2.08 * config.radius
                for other in positions
            )
            if _point_is_free(position, scene, clearance) and separated:
                positions.append(position)
                break
        else:
            raise RuntimeError("failed to place non-overlapping balls in scene")

        moving = rng.random() < config.moving_probability or object_id == guaranteed_moving_id
        if moving:
            angle = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(config.min_speed, config.max_speed)
            velocity = speed * np.asarray([np.cos(angle), np.sin(angle)])
        else:
            velocity = np.zeros(2, dtype=np.float64)
        simulator.add_ball(tuple(position), tuple(velocity), radius=config.radius, mass=1.0)
    return simulator


def _event_masks(events: list[Any], num_balls: int) -> tuple[np.ndarray, ...]:
    any_mask = np.zeros(num_balls, dtype=np.bool_)
    ball_mask = np.zeros(num_balls, dtype=np.bool_)
    obstacle_mask = np.zeros(num_balls, dtype=np.bool_)
    for event in events:
        if not isinstance(event, tuple) or len(event) < 3:
            continue
        first, second = event[1], event[2]
        if isinstance(first, (int, np.integer)):
            any_mask[int(first)] = True
        if isinstance(second, (int, np.integer)):
            any_mask[int(second)] = True
            ball_mask[int(first)] = True
            ball_mask[int(second)] = True
        elif isinstance(first, (int, np.integer)):
            obstacle_mask[int(first)] = True
    return any_mask, ball_mask, obstacle_mask


def simulate_trajectory(
    config: SceneSimulationConfig,
    scene: SceneSpec,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    simulator = _make_simulator(config, scene, rng)
    positions = [simulator.balls_position.copy()]
    collision_objects = []
    ball_collision_objects = []
    obstacle_collision_objects = []
    for step in range(config.trajectory_steps):
        events = simulator.evolve((step + 1) * config.dt)
        any_mask, ball_mask, obstacle_mask = _event_masks(events, config.num_balls)
        collision_objects.append(any_mask)
        ball_collision_objects.append(ball_mask)
        obstacle_collision_objects.append(obstacle_mask)
        positions.append(simulator.balls_position.copy())
    positions_array = np.asarray(positions, dtype=np.float32)
    return {
        "positions": positions_array,
        "flows": positions_array[1:] - positions_array[:-1],
        "collision_objects": np.asarray(collision_objects, dtype=np.bool_),
        "ball_collision_objects": np.asarray(ball_collision_objects, dtype=np.bool_),
        "obstacle_collision_objects": np.asarray(obstacle_collision_objects, dtype=np.bool_),
    }


def render_occupancy_mask(
    config: SceneSimulationConfig,
    scene: SceneSpec,
) -> np.ndarray:
    size = config.image_size
    scale = config.mask_supersample
    high_size = size * scale
    coordinates = (np.arange(high_size, dtype=np.float32) + 0.5) / high_size
    grid_y, grid_x = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.stack([grid_x, grid_y], axis=-1)
    normals = _polygon_normals(scene.vertices)
    wall_clearance = np.sum(
        (points[..., None, :] - scene.vertices[None, None])
        * normals[None, None],
        axis=-1,
    )
    free = (wall_clearance >= 0).all(axis=-1)
    for x, y, radius in scene.disk_obstacles:
        free &= (grid_x - x) ** 2 + (grid_y - y) ** 2 >= radius**2
    occupancy = (~free).astype(np.float32)
    if scale > 1:
        occupancy = occupancy.reshape(size, scale, size, scale).mean(axis=(1, 3))
    return occupancy


def corrupt_occupancy_mask(
    clean_mask: np.ndarray,
    config: AdaptiveEpisodeConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = clean_mask.shape
    shift_x, shift_y = rng.uniform(-config.mask_max_shift_px, config.mask_max_shift_px, 2)
    transform = np.asarray([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    noisy = cv2.warpAffine(
        clean_mask,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if config.mask_morphology_px and rng.random() < 0.7:
        radius = int(rng.integers(1, config.mask_morphology_px + 1))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        operation = cv2.MORPH_DILATE if rng.random() < 0.5 else cv2.MORPH_ERODE
        noisy = cv2.morphologyEx(noisy, operation, kernel)
    if rng.random() < config.mask_hole_probability:
        x0 = int(rng.integers(0, max(width - 8, 1)))
        y0 = int(rng.integers(0, max(height - 8, 1)))
        x1 = min(width, x0 + int(rng.integers(4, max(width // 5, 5))))
        y1 = min(height, y0 + int(rng.integers(4, max(height // 5, 5))))
        noisy[y0:y1, x0:x1] *= rng.uniform(0.0, 0.25)
    if rng.random() < config.mask_false_positive_probability:
        center = tuple(int(value) for value in rng.integers(width // 5, 4 * width // 5, 2))
        radius = int(rng.integers(2, max(3, width // 12)))
        cv2.circle(noisy, center, radius, float(rng.uniform(0.65, 1.0)), thickness=-1)
    if config.mask_noise_std:
        noisy += rng.normal(0, config.mask_noise_std, size=noisy.shape).astype(np.float32)
    noisy = cv2.GaussianBlur(noisy, (3, 3), sigmaX=0.6)
    noisy = np.clip(noisy, 0, 1).astype(np.float32)
    confidence = np.clip(2 * np.abs(noisy - 0.5), 0.05, 1.0).astype(np.float32)
    return noisy, confidence


def _select_context_tokens(
    trajectories: list[dict[str, np.ndarray]],
    max_tokens: int,
    seed: int,
) -> dict[str, np.ndarray]:
    positions = np.zeros((max_tokens, 2), dtype=np.float32)
    flows = np.zeros((max_tokens, 2), dtype=np.float32)
    next_flows = np.zeros((max_tokens, 2), dtype=np.float32)
    confidence = np.zeros((max_tokens,), dtype=np.float32)
    valid = np.zeros((max_tokens,), dtype=np.bool_)
    if not trajectories:
        return {
            "context_positions": positions,
            "context_flows": flows,
            "context_next_flows": next_flows,
            "context_confidence": confidence,
            "context_valid": valid,
        }

    all_positions = np.concatenate(
        [trajectory["positions"][1:-1].reshape(-1, 2) for trajectory in trajectories]
    )
    all_flows = np.concatenate(
        [trajectory["flows"][:-1].reshape(-1, 2) for trajectory in trajectories]
    )
    all_next_flows = np.concatenate(
        [trajectory["flows"][1:].reshape(-1, 2) for trajectory in trajectories]
    )
    num_candidates = all_positions.shape[0]
    rng = np.random.default_rng(seed)
    if num_candidates > max_tokens:
        acceleration = np.linalg.norm(all_next_flows - all_flows, axis=-1)
        important_count = min(max(max_tokens // 2, 1), num_candidates)
        important = np.argpartition(acceleration, -important_count)[-important_count:]
        remaining_pool = np.setdiff1d(
            np.arange(num_candidates), important, assume_unique=False
        )
        random_count = max_tokens - important_count
        random_selection = rng.choice(
            remaining_pool,
            size=random_count,
            replace=remaining_pool.size < random_count,
        )
        selection = np.concatenate([important, random_selection])
        rng.shuffle(selection)
    else:
        selection = np.arange(num_candidates)
    count = min(selection.size, max_tokens)
    selection = selection[:count]
    positions[:count] = all_positions[selection]
    flows[:count] = all_flows[selection]
    next_flows[:count] = all_next_flows[selection]
    confidence[:count] = 1.0
    valid[:count] = True
    return {
        "context_positions": positions,
        "context_flows": flows,
        "context_next_flows": next_flows,
        "context_confidence": confidence,
        "context_valid": valid,
    }


def render_scene(
    positions: np.ndarray | torch.Tensor,
    occupancy_mask: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor,
    radius: float,
    image_size: int,
) -> torch.Tensor:
    mask = np.asarray(occupancy_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != (image_size, image_size):
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    free_color = np.asarray([231, 235, 229], dtype=np.float32)
    wall_color = np.asarray([67, 74, 82], dtype=np.float32)
    image = free_color[None, None] * (1 - mask[..., None]) + wall_color[None, None] * mask[..., None]
    image = np.rint(image).astype(np.uint8)
    radius_px = max(1, round(radius * (image_size - 1)))
    for position, color in zip(np.asarray(positions), np.asarray(colors)):
        center = tuple(np.rint(position * (image_size - 1)).astype(np.int32))
        rgb = tuple(int(channel) for channel in np.rint(color * 255))
        cv2.circle(image, center, radius_px, rgb, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, radius_px, (35, 38, 42), thickness=1, lineType=cv2.LINE_AA)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255)


class AdaptiveSceneEpisodeDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Infinite online stream of newly simulated context/query scene episodes.

    This mirrors MYRIAD's billiards input strategy: workers generate samples on
    CPU only when the DataLoader asks for them, and no trajectory is written to
    disk. ``sample_at`` remains available for deterministic validation and
    debugging, while normal training must iterate over this dataset directly.
    """

    def __init__(
        self,
        simulation: SceneSimulationConfig | None = None,
        episode: AdaptiveEpisodeConfig | None = None,
        base_seed: int = 0,
        stream_start_index: int = 0,
    ) -> None:
        self.simulation = simulation or SceneSimulationConfig()
        self.episode = episode or AdaptiveEpisodeConfig()
        self.simulation.validate()
        self.episode.validate()
        if stream_start_index < 0:
            raise ValueError("stream_start_index must be non-negative")
        self.base_seed = int(base_seed)
        self.stream_start_index = int(stream_start_index)
        self.colors = make_object_colors(self.simulation.num_balls)
        # A persistent DataLoader worker owns its own dataset copy. Keeping its
        # cursor here prevents a newly requested iterator from replaying data.
        self._stream_step = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers_per_rank = worker.num_workers if worker is not None else 1
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if not 0 <= rank < world_size:
            raise RuntimeError(f"Invalid distributed rank/world size: {rank}/{world_size}")

        shard_id = rank * workers_per_rank + worker_id
        shard_count = world_size * workers_per_rank
        while True:
            sample_index = self.stream_start_index + shard_id + self._stream_step * shard_count
            self._stream_step += 1
            yield self.sample_at(sample_index)

    def sample_at(self, index: int) -> dict[str, torch.Tensor]:
        """Generate one reproducible sample without advancing the stream."""
        index = int(index)
        if index < 0:
            raise IndexError(index)
        scene_seed = self.base_seed + index
        scene = sample_scene(self.simulation, scene_seed)
        query = simulate_trajectory(
            self.simulation,
            scene,
            seed=71_000_000 + scene_seed,
        )
        context_rng = np.random.default_rng(83_000_000 + scene_seed)
        if self.episode.fixed_context_trajectories is None:
            context_count = int(
                context_rng.integers(self.episode.max_context_trajectories + 1)
            )
        else:
            context_count = self.episode.fixed_context_trajectories
        context_trajectories = [
            simulate_trajectory(
                self.simulation,
                scene,
                seed=97_000_000 + scene_seed * 17 + context_index,
            )
            for context_index in range(context_count)
        ]
        context = _select_context_tokens(
            context_trajectories,
            self.episode.max_context_tokens,
            seed=101_000_000 + scene_seed,
        )
        clean_mask = render_occupancy_mask(self.simulation, scene)
        noisy_mask, confidence = corrupt_occupancy_mask(
            clean_mask,
            self.episode,
            seed=109_000_000 + scene_seed,
        )
        max_disks = self.simulation.max_disk_obstacles
        disks = np.zeros((max_disks, 3), dtype=np.float32)
        disk_exists = np.zeros((max_disks,), dtype=np.bool_)
        for disk_index, disk in enumerate(scene.disk_obstacles):
            disks[disk_index] = disk
            disk_exists[disk_index] = True

        result = {
            "positions": torch.from_numpy(query["positions"]),
            "flows": torch.from_numpy(query["flows"]),
            "collision_objects": torch.from_numpy(query["collision_objects"]),
            "ball_collision_objects": torch.from_numpy(query["ball_collision_objects"]),
            "obstacle_collision_objects": torch.from_numpy(
                query["obstacle_collision_objects"]
            ),
            "object_colors": torch.from_numpy(self.colors.copy()),
            "exists": torch.ones(
                self.simulation.trajectory_steps + 1,
                self.simulation.num_balls,
                dtype=torch.bool,
            ),
            "clean_mask": torch.from_numpy(clean_mask).unsqueeze(0),
            "noisy_mask": torch.from_numpy(noisy_mask).unsqueeze(0),
            "mask_confidence": torch.from_numpy(confidence).unsqueeze(0),
            "scene_vertices": torch.from_numpy(scene.vertices.copy()),
            "scene_disks": torch.from_numpy(disks),
            "scene_disk_exists": torch.from_numpy(disk_exists),
            "scene_restitution": torch.tensor(scene.restitution, dtype=torch.float32),
            "context_count": torch.tensor(context_count, dtype=torch.long),
        }
        result.update(
            {key: torch.from_numpy(value) for key, value in context.items()}
        )
        return result


class DeterministicAdaptiveSceneEpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    """Finite map-style view for reproducible validation and export only."""

    def __init__(
        self,
        simulation: SceneSimulationConfig | None = None,
        episode: AdaptiveEpisodeConfig | None = None,
        num_samples: int = 100_000,
        base_seed: int = 0,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.num_samples = int(num_samples)
        self.stream = AdaptiveSceneEpisodeDataset(
            simulation=simulation,
            episode=episode,
            base_seed=base_seed,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        return self.stream.sample_at(index)


def make_online_dataloader(
    dataset: AdaptiveSceneEpisodeDataset,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Build the MYRIAD-style online training loader with safe defaults."""
    if not isinstance(dataset, AdaptiveSceneEpisodeDataset):
        raise TypeError("make_online_dataloader expects AdaptiveSceneEpisodeDataset")
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("batch_size/prefetch_factor must be positive and num_workers non-negative")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(**kwargs)
