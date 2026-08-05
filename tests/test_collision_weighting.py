import numpy as np
import pytest
import torch
import billiards

import train
from myriad.data_billiards import (
    collision_loss_weights,
    collision_token_mask_from_velocities,
    expand_collision_window,
    simulate_billiard_game,
)


def test_velocity_changes_map_to_collision_flow_tokens_and_forward_window():
    velocities = np.zeros((6, 2, 2), dtype=np.float32)
    velocities[2:, 0, 0] = 1.0
    velocities[4:, 1, 1] = -2.0

    mask = collision_token_mask_from_velocities(velocities, window_steps=2)

    expected = np.zeros((5, 2), dtype=bool)
    expected[1:4, 0] = True
    expected[3:5, 1] = True
    np.testing.assert_array_equal(mask, expected)


def test_window_expansion_preserves_ball_identity():
    collisions = np.zeros((5, 2), dtype=bool)
    collisions[1, 0] = True
    collisions[3, 1] = True
    expected = np.zeros_like(collisions)
    expected[1:4, 0] = True
    expected[3:5, 1] = True
    np.testing.assert_array_equal(expand_collision_window(collisions, 2), expected)


def test_collision_loss_weights_only_change_selected_tokens():
    mask = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(
        collision_loss_weights(mask, 10.0),
        np.array([[1.0, 10.0], [10.0, 1.0]], dtype=np.float32),
    )


def test_simulator_returns_exact_ball_ball_participants():
    table = billiards.Billiard()
    table.add_ball((0.0, 0.0), (1.0, 0.0), radius=0.5)
    table.add_ball((2.0, 0.0), (-1.0, 0.0), radius=0.5)
    _, _, _, collisions, participants = simulate_billiard_game(
        table, duration=0.8, dt=0.1, return_ball_collision_mask=True
    )
    collision_index = next(index for index, value in enumerate(collisions) if value)
    np.testing.assert_array_equal(participants[collision_index], np.array([True, True]))


def test_collision_mask_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="T \\+ 1"):
        collision_token_mask_from_velocities(np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        collision_token_mask_from_velocities(np.zeros((2, 1, 2), dtype=np.float32), window_steps=-1)


def test_atomic_checkpoint_save_replaces_only_after_success(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.pt"
    train.atomic_torch_save({"step": 1}, destination)
    assert torch.load(destination, weights_only=True)["step"] == 1

    def failing_save(payload, path):
        path.write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(train.torch, "save", failing_save)
    with pytest.raises(RuntimeError, match="disk full"):
        train.atomic_torch_save({"step": 2}, destination)

    assert torch.load(destination, weights_only=True)["step"] == 1
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()
