import copy

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from myriad.model import PhysicsRelationBiasMLP, FusedTransformer, FusedTransformerLayer
from train import configure_physics_training, load_init_checkpoint


def metadata(batch=2, lq=5, lk=7, device="cpu"):
    pos_q = torch.randn(batch, lq, 2, device=device)
    pos_k = torch.randn(batch, lk, 2, device=device)
    time_q = torch.arange(lq, device=device).expand(batch, -1).float()
    time_k = torch.arange(lk, device=device).expand(batch, -1).float()
    track_q = torch.arange(lq, device=device).expand(batch, -1).long() % 3
    track_k = torch.arange(lk, device=device).expand(batch, -1).long() % 3
    return pos_q, time_q, track_q, pos_k, time_k, track_k


def test_output_shape_and_finite():
    bias = PhysicsRelationBiasMLP(8)(*metadata())
    assert bias.shape == (2, 8, 5, 7)
    assert torch.isfinite(bias).all()


def test_zero_initialization():
    module = PhysicsRelationBiasMLP(8)
    bias = module(*metadata())
    torch.testing.assert_close(bias, torch.zeros_like(bias), atol=0, rtol=0)
    torch.testing.assert_close(
        module.layer_head_scales,
        torch.full_like(module.layer_head_scales, 0.5),
    )


def test_negative_initial_layer_scale_is_rejected():
    with pytest.raises(ValueError, match="initial_layer_scale"):
        PhysicsRelationBiasMLP(8, initial_layer_scale=-0.1)


def test_chunking_and_checkpoint_are_equivalent():
    full = PhysicsRelationBiasMLP(8, query_chunk_size=32)
    chunked = PhysicsRelationBiasMLP(8, query_chunk_size=2, checkpoint_chunks=True)
    chunked.load_state_dict(full.state_dict())
    with torch.no_grad():
        full.mlp[-1].weight.normal_(std=0.01)
        full.mlp[-1].bias.normal_(std=0.01)
    chunked.load_state_dict(full.state_dict())
    args = metadata()
    full.train(); chunked.train()
    torch.testing.assert_close(full(*args), chunked(*args))


def test_bias_is_bounded_and_layer_head_scales_are_independent():
    module = PhysicsRelationBiasMLP(3, max_abs_bias=0.5, n_bias_layers=4)
    with torch.no_grad():
        module.mlp[-1].weight.normal_(std=10.0)
        module.mlp[-1].bias.normal_(std=10.0)
        module.layer_head_scales.copy_(torch.arange(12).reshape(4, 3))
    bias = module(*metadata())
    assert bias.abs().max() <= 0.5
    layer_zero = module.for_layer(bias, 0)
    layer_one = module.for_layer(bias, 1)
    assert layer_zero.abs().max() <= 0.5
    assert layer_one.abs().max() <= 0.5
    assert not torch.equal(layer_zero, layer_one)


def test_historical_velocity_ignores_query_tokens():
    pos = torch.tensor([[[2.0, 0.0]]])
    time = torch.tensor([[2.0]])
    track = torch.tensor([[0]])
    history_pos = torch.tensor([[[0.0, 0.0], [100.0, 0.0]]])
    history_time = torch.tensor([[0.0, 1.0]])
    history_track = torch.tensor([[0, 0]])
    history_is_query = torch.tensor([[False, True]])
    velocity = PhysicsRelationBiasMLP._historical_velocity(
        pos, time, track, history_pos, history_time, history_track, history_is_query
    )
    torch.testing.assert_close(velocity, torch.tensor([[[1.0, 0.0]]]))


def test_historical_relation_features_do_not_leak_future_positions():
    module = PhysicsRelationBiasMLP(2)
    with torch.no_grad():
        module.mlp[-1].weight.normal_(std=0.1)
    pos_q = torch.tensor([[[0.4, 0.0]]])
    time_q = torch.tensor([[2.0]])
    track_q = torch.tensor([[0]])
    pos_k = torch.tensor([[[0.0, 0.0], [0.2, 0.0], [0.6, 0.0]]])
    time_k = torch.tensor([[0.0, 1.0, 3.0]])
    track_k = torch.tensor([[0, 0, 0]])
    queries_q = torch.tensor([[True]])
    queries_k = torch.zeros_like(track_k, dtype=torch.bool)
    before = module(pos_q, time_q, track_q, pos_k, time_k, track_k, queries_q, queries_k)
    changed = pos_k.clone()
    changed[:, 2] = torch.tensor([[-100.0, 50.0]])
    after = module(pos_q, time_q, track_q, changed, time_k, track_k, queries_q, queries_k)
    torch.testing.assert_close(before[..., :2], after[..., :2])


def test_physics_bias_only_reaches_last_four_layers():
    model = make_transformer(True, depth=6).eval()
    seen = []
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs["physics_bias"]), with_kwargs=True
        )
        for layer in model.mid_level
    ]
    try:
        model(**transformer_inputs(), block_mask=None)
    finally:
        for handle in handles:
            handle.remove()
    assert seen[:2] == [None, None]
    assert len(seen[2:]) == 4 and all(bias is not None for bias in seen[2:])


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_physics_bias_gradient(checkpoint_chunks):
    module = PhysicsRelationBiasMLP(8, query_chunk_size=2, checkpoint_chunks=checkpoint_chunks)
    with torch.no_grad():
        module.mlp[-1].weight.normal_(std=0.01)
    loss = module(*metadata()).square().mean()
    loss.backward()
    grads = [p.grad for p in module.parameters()]
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def make_transformer(use_physics_bias, depth=2):
    model = FusedTransformer(width=48, depth=depth, aux_feat_dim=32, d_head=16, out_mlp_depth=1,
                             ff_expand=2, track_id_embedding=False, use_physics_bias=use_physics_bias)
    for layer in model.mid_level:
        nn.init.normal_(layer.out_proj.weight, std=0.01)
    return model


def transformer_inputs(device="cpu", b=2, motion_l=4):
    return dict(
        x=torch.randn(b, motion_l, 2, device=device),
        x_cross=torch.randn(b, 4, 32, device=device),
        pos=torch.rand(b, motion_l, 2, device=device),
        pos_orig=torch.rand(b, motion_l, 2, device=device),
        pos_cross=torch.rand(b, 2, 2, 2, device=device),
        is_query=torch.zeros(b, motion_l, dtype=torch.bool, device=device),
        track_id=torch.arange(motion_l, device=device).expand(b, -1),
        camera_static=torch.ones(b, dtype=torch.bool, device=device),
        time=torch.arange(motion_l, device=device).expand(b, -1).float(),
        track_id_emb_table=torch.randn(256, 48, device=device),
    )


def test_zero_physics_bias_preserves_transformer_output():
    plain = make_transformer(False).eval()
    physics = make_transformer(True).eval()
    physics.load_state_dict(plain.state_dict(), strict=False)
    args = transformer_inputs()
    out_plain = plain(**args, block_mask=None)
    out_physics = physics(**args, block_mask=None)
    torch.testing.assert_close(out_plain, out_physics, atol=1e-5, rtol=1e-5)


def make_layer(device="cpu"):
    layer = FusedTransformerLayer(48, 16, ff_expand=2).to(device).eval()
    nn.init.normal_(layer.out_proj.weight, std=0.02)
    return layer


def test_bool_mask_cannot_be_overridden_by_large_bias():
    layer = make_layer()
    x = torch.randn(1, 3, 48)
    theta = torch.zeros(1, 3, 3, 3)
    allowed = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    allowed[:, :, 1, 0] = False
    bias = torch.full((1, 3, 3, 3), 100.0)
    out1 = layer(x, theta, scale=None, block_mask=allowed, physics_bias=bias)
    x_changed = x.clone(); x_changed[:, 0] += 1000
    out2 = layer(x_changed, theta, scale=None, block_mask=allowed, physics_bias=bias)
    torch.testing.assert_close(out1[:, 1], out2[:, 1], atol=1e-4, rtol=1e-4)


def test_causal_physics_mask_matches_explicit_bool_mask():
    layer = make_layer()
    x = torch.randn(1, 4, 48)
    theta = torch.zeros(1, 4, 3, 3)
    bias = torch.randn(1, 3, 4, 4) * 0.1
    causal = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    a = layer(x, theta, scale=None, block_mask="causal", physics_bias=bias)
    b = layer(x, theta, scale=None, block_mask=causal, physics_bias=bias)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlexAttention required")
def test_sdpa_and_flexattention_match():
    device = "cuda"
    sdpa = make_layer(device)
    flex = copy.deepcopy(sdpa)
    x = torch.randn(1, 4, 48, device=device)
    theta = torch.zeros(1, 4, 3, 3, device=device)
    bias = torch.randn(1, 3, 4, 4, device=device) * 0.1
    dense = torch.ones(1, 1, 4, 4, dtype=torch.bool, device=device)
    block = create_block_mask(lambda b, h, q, k: q >= k, B=1, H=1, Q_LEN=4, KV_LEN=4, device=device)
    dense = torch.ones_like(dense).tril()
    a = sdpa(x, theta, scale=None, block_mask=dense, physics_bias=bias)
    b = flex(x, theta, scale=None, block_mask=block, physics_bias=bias)
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlexAttention required")
def test_kv_metadata_cache_prefill_and_incremental():
    device = "cuda"
    model = make_transformer(True).to(device).eval()
    model.grow_kv_cache(1, 12)
    prefill = transformer_inputs(device=device, b=1, motion_l=3)
    block = create_block_mask(lambda b, h, q, k: q >= 4, B=1, H=1, Q_LEN=7, KV_LEN=7, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(**prefill, block_mask=block, i_kv=0, compute_cross=True)
    assert model.cached_image_prefix_length == 4
    torch.testing.assert_close(model.physics_pos_cache[:, :3], prefill["pos"])
    inc = transformer_inputs(device=device, b=1, motion_l=2)
    mask = torch.ones(1, 1, 2, 9, dtype=torch.bool, device=device)
    # Reuse the same rollout-level ID table, as predict_simulate does.
    inc["track_id_emb_table"] = prefill["track_id_emb_table"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(**inc, block_mask=mask, i_kv=7, compute_cross=False)
    torch.testing.assert_close(model.physics_pos_cache[:, 3:5], inc["pos"])
    assert model.mid_level[0].kv_cache.k.size(2) == 12


def test_cache_growth_preserves_kv_and_metadata_prefix():
    model = make_transformer(True)
    model.reset_kv_cache(1, 3)
    model.mid_level[0].kv_cache.k.fill_(1.25)
    model.mid_level[0].kv_cache.v.fill_(-2.5)
    model.physics_pos_cache.copy_(torch.arange(6).reshape(1, 3, 2))
    model.physics_time_cache.copy_(torch.arange(3).reshape(1, 3))
    model.physics_track_cache.copy_(torch.tensor([[4, 5, 6]]))
    model.grow_kv_cache(2, 7)
    torch.testing.assert_close(model.mid_level[0].kv_cache.k[:1, :, :3],
                               torch.full_like(model.mid_level[0].kv_cache.k[:1, :, :3], 1.25))
    torch.testing.assert_close(model.mid_level[0].kv_cache.v[:1, :, :3],
                               torch.full_like(model.mid_level[0].kv_cache.v[:1, :, :3], -2.5))
    torch.testing.assert_close(model.physics_pos_cache[:1, :3], torch.arange(6.).reshape(1, 3, 2))
    torch.testing.assert_close(model.physics_time_cache[:1, :3], torch.arange(3.).reshape(1, 3))
    torch.testing.assert_close(model.physics_track_cache[:1, :3], torch.tensor([[4, 5, 6]]))


def test_metadata_cache_is_not_persistent():
    state_keys = make_transformer(True).state_dict().keys()
    assert not any(key.startswith(("physics_pos_cache", "physics_time_cache", "physics_track_cache"))
                   for key in state_keys)


class TinyModel(nn.Module):
    def __init__(self, physics=True):
        super().__init__()
        self.image_embedder = nn.Linear(3, 3)
        self.transformer = make_transformer(physics, depth=4)
        self.distribution_head = nn.Linear(48, 2)


def test_physics_only_freezing():
    model = TinyModel()
    configure_physics_training(model, "physics-only", 1)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("transformer.physics_bias_generator.") for n in trainable)


def test_finetune_freezing():
    model = TinyModel()
    configure_physics_training(model, "finetune", 2)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("transformer.physics_bias_generator.") for n in trainable)
    assert any(n.startswith("transformer.mid_level.2.") for n in trainable)
    assert any(n.startswith("transformer.mid_level.3.") for n in trainable)
    assert not any(n.startswith("transformer.mid_level.1.") for n in trainable)
    assert any(n.startswith("transformer.out_proj.") for n in trainable)
    assert any(n.startswith("distribution_head.") for n in trainable)
    assert not any(n.startswith("image_embedder.") for n in trainable)


def test_old_checkpoint_compatibility(tmp_path):
    old = TinyModel(physics=False)
    new = TinyModel(physics=True)
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict()}, path)
    incompatible = load_init_checkpoint(new, path)
    assert incompatible.missing_keys
    assert all(k.startswith("transformer.physics_bias_generator.") for k in incompatible.missing_keys)
    assert not incompatible.unexpected_keys


def test_old_five_feature_physics_checkpoint_can_initialize(tmp_path):
    old = TinyModel(physics=True)
    state = old.state_dict()
    input_key = "transformer.physics_bias_generator.mlp.0.weight"
    state[input_key] = state[input_key][..., :5].clone()
    del state["transformer.physics_bias_generator.layer_head_scales"]
    path = tmp_path / "old_physics.pt"
    torch.save({"model": state}, path)

    new = TinyModel(physics=True)
    incompatible = load_init_checkpoint(new, path)
    assert incompatible.missing_keys == ["transformer.physics_bias_generator.layer_head_scales"]
    torch.testing.assert_close(
        new.state_dict()[input_key][..., :5], old.state_dict()[input_key][..., :5]
    )
    torch.testing.assert_close(
        new.state_dict()[input_key][..., 5:], torch.zeros_like(new.state_dict()[input_key][..., 5:])
    )


def test_small_model_forward_backward_smoke():
    model = make_transformer(True).train()
    for layer in model.mid_level:
        nn.init.normal_(layer.out_proj.weight, std=0.01)
    with torch.no_grad():
        model.physics_bias_generator.mlp[-1].weight.normal_(std=0.01)
    args = transformer_inputs()
    out = model(**args, block_mask=None)
    assert out.shape == (2, 4, 48) and torch.isfinite(out).all()
    out.square().mean().backward()
    grads = [p.grad for p in model.physics_bias_generator.parameters()]
    assert any(g is not None and torch.count_nonzero(g) for g in grads)
