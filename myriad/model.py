import os
import torch
import math
import einops

from contextlib import nullcontext
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
from torch.distributions import Distribution
from torch.distributions.mixture_same_family import MixtureSameFamily as _MixtureSameFamily
from torch.distributions.categorical import Categorical as _Categorical
from torch.distributions.multivariate_normal import MultivariateNormal as _MultivariateNormal
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Any, Literal
from abc import ABC, abstractmethod
from jaxtyping import Float, Bool, Int
from tqdm import trange
from functools import partial

from .dinov3 import DinoFeatureExtractor


def physics_flex_attention(q, k, v, physics_bias, block_mask, q_motion_offset, k_motion_offset, scale):
    """Apply physics score bias with eager FlexAttention."""
    lq_motion, lk_motion = physics_bias.shape[-2:]

    def score_mod(score, batch, head, q_idx, k_idx):
        q_motion_idx = q_idx - q_motion_offset
        k_motion_idx = k_idx - k_motion_offset
        valid = ((q_motion_idx >= 0) & (k_motion_idx >= 0)
                 & (q_motion_idx < lq_motion) & (k_motion_idx < lk_motion))
        q_safe = q_motion_idx.clamp(0, lq_motion - 1)
        k_safe = k_motion_idx.clamp(0, lk_motion - 1)
        value = physics_bias[batch, head, q_safe, k_safe].to(score.dtype)
        return score + torch.where(valid, value, torch.zeros_like(score))

    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale)


# ---------------------------------------------------------------------------------------------------------------------
# Helper Modules & Utilities
# ---------------------------------------------------------------------------------------------------------------------

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def scale_for_cosine_sim(q, k, scale, eps):
    dtype = torch.promote_types(q.dtype, torch.float32)
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)

def squareplus(x: torch.Tensor) -> torch.Tensor:
    # See https://x.com/jon_barron/status/1387167648669048833
    return (x + torch.sqrt(x**2 + 4)) / 2

class ResidualWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x) + x

class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x

# ---------------------------------------------------------------------------------------------------------------------
# Attention Masks
# ---------------------------------------------------------------------------------------------------------------------

def query_causal_mask_mod(l_prefix: int, l_seq: int, n_query: int = 1):
    """
    Seq: [img_tokens, pokes, queries], len(pokes) = l_seq, len(queries) = n_query * l_seq
    n_query queries attend to the same set of pokes each, not to each other
    Pokes attend causally to each other, but not to queries
    First query set attends to no pokes, next set to the first poke, then first two pokes, etc
    Prefix (image) tokens attend to nothing, but everything attends to them
    """

    def mask_mod(batch, head, q_idx, kv_idx):
        q_idx_np, kv_idx_np = q_idx - l_prefix, kv_idx - l_prefix
        return (  # Prefix (image) tokens don't attend to anything
            q_idx >= l_prefix
        ) & (  # This part is identical to the FPT query-causal attention mask, just shifted by l_prefix
            ((q_idx_np >= kv_idx_np) & (q_idx_np < l_seq))  # Normal causal part
            | (q_idx_np == kv_idx_np)  # Diagonal
            | (
                ((q_idx_np - l_seq - n_query) // n_query >= kv_idx_np) & (q_idx_np >= l_seq) & (kv_idx_np < l_seq)
            )  # Query heads
        )

    return mask_mod


def inference_query_causal_mask_mod(l_prefix: int, l_seq: int):
    """
    Seq: [img_tokens, pokes, queries], len(pokes) = l_seq, len(queries) = *
    Pokes attend causally to each other, but not to queries
    Queries attend to all pokes, but not to each other
    Prefix (image) tokens attend to nothing, but everything attends to them
    """

    def mask_mod(batch, head, q_idx, kv_idx):
        q_idx_np, kv_idx_np = q_idx - l_prefix, kv_idx - l_prefix
        return (  # Prefix (image) tokens don't attend to anything
            q_idx >= l_prefix
        ) & (  # This part is identical to the FPT inference attention mask, just shifted by l_prefix
            (q_idx_np == kv_idx_np)  # Diagonal
            | ((q_idx_np >= kv_idx_np) & (q_idx_np < l_seq))  # Normal causal part
            | ((kv_idx_np < l_seq) & (q_idx_np >= l_seq))  # Queries
        )

    return mask_mod

# ---------------------------------------------------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------------------------------------------------

class KVCache(nn.Module):
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.register_buffer("k", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v", torch.zeros(shape, dtype=dtype, device=device), persistent=False)

# ---------------------------------------------------------------------------------------------------------------------
# Fourier Feature Embedding Module
# ---------------------------------------------------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        target_dtype = input.dtype
        input = input.to(self.weight.dtype)
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1).to(target_dtype)

# ---------------------------------------------------------------------------------------------------------------------
# RoPE and Positional Embedding Utilities
# ---------------------------------------------------------------------------------------------------------------------

def bounding_box(
    h: int,
    w: int,
    pixel_aspect_ratio: float = 1.0,
) -> tuple[float, float, float, float]:
    w_adj = w
    h_adh = h * pixel_aspect_ratio

    ar_adj = w_adj / h_adh

    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1.0:
        y_min, y_max = -1.0 / ar_adj, 1.0 / ar_adj
    elif ar_adj < 1.0:
        x_min, x_max = -ar_adj, ar_adj
    return y_min, y_max, x_min, x_max

def centers(
    start: float,
    end: float,
    num: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
):
    edges = torch.linspace(start, end, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2.0

def make_grid(
    h_pos: Float[torch.Tensor, "(h)"],
    w_pos: Float[torch.Tensor, "(w)"],
) -> Float[torch.Tensor, "(h w) 2"]:
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)

def make_axial_pos_2d(
    h: int,
    w: int,
    pixel_aspect_ratio: float = 1.0,
    align_corners: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    relative_pos: bool = False,
    scale_to_unit: bool = True,
) -> Float[torch.Tensor, "(h w) 2"]:
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2
    
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)

    grid = make_grid(h_pos, w_pos)
    if scale_to_unit:
        scale_y = 1.0 / h
        scale_x = 1.0 / w
        scale_tens = torch.tensor([scale_x, scale_y], device=grid.device, dtype=grid.dtype)
        return grid.flip(-1).mul(scale_tens).add(0.5)
    else:
        return grid.flip(-1).mul(0.5).add(0.5)

def apply_rotary_emb(
    x: Float[torch.Tensor, "..."],
    theta: Float[torch.Tensor, "..."],
    conj: bool=False
) -> Float[torch.Tensor, "..."]:
    out_dtype = x.dtype
    dtype = torch.promote_types(x.dtype, theta.dtype)
    theta = theta.to(dtype)
    x = x.to(dtype)

    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1], "Rotary embedding dimension must be less than or equal to half of head dimension"
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat([y1, y2, x3.to(out_dtype)], dim=-1)

class FPRoPE(nn.Module):

    def __init__(
        self,
        d_head: int,
        n_heads: int,
        theta_t: float = 100.0,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

        min_freq = math.pi
        max_freq = 10.0 * math.pi
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)

        freqs = torch.stack(
            [torch.linspace(log_min, log_max, n_heads * (d_head // 12) + 1)[:-1].exp()] * 2
        )
        self.freqs = nn.Parameter(freqs.view(2, d_head // 12, n_heads).mT.contiguous(), requires_grad=False)

        t_freqs = theta_t ** (-torch.arange(0, n_heads * (d_head // 12)).float() / (n_heads * (d_head // 12)))
        self.t_freqs = nn.Parameter(t_freqs.view(d_head // 12, n_heads).T.contiguous(), requires_grad=False)


    def apply_emb(self, x, theta):
        return apply_rotary_emb(x, theta)
    

    def forward(
            self,
            pos: Float[torch.Tensor, "... 2"],
            pos_orig: Float[torch.Tensor, "... 2"],
            time: Float[torch.Tensor, "..."],
            **kwargs
        ): 
        dtype = torch.promote_types(pos.dtype, torch.float32)
        pos_to_use = torch.stack([pos_orig, pos], dim=2).to(dtype)
        pos_to_use = pos_to_use.mul(2.0).sub(1.0)
        theta_w = pos_to_use[..., None, 0:1] * self.freqs[0].to(dtype)
        theta_h = pos_to_use[..., None, 1:2] * self.freqs[1].to(dtype)
        theta_t = time[..., None, None, None].to(dtype) * self.t_freqs.to(dtype)
        theta = torch.cat([theta_h, theta_w, theta_t], dim=2)
        theta = einops.rearrange(theta, "b l c n_heads d_theta -> b l n_heads (c d_theta)")
        return theta.to(dtype)

# ---------------------------------------------------------------------------------------------------------------------
# (Adaptive) Normalizations
# ---------------------------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.scale = nn.Parameter(torch.ones(self.shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        dtype = torch.promote_types(x.dtype, torch.float32)
        mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
        scale = self.scale.to(dtype) * torch.rsqrt(mean_sq + self.eps)
        return x * scale.to(x.dtype)
    

class AdaRMSNorm(nn.Module):
    
    def __init__(self, features: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.features = features
    
    def extra_repr(self):
        return f"eps={self.eps},"
    
    def forward(self, x: Float[torch.Tensor, "... l d"], scale: Float[torch.Tensor, "... d"]):
        assert scale.size(-1) == x.size(-1), f"Condition features {scale.size(-1)} must match input features {x.size(-1)}"
        # print(f"[DEBUG] scaling with scale of shape {scale.shape=} and mean {scale.mean().item()}")
        
        dtype = x.dtype
        x = F.rms_norm(x.float(), (self.features,), eps=self.eps)
        x = x * (1 + scale[..., None, :].to(x.dtype))
        return x.to(dtype)


class AdaLN(nn.Module):
    def __init__(self, shape, cond_features: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.linear = zero_init(nn.Linear(cond_features, math.prod(self.shape) * 2, bias=True))

    def forward(self, x, cond):
        d = torch.promote_types(x.dtype, torch.float32)
        scale, shift = einops.rearrange(self.linear(cond), "... (n d) -> n ... d", n=2)
        return F.layer_norm(x.to(d), self.shape, weight=None, bias=None, eps=self.eps).to(x.dtype) * (1 + scale) + shift

def _layer_norm_wrapper(x, norm_module, norm_dtype, out_dtype):
    x_fp32 = x.to(norm_dtype)
    w = norm_module.weight.to(norm_dtype) if norm_module.weight is not None else None
    b = norm_module.bias.to(norm_dtype) if norm_module.bias is not None else None
    x = F.layer_norm(x_fp32, norm_module.normalized_shape, w, b, norm_module.eps).to(out_dtype)
    return x

# ---------------------------------------------------------------------------------------------------------------------
# Flow Poke Embedder Module
# ---------------------------------------------------------------------------------------------------------------------

def embedder_mlp(in_features: int, out_features: int, depth: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        *(
            ResidualWrapper(
                nn.Sequential(
                    RMSNorm(out_features),
                    nn.Linear(out_features, out_features, bias=False),
                    nn.SiLU(),
                    nn.Linear(out_features, out_features, bias=False),
                )
            )
            for _ in range(depth)
        ),
    )

def mask_pad(
    mask: Bool[torch.Tensor, "..."],
    raw: Float[torch.Tensor, "..."],
    replace: Float[torch.Tensor, "..."],
    rearrange_mask: str,
    rearrange_raw: str,
    rearrange_replace: str,
) -> Float[torch.Tensor, "..."]:
    return torch.where(
        einops.rearrange(mask, rearrange_mask),
        einops.rearrange(replace, rearrange_replace),
        einops.rearrange(raw, rearrange_raw),
    )

class FlowPokeEmbedder(nn.Module):

    def __init__(
        self,
        out_features: int,
        input_scale: float = 0.5,
        aux_feature_dim: int = 0,
        depth: int = 2,
        track_id_embedding: bool = True,
        ada_norm_size: int = -1,
        max_num_track_ids: int = 256,
    ):
        super().__init__()

        self.in_features = 2   # must be 2D flow vectors
        self.out_features = out_features
        self.input_scale = input_scale
        self.aux_feature_dim = aux_feature_dim
        self.track_id_embedding = track_id_embedding
        self.max_num_track_ids = max_num_track_ids
        self.register_buffer(
            "coeffs", einops.rearrange(2 * torch.pi * (torch.arange(self.out_features // 2)+1), "f -> 1 1 f")
        )

        self.future_past_dropout_token = nn.Parameter(torch.randn((self.out_features*2,)), requires_grad=True)
        self.query_token = nn.Parameter(torch.randn(self.out_features*2), requires_grad=True)
        nn.init.trunc_normal_(self.query_token)

        input_width = (
            self.out_features * 2 + 2 * self.aux_feature_dim + self.out_features * int(self.track_id_embedding)
        )
        width = self.out_features
        self.mlp = nn.Sequential(
            nn.Linear(input_width, width, bias=False),
            *[
                ResidualWrapper(
                    nn.Sequential(
                        RMSNorm(width),
                        nn.Linear(width, width * 4, bias=False),
                        nn.GELU(approximate="tanh"),
                        zero_init(nn.Linear(width * 4, width, bias=False)),
                    )
                )
                for _ in range(depth)
            ],
        )

        self.camera_static_emb = nn.Identity() if ada_norm_size <= 0 else nn.Embedding(2, ada_norm_size)

    def _fourier_emb_flow(self, flow: Float[torch.Tensor, "b l 2"]) -> Float[torch.Tensor, "b l f"]:
        B, L, D = flow.shape
        assert D == 2, "Flow input must have last dimension of size 2"

        flow = (torch.tanh(self.input_scale * flow) + 1.0) / 2.0  # scale to [0, 1]
        delta_x, delta_y = flow.unbind(dim=-1)
        delta_x = einops.rearrange(delta_x, "b l -> b l 1")
        delta_y = einops.rearrange(delta_y, "b l -> b l 1")

        fourier_emb = torch.cat(
            [
            torch.sin(self.coeffs * delta_x),
            torch.cos(self.coeffs * delta_x),
            torch.sin(self.coeffs * delta_y),
            torch.cos(self.coeffs * delta_y),
            ],
            dim=-1
        ).to(self.future_past_dropout_token.dtype)
        return fourier_emb
    
    def embed_camera_static(self, camera_static: Bool[torch.Tensor, "b"]):
        camera_static = camera_static.to(torch.long)
        emb: Float[torch.Tensor, "b f"] = self.camera_static_emb(camera_static)
        return emb

    def get_id_embedding_table(self, device: torch.device, dtype: torch.dtype, n_tracks: int=None) -> Float[torch.Tensor, "num_ids f"]:
        with torch.no_grad():
            n_tracks = self.max_num_track_ids if n_tracks is None else n_tracks
            id_emb_table = torch.randn((n_tracks, self.out_features), device=device, dtype=dtype)
            id_emb_table = F.normalize(id_emb_table, dim=-1)
        return id_emb_table

    def forward(
        self,
        x: Float[torch.Tensor, "b l 2"],
        pos: Float[torch.Tensor, "b l 2"],
        pos_orig: Float[torch.Tensor, "b l 2"],
        is_query: Bool[torch.Tensor, "b l"],
        aux_feats: Float[torch.Tensor, "b d_aux h w"],
        track_id: Int[torch.Tensor, "b l"],
        id_emb_table: Float[torch.Tensor, "num_ids f"] | None = None,
        **kwargs,
    ):
        # aux_feats: Float[torch.Tensor, "b h w d_aux"] = einops.rearrange(aux_feats, "b d_aux h w -> b h w d_aux")
        flow_embedding: Float[torch.Tensor, "b l f"] = self._fourier_emb_flow(x)
        # print("[DEBUG] using is_query instead of ~is_query", flush=True)
        flow_embedding = mask_pad(
            is_query,
            flow_embedding,
            self.query_token,
            "B L -> B L 1",
            "... -> ...",
            "C -> 1 1 C"
        )

        if self.aux_feature_dim > 0:
            # get image features at the current and original positions
            aux_feats_ = F.grid_sample(
                aux_feats,
                torch.cat([pos, pos_orig], dim=1)[:, :, None, :].mul(2.0).sub(1.0),
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            )
            aux_feats_ = einops.rearrange(aux_feats_, "b d_aux (c l) 1 -> b l (c d_aux)", c=2)
            flow_embedding = torch.cat([flow_embedding, aux_feats_], dim=-1)
        if self.track_id_embedding:
            if id_emb_table is None:
                id_emb_table = self.get_id_embedding_table(device=x.device, dtype=x.dtype)
            track_id_emb = id_emb_table[track_id]
            flow_embedding = torch.cat([flow_embedding, track_id_emb], dim=-1)
        result: Float[torch.Tensor, "b l out_features"] = self.mlp(flow_embedding)
        return result, pos

# ---------------------------------------------------------------------------------------------------------------------
# Flow Poke Output Module
# ---------------------------------------------------------------------------------------------------------------------

class FlowPokeOutput(nn.Module):

    def __init__(
        self,
        width: int,
        depth: int = 2,
    ):
        super().__init__()

        self.in_norm = RMSNorm(width, eps=1e-6)
        self.out_norm = RMSNorm(width, eps=1e-6)
        layer = lambda: nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.mlp = nn.Sequential(
            *[ResidualWrapper(layer()) for _ in range(depth)],
        )
    
    def forward(self, x: Float[torch.Tensor, "... width"]) -> Float[torch.Tensor, "... width"]:
        return self.out_norm(self.mlp(self.in_norm(x)))


# ---------------------------------------------------------------------------------------------------------------------
# Fused Flow Poke Transformer
# ---------------------------------------------------------------------------------------------------------------------

class PhysicsRelationBiasMLP(nn.Module):
    """Per-head attention bias built only from current and historical positions."""

    def __init__(
        self,
        n_heads: int,
        hidden_dim: int = 64,
        depth: int = 2,
        time_scale: float = 50.0,
        max_abs_bias: float = 1.0,
        ball_radius: float = 0.033,
        n_bias_layers: int = 4,
        initial_layer_scale: float = 0.5,
        history_window: int = 8,
        history_decay: float = 4.0,
        query_chunk_size: int = 64,
        checkpoint_chunks: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if query_chunk_size < 1:
            raise ValueError(f"query_chunk_size must be >= 1, got {query_chunk_size}")
        if time_scale <= 0 or max_abs_bias <= 0 or ball_radius <= 0:
            raise ValueError("time_scale, max_abs_bias, and ball_radius must be positive")
        if n_bias_layers < 1:
            raise ValueError(f"n_bias_layers must be >= 1, got {n_bias_layers}")
        if initial_layer_scale < 0:
            raise ValueError("initial_layer_scale must be non-negative")
        if history_window < 2:
            raise ValueError("history_window must be at least 2")
        if history_decay <= 0:
            raise ValueError("history_decay must be positive")
        self.n_heads = n_heads
        self.time_scale = float(time_scale)
        self.max_abs_bias = float(max_abs_bias)
        self.ball_radius = float(ball_radius)
        self.n_bias_layers = n_bias_layers
        self.history_window = int(history_window)
        self.history_decay = float(history_decay)
        self.query_chunk_size = query_chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        # Runtime-only ablation control. It deliberately does not enter the
        # state dict, so existing 11- and 14-feature checkpoints remain loadable.
        self.kinematics_mode = "long"
        self.collision_long_history_distance = 0.04
        self.collision_long_history_temperature = 0.008

        # Keep the original 11 relation inputs first, then append causal
        # long-history acceleration (2) and radial closing acceleration.
        layers: list[nn.Module] = [nn.Linear(14, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(zero_init(nn.Linear(hidden_dim, n_heads)))
        self.mlp = nn.Sequential(*layers)
        # The physical feature encoder is shared, but every affected layer and
        # attention head can independently gate its contribution.
        self.layer_head_scales = nn.Parameter(
            torch.full((n_bias_layers, n_heads), float(initial_layer_scale))
        )

    @staticmethod
    def _historical_kinematics(
        pos: torch.Tensor,
        time: torch.Tensor,
        track: torch.Tensor,
        history_pos: torch.Tensor,
        history_time: torch.Tensor,
        history_track: torch.Tensor,
        history_is_query: torch.Tensor,
        history_window: int = 8,
        history_decay: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit causal velocity/acceleration from recent same-track positions."""
        same_track = track[:, :, None] == history_track[:, None, :]
        earlier = history_time[:, None, :] < time[:, :, None]
        observed = ~history_is_query[:, None, :]
        valid = same_track & earlier & observed

        candidate_time = history_time[:, None, :].expand(-1, pos.size(1), -1)
        candidate_time = candidate_time.masked_fill(~valid, float("-inf"))
        window = min(history_window, history_pos.size(1))
        selected_time, selected_idx = candidate_time.topk(window, dim=-1)
        selected_valid = torch.isfinite(selected_time)
        expanded_history = history_pos[:, None].expand(-1, pos.size(1), -1, -1)
        selected_pos = torch.gather(
            expanded_history, 2, selected_idx[..., None].expand(-1, -1, -1, 2)
        )

        # Anchor a weighted quadratic at the current position:
        # p(t + dt) - p(t) = velocity * dt + 0.5 * acceleration * dt^2.
        # Recent samples receive more weight so collision discontinuities do not
        # get averaged over the entire rollout.
        fit_dtype = torch.float32
        dt = (selected_time - time[..., None]).to(fit_dtype)
        dt = torch.where(selected_valid, dt, torch.zeros_like(dt))
        y = (selected_pos - pos[:, :, None]).to(fit_dtype)
        weights = torch.exp(dt / history_decay) * selected_valid.to(fit_dtype)
        x1 = dt
        x2 = 0.5 * dt.square()
        a11 = (weights * x1.square()).sum(dim=-1)
        a12 = (weights * x1 * x2).sum(dim=-1)
        a22 = (weights * x2.square()).sum(dim=-1)
        b1 = (weights[..., None] * x1[..., None] * y).sum(dim=-2)
        b2 = (weights[..., None] * x2[..., None] * y).sum(dim=-2)
        determinant = a11 * a22 - a12.square()
        stable = (selected_valid.sum(dim=-1) >= 2) & (determinant.abs() > 1e-8)
        safe_determinant = determinant.masked_fill(~stable, 1.0)
        velocity_fit = (b1 * a22[..., None] - b2 * a12[..., None]) / safe_determinant[..., None]
        acceleration_fit = (b2 * a11[..., None] - b1 * a12[..., None]) / safe_determinant[..., None]

        latest_valid = selected_valid[..., 0]
        latest_dt = (time - selected_time[..., 0]).clamp_min(1e-6)
        latest_velocity = (pos - selected_pos[..., 0, :]) / latest_dt[..., None]
        latest_velocity = torch.where(latest_valid[..., None], latest_velocity, torch.zeros_like(pos))
        velocity = torch.where(stable[..., None], velocity_fit.to(pos.dtype), latest_velocity)
        acceleration = torch.where(
            stable[..., None], acceleration_fit.to(pos.dtype), torch.zeros_like(pos)
        )
        return velocity.clamp(-1.0, 1.0), acceleration.clamp(-1.0, 1.0)

    @staticmethod
    def _historical_velocity(
        pos: torch.Tensor,
        time: torch.Tensor,
        track: torch.Tensor,
        history_pos: torch.Tensor,
        history_time: torch.Tensor,
        history_track: torch.Tensor,
        history_is_query: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compatible velocity-only view of the history estimator."""
        velocity, _ = PhysicsRelationBiasMLP._historical_kinematics(
            pos, time, track, history_pos, history_time, history_track, history_is_query
        )
        return velocity

    @staticmethod
    def _latest_velocity(
        pos: torch.Tensor,
        time: torch.Tensor,
        track: torch.Tensor,
        history_pos: torch.Tensor,
        history_time: torch.Tensor,
        history_track: torch.Tensor,
        history_is_query: torch.Tensor,
    ) -> torch.Tensor:
        """Finite-difference velocity from only the latest causal observation."""
        same_track = track[:, :, None] == history_track[:, None, :]
        earlier = history_time[:, None, :] < time[:, :, None]
        observed = ~history_is_query[:, None, :]
        valid = same_track & earlier & observed
        candidate_time = history_time[:, None, :].expand(-1, pos.size(1), -1)
        candidate_time = candidate_time.masked_fill(~valid, float("-inf"))
        previous_time, previous_idx = candidate_time.max(dim=-1)
        has_history = valid.any(dim=-1)
        expanded_history = history_pos[:, None].expand(-1, pos.size(1), -1, -1)
        previous_pos = torch.gather(
            expanded_history, 2, previous_idx[..., None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        dt = (time - previous_time).clamp_min(1e-6)
        velocity = (pos - previous_pos) / dt[..., None]
        velocity = torch.where(has_history[..., None], velocity, torch.zeros_like(velocity))
        return velocity.clamp(-1.0, 1.0)

    def set_kinematics_mode(
        self,
        mode: str,
        collision_distance: float | None = None,
        collision_temperature: float | None = None,
    ) -> None:
        """Select long, short, or collision-local long-history features."""
        if mode not in {"long", "short", "collision-gated", "collision-smooth"}:
            raise ValueError(f"Unknown physics kinematics mode: {mode}")
        if collision_distance is not None:
            if collision_distance <= 0:
                raise ValueError("collision_distance must be positive")
            self.collision_long_history_distance = float(collision_distance)
        if collision_temperature is not None:
            if collision_temperature <= 0:
                raise ValueError("collision_temperature must be positive")
            self.collision_long_history_temperature = float(collision_temperature)
        self.kinematics_mode = mode

    def _long_history_gate(
        self,
        surface_distance: torch.Tensor,
        same_track: torch.Tensor,
    ) -> torch.Tensor:
        different_track = (~same_track.bool()).to(surface_distance.dtype)
        if self.kinematics_mode == "collision-gated":
            return different_track * (
                surface_distance.abs() <= self.collision_long_history_distance
            ).to(surface_distance.dtype)
        if self.kinematics_mode == "collision-smooth":
            return different_track * torch.sigmoid(
                (self.collision_long_history_distance - surface_distance.abs())
                / self.collision_long_history_temperature
            )
        return torch.ones_like(surface_distance)

    def _chunk(
        self,
        pos_q: torch.Tensor,
        time_q: torch.Tensor,
        track_q: torch.Tensor,
        pos_k: torch.Tensor,
        time_k: torch.Tensor,
        track_k: torch.Tensor,
        velocity_q: torch.Tensor,
        acceleration_q: torch.Tensor,
        velocity_k: torch.Tensor,
        acceleration_k: torch.Tensor,
        latest_velocity_q: torch.Tensor,
        latest_velocity_k: torch.Tensor,
        is_query_q: torch.Tensor,
        is_query_k: torch.Tensor,
    ) -> torch.Tensor:
        rel_pos = pos_k[:, None, :, :] - pos_q[:, :, None, :]
        distance = torch.linalg.vector_norm(rel_pos, dim=-1, keepdim=True)
        delta_time = (time_q[:, :, None] - time_k[:, None, :]).unsqueeze(-1)
        delta_time = torch.clamp(delta_time / self.time_scale, min=-1.0, max=1.0)
        same_track = (track_q[:, :, None] == track_k[:, None, :]).unsqueeze(-1).to(rel_pos.dtype)
        long_rel_velocity = velocity_k[:, None, :, :] - velocity_q[:, :, None, :]
        rel_acceleration = acceleration_k[:, None, :, :] - acceleration_q[:, :, None, :]
        short_rel_velocity = latest_velocity_k[:, None, :, :] - latest_velocity_q[:, :, None, :]
        surface_distance = distance - 2.0 * self.ball_radius
        if self.kinematics_mode == "short":
            rel_velocity = short_rel_velocity
            rel_acceleration = torch.zeros_like(rel_acceleration)
        elif self.kinematics_mode in {"collision-gated", "collision-smooth"}:
            # Geometry provides a causal proxy for a short collision window.
            # Long-history features are used only for different balls whose
            # surfaces are close; all other relations use the latest step.
            gate = self._long_history_gate(surface_distance, same_track)
            rel_velocity = short_rel_velocity + gate * (long_rel_velocity - short_rel_velocity)
            rel_acceleration = gate * rel_acceleration
        else:
            rel_velocity = long_rel_velocity
        radial_velocity = (rel_pos * rel_velocity).sum(dim=-1, keepdim=True)
        closing_speed = -radial_velocity / distance.clamp_min(1e-6)
        relative_speed_sq = rel_velocity.square().sum(dim=-1, keepdim=True)
        t_closest = -radial_velocity / relative_speed_sq.clamp_min(1e-8)
        t_closest = torch.clamp(t_closest / self.time_scale, min=-1.0, max=1.0)
        approaching = (closing_speed > 0).to(rel_pos.dtype)
        closing_acceleration = -(rel_pos * rel_acceleration).sum(dim=-1, keepdim=True) / distance.clamp_min(1e-6)
        features = torch.cat([
            rel_pos, distance, delta_time.to(rel_pos.dtype), same_track,
            rel_velocity, closing_speed, surface_distance, t_closest, approaching,
            rel_acceleration, closing_acceleration,
        ], dim=-1)
        assert features.shape[-1] == 14
        raw_bias = self.mlp(features)
        bias = self.max_abs_bias * torch.tanh(raw_bias / self.max_abs_bias)
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(self, pos_q, time_q, track_q, pos_k, time_k, track_k,
                is_query_q=None, is_query_k=None):
        if pos_q.ndim != 3 or pos_k.ndim != 3 or pos_q.size(-1) != 2 or pos_k.size(-1) != 2:
            raise ValueError("pos_q and pos_k must have shape [B, L, 2]")
        if pos_q.size(0) != pos_k.size(0):
            raise ValueError("query/key batch sizes must match")
        if is_query_q is None:
            is_query_q = torch.zeros_like(track_q, dtype=torch.bool)
        if is_query_k is None:
            is_query_k = torch.zeros_like(track_k, dtype=torch.bool)
        velocity_q, acceleration_q = self._historical_kinematics(
            pos_q, time_q, track_q, pos_k, time_k, track_k, is_query_k,
            self.history_window, self.history_decay,
        )
        velocity_k, acceleration_k = self._historical_kinematics(
            pos_k, time_k, track_k, pos_k, time_k, track_k, is_query_k,
            self.history_window, self.history_decay,
        )
        latest_velocity_q = self._latest_velocity(
            pos_q, time_q, track_q, pos_k, time_k, track_k, is_query_k,
        )
        latest_velocity_k = self._latest_velocity(
            pos_k, time_k, track_k, pos_k, time_k, track_k, is_query_k,
        )
        chunks = []
        for start in range(0, pos_q.size(1), self.query_chunk_size):
            args = (pos_q[:, start : start + self.query_chunk_size], time_q[:, start : start + self.query_chunk_size],
                    track_q[:, start : start + self.query_chunk_size], pos_k, time_k, track_k,
                    velocity_q[:, start : start + self.query_chunk_size],
                    acceleration_q[:, start : start + self.query_chunk_size], velocity_k, acceleration_k,
                    latest_velocity_q[:, start : start + self.query_chunk_size], latest_velocity_k,
                    is_query_q[:, start : start + self.query_chunk_size], is_query_k)
            if self.checkpoint_chunks and self.training:
                chunk_bias = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                chunk_bias = self._chunk(*args)
            chunks.append(chunk_bias)
        if not chunks:
            return pos_q.new_zeros((pos_q.size(0), self.n_heads, 0, pos_k.size(1)))
        bias = torch.cat(chunks, dim=2)
        if not torch.isfinite(bias).all():
            raise FloatingPointError("PhysicsRelationBiasMLP produced NaN or Inf")
        return bias

    def for_layer(self, bias: torch.Tensor, active_layer_index: int) -> torch.Tensor:
        if not 0 <= active_layer_index < self.n_bias_layers:
            raise IndexError(f"active_layer_index out of range: {active_layer_index}")
        scale = self.layer_head_scales[active_layer_index][None, :, None, None].to(dtype=bias.dtype)
        scaled_bias = bias * scale
        return self.max_abs_bias * torch.tanh(scaled_bias / self.max_abs_bias)


class LongHistoryAttentionBiasMLP(nn.Module):
    """Attention bias using temporal/identity metadata, never physical state.

    Positions are accepted only to keep the relation-bias call interface shared
    with ``PhysicsRelationBiasMLP``.  The bias is a function of time, track
    identity, observation availability, and the amount of causal history.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_dim: int = 64,
        depth: int = 2,
        time_scale: float = 50.0,
        max_abs_bias: float = 0.25,
        n_bias_layers: int = 4,
        initial_layer_scale: float = 0.25,
        history_window: int = 8,
        query_chunk_size: int = 64,
        checkpoint_chunks: bool = False,
        cross_track_initial_scale: float = 0.1,
    ):
        super().__init__()
        if depth < 1 or n_bias_layers < 1 or query_chunk_size < 1:
            raise ValueError("depth, n_bias_layers, and query_chunk_size must be positive")
        if time_scale <= 0 or max_abs_bias <= 0 or history_window < 1:
            raise ValueError("time_scale, max_abs_bias, and history_window must be positive")
        if initial_layer_scale < 0:
            raise ValueError("initial_layer_scale must be non-negative")
        if not 0.0 < cross_track_initial_scale <= 1.0:
            raise ValueError("cross_track_initial_scale must be in (0, 1]")
        self.n_heads = n_heads
        self.time_scale = float(time_scale)
        self.max_abs_bias = float(max_abs_bias)
        self.n_bias_layers = n_bias_layers
        self.history_window = int(history_window)
        self.query_chunk_size = query_chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        initial_logit = float(torch.logit(torch.tensor(cross_track_initial_scale)))
        self.cross_track_logit = nn.Parameter(torch.full((n_heads,), initial_logit))

        layers: list[nn.Module] = [nn.Linear(6, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(zero_init(nn.Linear(hidden_dim, n_heads)))
        self.mlp = nn.Sequential(*layers)
        self.layer_head_scales = nn.Parameter(
            torch.full((n_bias_layers, n_heads), float(initial_layer_scale))
        )

    def _chunk(self, time_q, track_q, time_k, track_k, is_query_k):
        delta = time_q[:, :, None] - time_k[:, None, :]
        delta_scaled = (delta / self.time_scale).clamp(-1.0, 1.0)
        same_track = track_q[:, :, None] == track_k[:, None, :]
        observed = ~is_query_k[:, None, :]
        causal_history = same_track & observed & (delta > 0)
        history_count = causal_history.sum(dim=-1, keepdim=True).clamp(max=self.history_window)
        history_fraction = history_count.to(delta.dtype) / self.history_window
        history_fraction = history_fraction.expand_as(delta)
        features = torch.stack(
            (
                delta_scaled,
                delta_scaled.abs(),
                same_track.to(delta.dtype),
                observed.expand_as(delta).to(delta.dtype),
                causal_history.to(delta.dtype),
                history_fraction,
            ),
            dim=-1,
        )
        raw_bias = self.mlp(features)
        bias = self.max_abs_bias * torch.tanh(raw_bias / self.max_abs_bias)
        # Long-history evidence is strongest for the same track.  Keep a small,
        # learnable cross-track residual so the model can still adapt to
        # interactions without letting unrelated tracks dominate by default.
        cross_track_gate = torch.sigmoid(self.cross_track_logit).to(delta.dtype)
        bias = bias.permute(0, 3, 1, 2).contiguous()
        relation_gate = same_track[:, None].to(delta.dtype) + (
            1.0 - same_track[:, None].to(delta.dtype)
        ) * cross_track_gate[None, :, None, None]
        return bias * relation_gate

    def forward(self, pos_q, time_q, track_q, pos_k, time_k, track_k,
                is_query_q=None, is_query_k=None):
        if pos_q.ndim != 3 or pos_k.ndim != 3:
            raise ValueError("pos_q and pos_k must have shape [B, L, C]")
        if is_query_k is None:
            is_query_k = torch.zeros_like(track_k, dtype=torch.bool)
        chunks = []
        for start in range(0, time_q.size(1), self.query_chunk_size):
            args = (
                time_q[:, start:start + self.query_chunk_size],
                track_q[:, start:start + self.query_chunk_size],
                time_k, track_k, is_query_k,
            )
            if self.checkpoint_chunks and self.training:
                chunk_bias = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                chunk_bias = self._chunk(*args)
            chunks.append(chunk_bias)
        if not chunks:
            return pos_q.new_zeros((pos_q.size(0), self.n_heads, 0, pos_k.size(1)))
        bias = torch.cat(chunks, dim=2)
        if not torch.isfinite(bias).all():
            raise FloatingPointError("LongHistoryAttentionBiasMLP produced NaN or Inf")
        return bias

    def for_layer(self, bias: torch.Tensor, active_layer_index: int) -> torch.Tensor:
        if not 0 <= active_layer_index < self.n_bias_layers:
            raise IndexError(f"active_layer_index out of range: {active_layer_index}")
        scale = self.layer_head_scales[active_layer_index][None, :, None, None].to(bias.dtype)
        scaled_bias = bias * scale
        return self.max_abs_bias * torch.tanh(scaled_bias / self.max_abs_bias)

class FusedTransformerLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_head: int,
        dropout: float = 0.0,
        ff_expand: int = 4,
        scaled_cosine_sim: bool = True,
        ada_norm_size: int = -1,
        time_norm_size: int = -1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.d_ff = d_model * ff_expand
        self.n_heads = d_model // d_head
        self.scaled_cosine_sim = scaled_cosine_sim
        self.ada_norm_size = ada_norm_size
        self.time_norm_size = time_norm_size

        if ada_norm_size <= 0 and time_norm_size <= 0:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = AdaRMSNorm(d_model)

        self.in_proj = nn.Linear(d_model, d_model * (ff_expand + 3), bias=False)  # d_model -> FFN + QKV
        self.out_proj = zero_init(nn.Linear((1 + ff_expand) * d_model, d_model, bias=False)) # Attn + FFN -> d_model
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU(approximate="tanh")

        if self.scaled_cosine_sim:
            self.scale = nn.Parameter(torch.full([2, self.n_heads], 10.0))

        self.kv_cache = KVCache((1, self.n_heads, 0, self.d_head), dtype=torch.bfloat16, device=None)
    
    # Factored-out first part of forward pass (until attention)
    # During KV-cached inference, this will only ever see two shapes: [B, L_prefill] or [B, 2], so we can nicely compile this
    def _fwd_1(
        self,
        x: Float[torch.Tensor, "b l d_model"],  # this is fused x + x_cross
        theta: Float[torch.Tensor, "b n_h l d_head"],
        scale: Float[torch.Tensor, "b d_model"],
    ):
        # Normalization (adaptive if needed)
        if self.ada_norm_size > 0 or self.time_norm_size > 0:
            proj = self.norm(x, scale)
        else:
            proj = self.norm(x)

        # Combined QKV + Up projection
        proj = self.in_proj(proj)  # B x L x (3*d_model + d_ff)
        qkv_self, up = torch.split(proj, [3 * self.d_model, self.d_ff], dim=-1)

        # Non-linearity on up projection
        up = self.act(up)

        # Split QKV and scale
        q, k, v = einops.rearrange(qkv_self, "b l (t nh e) -> t b nh l e", t=3, e=self.d_head)
        if self.scaled_cosine_sim:
            q, k = scale_for_cosine_sim(q, k, self.scale[0, :, None, None], 1e-6)

        # Apply positional embedding
        theta = theta.movedim(-2, -3)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)
        v = v.to(q.dtype)

        return q, k, v, up

    def forward(
        self,
        x: Float[torch.Tensor, "b l d_model"],  # this is fused x + x_cross
        theta: Float[torch.Tensor, "b n_h l d_head"],
        scale: Float[torch.Tensor, "b d_model"],
        block_mask,
        i_kv: int | None = None,
        physics_bias: torch.Tensor | None = None,
        q_motion_offset: int = 0,
        k_motion_offset: int = 0,
        **kwargs,
    ):
        B, L, _ = x.shape
        skip = x 

        q, k, v, up = self._fwd_1(x=x, theta=theta, scale=scale)

        # for KVCaching
        if i_kv is not None:
            self.kv_cache.k[:B, :, i_kv : i_kv + L] = k
            self.kv_cache.v[:B, :, i_kv : i_kv + L] = v
            k = self.kv_cache.k[:B, :, : i_kv + L]
            v = self.kv_cache.v[:B, :, : i_kv + L]

        # Keep the original no-bias path unchanged for checkpoint-compatible ablations.
        if physics_bias is None and isinstance(block_mask, str) and block_mask == "causal":
            attn = F.scaled_dot_product_attention(
                q, k, v, scale=1.0 if self.scaled_cosine_sim else None, is_causal=True
            )
        elif physics_bias is None and (isinstance(block_mask, torch.Tensor) or block_mask is None):
            attn = F.scaled_dot_product_attention(
                q, k, v, scale=1.0 if self.scaled_cosine_sim else None, attn_mask=block_mask
            )
        elif physics_bias is None:
            attn = flex_attention(
                q, k, v, scale=1.0 if self.scaled_cosine_sim else None, block_mask=block_mask
            )
        else:
            if physics_bias.shape[:2] != (B, self.n_heads):
                raise AssertionError(f"physics bias batch/head mismatch: {physics_bias.shape=}, {B=}, {self.n_heads=}")
            lq_motion, lk_motion = physics_bias.shape[-2:]
            if q_motion_offset + lq_motion != q.size(-2) or k_motion_offset + lk_motion != k.size(-2):
                raise AssertionError(
                    f"physics bias does not align with attention: bias={physics_bias.shape}, "
                    f"q={q.shape}, k={k.shape}, offsets=({q_motion_offset}, {k_motion_offset})"
                )

            if isinstance(block_mask, BlockMask):
                attn = physics_flex_attention(
                    q, k, v, physics_bias, block_mask, q_motion_offset, k_motion_offset,
                    1.0 if self.scaled_cosine_sim else None,
                )
            else:
                full_bias = q.new_zeros((B, self.n_heads, q.size(-2), k.size(-2)))
                full_bias[:, :, q_motion_offset:, k_motion_offset:] = physics_bias.to(q.dtype)
                if isinstance(block_mask, str):
                    if block_mask != "causal":
                        raise ValueError(f"Unknown attention mask string: {block_mask}")
                    q_idx = torch.arange(q.size(-2), device=q.device)[:, None]
                    k_idx = torch.arange(k.size(-2), device=q.device)[None, :]
                    allowed = k_idx <= q_idx
                    attn_mask = full_bias.masked_fill(~allowed, float("-inf"))
                elif block_mask is None:
                    attn_mask = full_bias
                elif block_mask.dtype == torch.bool:
                    attn_mask = full_bias.masked_fill(~block_mask, float("-inf"))
                else:
                    attn_mask = full_bias + block_mask.to(q.dtype)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                      scale=1.0 if self.scaled_cosine_sim else None)
        attn_out = einops.rearrange(attn, "b nh l d -> b l (nh d)")

        # Combine attention output and FFN output
        x = torch.cat([attn_out, up], dim=-1)

        # Dropout and projection to input size (finalizing the FFN)
        x = self.dropout(x)
        x = self.out_proj(x)

        # skip connection
        return x + skip

class FusedTransformer(nn.Module):

    def __init__(
        self,
        width: int,
        depth: int,
        aux_feat_dim: int,
        d_head: int,
        out_mlp_depth: int,
        input_scale: float = 1.0,
        emb_depth: int = 3,
        track_id_embedding: bool = True,
        max_num_track_ids: int = 256,
        ff_expand: int = 4,
        ada_norm_size: int = -1,
        time_norm_size: int = -1,
        dropout: float = 0.0,
        scaled_cosine_sim: bool = True,
        use_physics_bias: bool = False,
        physics_bias_hidden_dim: int = 64,
        physics_bias_depth: int = 2,
        physics_bias_time_scale: float = 50.0,
        physics_bias_max_abs: float = 1.0,
        physics_bias_ball_radius: float = 0.033,
        physics_bias_num_layers: int | None = None,
        physics_bias_initial_scale: float = 0.5,
        physics_bias_history_window: int = 8,
        physics_bias_history_decay: float = 4.0,
        physics_bias_query_chunk_size: int = 64,
        physics_bias_checkpoint_chunks: bool = False,
        use_long_history_bias: bool = False,
        long_history_bias_hidden_dim: int = 64,
        long_history_bias_depth: int = 2,
        long_history_bias_time_scale: float = 50.0,
        long_history_bias_max_abs: float = 0.25,
        long_history_bias_num_layers: int | None = None,
        long_history_bias_initial_scale: float = 0.25,
        long_history_bias_window: int = 8,
        long_history_bias_query_chunk_size: int = 64,
        long_history_bias_cross_track_initial_scale: float = 0.1,
        long_history_bias_checkpoint_chunks: bool = False,
        use_combined_bias_gates: bool = False,
        combined_physics_initial_scale: float = 0.5,
        combined_long_history_initial_scale: float = 0.25,
        combined_gate_hidden_dim: int = 32,
        # use_full_skip: bool = True,
        # num_learnable_track_ids: int = -1,
    ):
        super().__init__()

        self.ada_norm_size = ada_norm_size
        self.time_norm_size = time_norm_size
        self.use_physics_bias = use_physics_bias
        self.use_long_history_bias = use_long_history_bias
        self.use_combined_bias_gates = bool(
            use_combined_bias_gates and use_physics_bias and use_long_history_bias
        )
        physics_bias_num_layers = min(4, depth) if physics_bias_num_layers is None else physics_bias_num_layers
        long_history_bias_num_layers = (
            min(4, depth) if long_history_bias_num_layers is None else long_history_bias_num_layers
        )
        if use_physics_bias and not 1 <= physics_bias_num_layers <= depth:
            raise ValueError(f"physics_bias_num_layers must be in [1, {depth}], got {physics_bias_num_layers}")
        if use_long_history_bias and not 1 <= long_history_bias_num_layers <= depth:
            raise ValueError(
                f"long_history_bias_num_layers must be in [1, {depth}], got {long_history_bias_num_layers}"
            )
        self.physics_bias_num_layers = max(physics_bias_num_layers, long_history_bias_num_layers)
        # self.use_full_skip = use_full_skip

        self.embedder = FlowPokeEmbedder(
            out_features=width,
            input_scale=input_scale,
            aux_feature_dim=aux_feat_dim,
            depth=emb_depth,
            track_id_embedding=track_id_embedding,
            max_num_track_ids=max_num_track_ids,
            ada_norm_size=ada_norm_size,
        )
        self.out_proj = FlowPokeOutput(width, out_mlp_depth)

        cond_in_shape = max(0, ada_norm_size) + max(0, time_norm_size)
        if cond_in_shape > 0:
            self.scale_proj = nn.Sequential(
                nn.Linear(cond_in_shape, width, bias=False),
                nn.SiLU(),
                zero_init(nn.Linear(width, width, bias=False)),
            )
        else:
            self.scale_proj = None

        self.pos_emb = FPRoPE(d_head, n_heads=width // d_head)
        self.cross_proj = nn.Linear(aux_feat_dim, width, bias=False)

        mid_level = []
        for i in range(depth):
            layer = FusedTransformerLayer(
                d_model=width,
                d_head=d_head,
                dropout=dropout,
                ff_expand=ff_expand,
                scaled_cosine_sim=scaled_cosine_sim,
                ada_norm_size=ada_norm_size,
                time_norm_size=time_norm_size,
            )
            mid_level.append(layer)
        self.mid_level = Level(mid_level)

        self.physics_bias_generator = None
        self.long_history_bias_generator = None
        if use_physics_bias:
            self.physics_bias_generator = PhysicsRelationBiasMLP(
                n_heads=width // d_head,
                hidden_dim=physics_bias_hidden_dim,
                depth=physics_bias_depth,
                time_scale=physics_bias_time_scale,
                max_abs_bias=physics_bias_max_abs,
                ball_radius=physics_bias_ball_radius,
                n_bias_layers=physics_bias_num_layers,
                initial_layer_scale=physics_bias_initial_scale,
                history_window=physics_bias_history_window,
                history_decay=physics_bias_history_decay,
                query_chunk_size=physics_bias_query_chunk_size,
                checkpoint_chunks=physics_bias_checkpoint_chunks,
            )
        if use_long_history_bias:
            long_history_generator = LongHistoryAttentionBiasMLP(
                n_heads=width // d_head,
                hidden_dim=long_history_bias_hidden_dim,
                depth=long_history_bias_depth,
                time_scale=long_history_bias_time_scale,
                max_abs_bias=long_history_bias_max_abs,
                n_bias_layers=long_history_bias_num_layers,
                initial_layer_scale=long_history_bias_initial_scale,
                history_window=long_history_bias_window,
                query_chunk_size=long_history_bias_query_chunk_size,
                cross_track_initial_scale=long_history_bias_cross_track_initial_scale,
                checkpoint_chunks=long_history_bias_checkpoint_chunks,
            )
            # Keep the historical state-dict path for the temporal-only model;
            # combined models use the dedicated second generator path.
            if use_physics_bias:
                self.long_history_bias_generator = long_history_generator
            else:
                self.physics_bias_generator = long_history_generator
        if self.use_combined_bias_gates:
            if not 0.0 < combined_physics_initial_scale <= 1.0:
                raise ValueError("combined_physics_initial_scale must be in (0, 1]")
            if not 0.0 < combined_long_history_initial_scale <= 1.0:
                raise ValueError("combined_long_history_initial_scale must be in (0, 1]")
            n_heads = width // d_head
            self.physics_source_logit = nn.Parameter(torch.full(
                (n_heads,), float(torch.logit(torch.tensor(combined_physics_initial_scale)))
            ))
            self.long_history_source_logit = nn.Parameter(torch.full(
                (n_heads,), float(torch.logit(torch.tensor(combined_long_history_initial_scale)))
            ))
            self.combined_gate_mlp = nn.Sequential(
                nn.Linear(6, combined_gate_hidden_dim),
                nn.SiLU(),
                zero_init(nn.Linear(combined_gate_hidden_dim, 2 * n_heads)),
            )
        self.register_buffer("physics_pos_cache", torch.empty((1, 0, 2)), persistent=False)
        self.register_buffer("physics_time_cache", torch.empty((1, 0)), persistent=False)
        self.register_buffer("physics_track_cache", torch.empty((1, 0), dtype=torch.long), persistent=False)
        self.register_buffer("physics_query_cache", torch.empty((1, 0), dtype=torch.bool), persistent=False)
        self.cached_image_prefix_length: int | None = None

    
    # Factored-out first part of forward pass (until layers)
    # During KV-cached inference, this will only ever see two shapes: [B, L_prefill] or [B, 2], so we can nicely compile this
    def _fwd_1(
        self,
        x: Float[torch.Tensor, "b *DIMS C"],
        x_cross: Float[torch.Tensor, "b l_c d_cross"],
        pos: Float[torch.Tensor, "b *DIMS 2"],
        pos_orig: Float[torch.Tensor, "b *DIMS 2"],
        pos_cross: Float[torch.Tensor, "b h w 2"],
        is_query: Bool[torch.Tensor, "b l"],
        track_id: Int[torch.Tensor, "b l"],
        camera_static: Bool[torch.Tensor, "b"],
        time: Float[torch.Tensor, "b 1"],
        compute_cross: bool,
        kwargs,
    ):
        # single prediction of scale and scale_cross for all layers based on camera_static and time
        if self.scale_proj is None:
            # kwargs["scale"] = None
            # kwargs["scale_cross"] = None
            scale = None
        else:
            camera_static_emb = self.embedder.embed_camera_static(camera_static)
            time_emb = time  # self.embedder.time_emb(time)
            if self.ada_norm_size > 0 and self.time_norm_size > 0:
                cond_emb = torch.cat([camera_static_emb, time_emb], dim=-1)
            elif self.ada_norm_size > 0:
                cond_emb = camera_static_emb
            else:
                cond_emb = time_emb
            scale = self.scale_proj(cond_emb)

        # Embed flows using embedder
        C_pos = pos.shape[-1]
        x, pos = self.embedder(
            x=x,
            pos=pos,
            pos_orig=pos_orig,
            is_query=is_query,
            aux_feats=einops.rearrange(x_cross, "b (h w) d -> b d h w", h=pos_cross.shape[1], w=pos_cross.shape[2]),
            track_id=track_id,
            **kwargs,
        )

        # single shared positional embedding (possible as frequencies are not learnable)
        theta = self.pos_emb(pos, pos_orig, time, **kwargs)
        if compute_cross:
            pos_cross = einops.rearrange(pos_cross, "b h w c -> b (h w) c")
            # compute positional embedding for cross-attention and concatenate along sequence dimension
            theta_cross = self.pos_emb(
                pos_cross, pos_cross, time=time.new_zeros((pos_cross.shape[0], pos_cross.shape[1]))
            )  # NOTE: zeroing out time might not be ideal
            theta = torch.cat([theta_cross, theta], dim=1)

            # concatenate cross tokens
            x_cross = self.cross_proj(x_cross)  # project cross tokens to model dimension
            x = torch.cat([x_cross, x], dim=1)  # cat along sequence dimension as prefix
            L_cross = x_cross.shape[1]
        else:
            L_cross = 0

        return x, theta, scale, L_cross, kwargs
    
    def forward(
        self,
        x: Float[torch.Tensor, "b *DIMS C"],
        x_cross: Float[torch.Tensor, "b l_c d_cross"],
        pos: Float[torch.Tensor, "b *DIMS 2"],
        pos_orig: Float[torch.Tensor, "b *DIMS 2"],
        pos_cross: Float[torch.Tensor, "b h w 2"],
        is_query: Bool[torch.Tensor, "b l"],
        track_id: Int[torch.Tensor, "b l"],
        camera_static: Bool[torch.Tensor, "b"],
        time: Float[torch.Tensor, "b 1"],
        block_mask: Any = None,
        i_kv: int | None = None,
        compute_cross: bool = True,
        **kwargs,
    ):
        motion_pos_current, motion_time_current, motion_track_current = pos, time, track_id
        motion_query_current = is_query
        x, theta, scale, L_cross, kwargs = self._fwd_1(
            x=x,
            x_cross=x_cross,
            pos=pos,
            pos_orig=pos_orig,
            pos_cross=pos_cross,
            is_query=is_query,
            track_id=track_id,
            camera_static=camera_static,
            time=time,
            compute_cross=compute_cross,
            kwargs=kwargs,
        )

        skip = x[:, L_cross:]

        physics_bias = None
        long_history_bias = None
        combined_context_gates = None
        q_motion_offset = k_motion_offset = 0
        if self.use_physics_bias or self.use_long_history_bias:
            if self.use_physics_bias and self.physics_bias_generator is None:
                raise AssertionError("use_physics_bias=True without a generator")
            if self.use_long_history_bias and self.long_history_bias_generator is None and self.physics_bias_generator is None:
                raise AssertionError("use_long_history_bias=True without a generator")
            B_motion, L_motion, _ = motion_pos_current.shape
            if i_kv is None:
                pos_k, time_k, track_k = motion_pos_current, motion_time_current, motion_track_current
                query_k = motion_query_current
                q_motion_offset = k_motion_offset = L_cross
            else:
                if i_kv == 0:
                    if not compute_cross or L_cross <= 0:
                        raise AssertionError("physics metadata prefill requires image-prefix computation")
                    self.cached_image_prefix_length = L_cross
                    self._ensure_physics_cache(B_motion, max(self.physics_pos_cache.size(1), L_motion),
                                               motion_pos_current.dtype, motion_pos_current.device)
                    write_start = 0
                else:
                    if compute_cross or self.cached_image_prefix_length is None:
                        raise AssertionError("incremental physics inference requires a previous prefill")
                    if i_kv < self.cached_image_prefix_length:
                        raise AssertionError("i_kv points inside the cached image prefix")
                    write_start = i_kv - self.cached_image_prefix_length
                write_end = write_start + L_motion
                if write_end > self.physics_pos_cache.size(1):
                    raise AssertionError("physics metadata write exceeds allocated cache; call grow_kv_cache first")
                self.physics_pos_cache[:B_motion, write_start:write_end] = motion_pos_current
                self.physics_time_cache[:B_motion, write_start:write_end] = motion_time_current
                self.physics_track_cache[:B_motion, write_start:write_end] = motion_track_current.long()
                self.physics_query_cache[:B_motion, write_start:write_end] = motion_query_current.bool()
                pos_k = self.physics_pos_cache[:B_motion, :write_end]
                time_k = self.physics_time_cache[:B_motion, :write_end]
                track_k = self.physics_track_cache[:B_motion, :write_end]
                query_k = self.physics_query_cache[:B_motion, :write_end]
                q_motion_offset = L_cross if i_kv == 0 else 0
                k_motion_offset = self.cached_image_prefix_length
            if self.use_physics_bias:
                physics_bias = self.physics_bias_generator(
                    motion_pos_current, motion_time_current, motion_track_current, pos_k, time_k, track_k,
                    motion_query_current, query_k,
                )
            if self.use_long_history_bias:
                long_generator = self.long_history_bias_generator or self.physics_bias_generator
                long_history_bias = long_generator(
                    motion_pos_current, motion_time_current, motion_track_current, pos_k, time_k, track_k,
                    motion_query_current, query_k,
                )
            if self.use_combined_bias_gates:
                delta = motion_time_current[:, :, None] - time_k[:, None, :]
                delta_scaled = (delta / 50.0).clamp(-1.0, 1.0)
                same_track = motion_track_current[:, :, None] == track_k[:, None, :]
                observed = (~query_k)[:, None, :].expand_as(delta)
                distance = torch.linalg.vector_norm(
                    motion_pos_current[:, :, None, :] - pos_k[:, None, :, :], dim=-1
                ).clamp(max=1.0)
                wall_distance = torch.minimum(
                    torch.minimum(motion_pos_current[..., 0], 1.0 - motion_pos_current[..., 0]),
                    torch.minimum(motion_pos_current[..., 1], 1.0 - motion_pos_current[..., 1]),
                )[:, :, None].expand_as(delta).clamp(max=1.0)
                gate_features = torch.stack(
                    (delta_scaled, delta_scaled.abs(), same_track.to(delta.dtype),
                     observed.to(delta.dtype), distance, wall_distance), dim=-1
                )
                gate_delta = self.combined_gate_mlp(gate_features)
                gate_delta = gate_delta.permute(0, 3, 1, 2).contiguous()
                base_physics = self.physics_source_logit[None, :, None, None]
                base_temporal = self.long_history_source_logit[None, :, None, None]
                combined_context_gates = (
                    torch.sigmoid(base_physics + gate_delta[:, :self.pos_emb.n_heads]),
                    torch.sigmoid(base_temporal + gate_delta[:, self.pos_emb.n_heads:]),
                )

        # standard transformer forward
        B, *DIMS, C = x.shape
        x = x.reshape(B, -1, C)
        first_physics_layer = len(self.mid_level) - self.physics_bias_num_layers
        for layer_index, layer in enumerate(self.mid_level):
            layer_bias = None
            if (physics_bias is not None or long_history_bias is not None) and layer_index >= first_physics_layer:
                active_index = layer_index - first_physics_layer
                bias_parts = []
                if physics_bias is not None:
                    physics_part = self.physics_bias_generator.for_layer(physics_bias, active_index)
                    if self.use_combined_bias_gates:
                        if combined_context_gates is None:
                            physics_gate = torch.sigmoid(self.physics_source_logit)[None, :, None, None]
                        else:
                            physics_gate = combined_context_gates[0]
                        physics_part = physics_part * physics_gate
                    bias_parts.append(physics_part)
                if long_history_bias is not None:
                    long_generator = self.long_history_bias_generator or self.physics_bias_generator
                    long_part = long_generator.for_layer(long_history_bias, active_index)
                    if self.use_combined_bias_gates:
                        if combined_context_gates is None:
                            temporal_gate = torch.sigmoid(self.long_history_source_logit)[None, :, None, None]
                        else:
                            temporal_gate = combined_context_gates[1]
                        long_part = long_part * temporal_gate
                    bias_parts.append(long_part)
                layer_bias = torch.stack(bias_parts, dim=0).sum(dim=0)
            x = layer(x, theta, scale=scale, block_mask=block_mask, i_kv=i_kv,
                      physics_bias=layer_bias, q_motion_offset=q_motion_offset,
                      k_motion_offset=k_motion_offset, **kwargs)
        x = x.reshape(B, *DIMS, C)

        # out projection after removing cross tokens
        # ->  we process cross tokens in the same pass, but do not update them

        # if self.use_full_skip:
        #     x = self.out_proj(x[:, L_cross:]) + skip
        # else:
        x = self.out_proj(x[:, L_cross:])
        return x

    def reset_kv_cache(self, b: int, l: int):
        for layer in self.mid_level:
            layer.kv_cache.k = layer.kv_cache.k.new_zeros((b, layer.kv_cache.k.size(1), l, layer.kv_cache.k.size(3)))
            layer.kv_cache.v = layer.kv_cache.v.new_zeros((b, layer.kv_cache.v.size(1), l, layer.kv_cache.v.size(3)))
        self.cached_image_prefix_length = None
        self._ensure_physics_cache(b, l, self.physics_pos_cache.dtype, self.physics_pos_cache.device, clear=True)

    def _ensure_physics_cache(self, b: int, l: int, dtype: torch.dtype, device: torch.device, clear: bool = False):
        needs_resize = (self.physics_pos_cache.size(0) < b or self.physics_pos_cache.size(1) < l
                        or self.physics_pos_cache.dtype != dtype or self.physics_pos_cache.device != device)
        if needs_resize or clear:
            old_pos, old_time, old_track = self.physics_pos_cache, self.physics_time_cache, self.physics_track_cache
            old_query = self.physics_query_cache
            self.physics_pos_cache = torch.zeros((b, l, 2), dtype=dtype, device=device)
            self.physics_time_cache = torch.zeros((b, l), dtype=dtype, device=device)
            self.physics_track_cache = torch.zeros((b, l), dtype=torch.long, device=device)
            self.physics_query_cache = torch.zeros((b, l), dtype=torch.bool, device=device)
            if needs_resize and not clear and old_pos.numel() > 0:
                cb, cl = min(b, old_pos.size(0)), min(l, old_pos.size(1))
                self.physics_pos_cache[:cb, :cl] = old_pos[:cb, :cl].to(device=device, dtype=dtype)
                self.physics_time_cache[:cb, :cl] = old_time[:cb, :cl].to(device=device, dtype=dtype)
                self.physics_track_cache[:cb, :cl] = old_track[:cb, :cl].to(device=device)
                self.physics_query_cache[:cb, :cl] = old_query[:cb, :cl].to(device=device)

    def grow_kv_cache(self, b: int, l: int):
        for layer in self.mid_level:
            b_cur, _, l_cur, _ = layer.kv_cache.k.shape
            if b_cur < b or l_cur < l:
                old_k, old_v = layer.kv_cache.k, layer.kv_cache.v
                new_k = old_k.new_zeros(
                    (b, layer.kv_cache.k.size(1), l, layer.kv_cache.k.size(3))
                )
                new_v = old_v.new_zeros(
                    (b, layer.kv_cache.v.size(1), l, layer.kv_cache.v.size(3))
                )
                copy_b, copy_l = min(b, b_cur), min(l, l_cur)
                new_k[:copy_b, :, :copy_l] = old_k[:copy_b, :, :copy_l]
                new_v[:copy_b, :, :copy_l] = old_v[:copy_b, :, :copy_l]
                layer.kv_cache.k, layer.kv_cache.v = new_k, new_v
        self._ensure_physics_cache(b, l, self.physics_pos_cache.dtype, self.physics_pos_cache.device)

# ---------------------------------------------------------------------------------------------------------------------
# Rectified Flow Posterior Distribution Head
# ---------------------------------------------------------------------------------------------------------------------

class RFHeadFFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_cond_norm: int,
        expansion_factor: float=1.,
        relu: bool = False,
    ):
        super().__init__()
        d_ff = int(d_model * expansion_factor)
        self.norm = AdaLN(d_model, d_cond_norm)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.act = self.ReLU(inplace=True) if relu else nn.GELU(approximate="tanh")
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, scale, shift):
        skip = x
        x_dtype = x.dtype
        norm_dtype = torch.promote_types(x.dtype, torch.float32)
        x = F.layer_norm(x.to(norm_dtype), (x.shape[-1],), weight=None, bias=None, eps=1e-5) * (
            1 + scale.to(norm_dtype)
        ) + shift.to(norm_dtype)
        x = x.to(x_dtype)
        x = self.act(self.up_proj(x))
        x = self.down_proj(x)
        return x + skip

class RFHead(nn.Module):

    def __init__(
        self,
        depth: int,
        d_model: int,
        d_cond: int,
        d_out: int,
        expansion_factor: int = 1,
        internal_value_scale: float = 1.0,
        value_scale_cascade: bool = True,
        value_scale_cascade_steps: int = 512,
        cat_embs: bool = False,
        relu: bool = False,
        cond_dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        self.d_model = d_model
        self.d_cond = d_cond
        assert d_cond <= d_model, "Conditioning dimension must be less than or equal to model dimension"
        self.d_out = d_out
        self.cat_embs = cat_embs
        self.expansion_factor = expansion_factor
        self.relu = relu
        self.internal_value_scale = internal_value_scale
        self.cond_dropout = cond_dropout

        self.time_emb = FourierFeatures(1, d_model)
        self.time_mapping = ResidualWrapper(
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(approximate="tanh") if not self.relu else nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
                nn.GELU(approximate="tanh") if not self.relu else nn.ReLU(inplace=True),
                zero_init(nn.Linear(d_model, d_model)),
            )
        )

        self.cond_emb = nn.Linear(
            d_cond, d_model, bias=False
        )   # only a linear projection -> the processing should be done in the Transformer model
        # project concatenated or added embedding to the correct size for scales and shifts
        self.emb_mapping = zero_init(nn.Linear(d_model * (2 if self.cat_embs else 1), depth * d_model * 2, bias=False))
        self.layers = nn.ModuleList(
            [
                RFHeadFFN(
                    d_model=self.d_model,
                    d_cond_norm=self.d_model,
                    relu=self.relu,
                    expansion_factor=self.expansion_factor
                )
                for _ in range(depth)
            ]
        )

        self.out_norm = nn.LayerNorm(d_model, eps=1e-6).to(torch.float32)
        self.out_proj = zero_init(nn.Linear(d_model, d_out, bias=False))

        # initialize value scale cascade
        self.value_scale_cascade = value_scale_cascade
        if not value_scale_cascade:
            self.in_proj = nn.Linear(d_out, d_model, bias=False)
        else:
            assert d_model % d_out == 0, "d_out must divide d_model for value scale cascade"
            self.in_proj_scales = torch.nn.Parameter(
                torch.linspace(math.log(0.1), math.log(1e5), value_scale_cascade_steps).exp(),
                requires_grad=False
            )
            self.in_proj = nn.Linear(value_scale_cascade_steps * d_out, d_model, bias=False)
        
        self._timestep_emb_cache = {}

    def forward(
        self,
        x: Float[torch.Tensor, "... d_out"],
        cond: Float[torch.Tensor, "... d_cond"],
        t: Float[torch.Tensor, "... 1"],
    ) -> Float[torch.Tensor, "... d_out"]:
        if self.value_scale_cascade:
            x = einops.rearrange((x[..., None, :] * self.in_proj_scales[:, None]).tanh(), "... s d -> ... (s d)")
        x = self.in_proj(x)

        if self.cat_embs:
            cond_embedding: Float[torch.Tensor, "... d_model"] = self.emb_mapping(
                torch.cat([self.time_mapping(self.time_emb(t)), self.cond_emb(cond)], dim=-1)
            )
        else:
            cond_embedding: Float[torch.Tensor, "... d_model"] = self.emb_mapping(
                self.time_mapping(self.time_emb(t)) + self.cond_emb(cond)
            )
        
        scales, shifts = einops.rearrange(cond_embedding, "... (depth two d) -> two depth ... d", two=2, depth=self.depth).contiguous()
        
        for layer, scale, shifts in zip(self.layers, scales, shifts):
            x = layer(x, scale, shifts)
        
        norm_dtype = torch.promote_types(x.dtype, torch.float32)
        x = _layer_norm_wrapper(x, self.out_norm, norm_dtype, x.dtype)
        x = self.out_proj(x)
        return x

    def forward_cached(
        self,
        x: Float[torch.Tensor, "... d_out"],
        cond_emb,
        t_emb,
    ):
        if self.value_scale_cascade:
            x = einops.rearrange((x[..., None, :] * self.in_proj_scales[:, None]).tanh(), "... s d -> ... (s d)")
        x = self.in_proj(x)
        if self.cat_embs:
            cond_embedding: Float[torch.Tensor, "... d_model"] = self.emb_mapping(
                torch.cat([t_emb, cond_emb], dim=-1)
            )
        else:
            cond_embedding: Float[torch.Tensor, "... d_model"] = self.emb_mapping(
                t_emb + cond_emb
            )

        scales, shifts = einops.rearrange(
            cond_embedding, "... (depth two d) -> two depth ... d", two=2, depth=self.depth
        ).contiguous()
        
        for layer, scale, shifts in zip(self.layers, scales, shifts):
            x = layer(x, scale, shifts)

        norm_dtype = torch.promote_types(x.dtype, torch.float32)
        x = _layer_norm_wrapper(x, self.out_norm, norm_dtype, x.dtype)
        x = self.out_proj(x)
        return x
    
    def get_forward_cache(
        self,
        cond: Float[torch.Tensor, "... d_cond"],
    ):
        target_dtype = cond.dtype
        cond_embedding: Float[torch.Tensor, "... d_model"] = self.cond_emb(cond)
        return {"cond_emb": cond_embedding.to(target_dtype)}
    
    def loss(
        self,
        x: Float[torch.Tensor, "... d_out"],
        cond: Float[torch.Tensor, "... d_cond"],
        t: Float[torch.Tensor, "... 1"] | None = None,
    ) -> Float[torch.Tensor, "..."]:
        x = x * self.internal_value_scale
        if t is None:
            t = torch.rand(x.shape[:-1], device=x.device, dtype=torch.float32)[..., None]
        else:
            t = t.float()[..., None]
        z = torch.randn(x.shape, device=x.device, dtype=torch.float32)
        xt = (1 - t) * x + t * z

        if self.cond_dropout > 0.0 and self.training:
            mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_dropout
            mask = mask.view(cond.shape[0], *([1] * (cond.dim() - 1)))
            cond = cond.masked_fill(mask, 0.0) # using all zeros for dropped conds (might not be ideal?) 

        vtheta = self(xt.to(cond.dtype), cond, t.to(cond.dtype)).float()
        return ((z - x - vtheta) ** 2).mean(dim=-1)
    
    def sample_inner(
        self,
        cond: Float[torch.Tensor, "... d_cond"],
        t_embs: list[Float[torch.Tensor, "... d_model"]],
        steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> Float[torch.Tensor, "... d_out"]:
        *DIMS, _ = cond.shape
        dt = torch.full(DIMS + [1], 1.0 / steps, device=cond.device, dtype=torch.float32)
        z = torch.randn(DIMS + [self.d_out], device=cond.device, dtype=torch.float32)
        cache_kwargs = self.get_forward_cache(cond=cond)
        if cfg_scale != 1.0:
            uncond_cache_kwargs = self.get_forward_cache(torch.zeros_like(cond))
        for i in range(steps):
            v = self.forward_cached(z.to(cond.dtype), t_emb=t_embs[i].to(cond.dtype), **cache_kwargs).float()

            if cfg_scale != 1.0:
                v_uncond = self.forward_cached(
                    z.to(cond.dtype), t_emb=t_embs[i].to(cond.dtype), **uncond_cache_kwargs
                ).float()
                v = v_uncond + cfg_scale * (v - v_uncond)
            z = z - dt * v
        return z / self.internal_value_scale

    @torch.no_grad()
    def sample(
        self,
        cond: Float[torch.Tensor, "... d_cond"],
        steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> Float[torch.Tensor, "... d_out"]:
        *DIMS, _ = cond.shape
        if self.training:
            self._timestep_emb_cache = {}
        key = f"{steps}_{','.join(str(l) for l in cond.shape)}"
        if not key in self._timestep_emb_cache:
            self._timestep_emb_cache[key] = [
                self.time_mapping(
                    self.time_emb(torch.full(DIMS + [1], i / steps, device=cond.device, dtype=cond.dtype))
                )
                for i in range(steps, 0, -1)
            ]
        t_embs = self._timestep_emb_cache[key]
        return self.sample_inner(cond=cond, steps=steps, t_embs=t_embs, cfg_scale=cfg_scale)

    def build_timestep_cache(self, steps: int, dims: tuple[int, ...], device: torch.device, dtype: torch.dtype):
        key = f"{steps}_{','.join(str(l) for l in list(dims) + [self.d_cond])}"
        if not key in self._timestep_emb_cache:
            self._timestep_emb_cache[key] = [
                self.time_mapping(self.time_emb(torch.full(list(dims) + [1], i / steps, device=device, dtype=dtype)))
                for i in range(steps, 0, -1)
            ]
        
class RFHeadDistribution(Distribution):

    def __init__(
        self,
        rf_head: RFHead,
        cond: Float[torch.Tensor, "... d_cond"],
        steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> None:
        self.head = rf_head
        self.cond = cond
        self.steps = steps
        self.cfg_scale = cfg_scale

        self._batch_shape = self.cond.shape[:-2]
        self._event_shape = self.cond.shape[-1:]

    def __getattribute__(self, name: str) -> Any:
        if name != "log_prob":
            return super().__getattribute__(name)
        raise AttributeError()
    
    def loss(self, x: Float[torch.Tensor, "... d_out"]) -> Float[torch.Tensor, "..."]:
        return self.head.loss(x, self.cond)
    
    @torch.no_grad()
    def rsample(self, sample_shape=torch.Size()) -> Float[torch.Tensor, "... d_out"]:
        cond = self.cond
        if len(sample_shape) > 0:
            for _ in range(len(sample_shape)):
                cond = cond[None]
            cond = cond.expand(*sample_shape, *cond.shape)
        return self.head.sample(cond=cond, steps=self.steps, cfg_scale=self.cfg_scale)

    def __getitem__(self, idx) -> "RFHeadDistribution":
        idx = idx if isinstance(idx, tuple) else (idx,)
        cond_sliced = self.cond[idx + (slice(None),)]
        return RFHeadDistribution(rf_head=self.head, cond=cond_sliced, steps=self.steps, cfg_scale=self.cfg_scale)


class RFHeadDistributionOutput(nn.Module):
    def __init__(
        self,
        depth: int,
        denoiser_width: int,
        d_cond: int,
        expansion_factor: int = 1,
        flow_dim: int = 2,
        internal_value_scale: float = 1.0,
        value_scale_cascade: bool = False,
        value_scale_cascade_steps: int = 512,
        cat_embs: bool = False,
        relu: bool = False,
        cond_dropout: float=0.0,
        steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.head = RFHead(
            depth=depth,
            d_model=denoiser_width,
            d_cond=d_cond,
            expansion_factor=expansion_factor,
            d_out=flow_dim,
            internal_value_scale=internal_value_scale,
            value_scale_cascade=value_scale_cascade,
            value_scale_cascade_steps=value_scale_cascade_steps,
            cat_embs=cat_embs,
            relu=relu,
            cond_dropout=cond_dropout,
        )
        self.head_dtype: torch.dtype | None = None
        self.steps = steps
        self.cfg_scale = cfg_scale


    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        cond: Float[torch.Tensor, "... d_cond"],
        steps: int | None = None,
        cfg_scale: float | None = None,
    ) -> RFHeadDistribution:
        steps = steps if steps is not None else self.steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale
        if self.head_dtype is None or self.head_dtype != cond.dtype:
            self.head_dtype = cond.dtype  # type: ignore
            self.head = self.head.to(cond.dtype)
        return RFHeadDistribution(rf_head=self.head, cond=cond, steps=steps, cfg_scale=cfg_scale)

# ---------------------------------------------------------------------------------------------------------------------
# Alternative Posterior Distribution Heads
# - GMM model similar to Flow Poke Transformer model/GIVT
# - Regression head
# ---------------------------------------------------------------------------------------------------------------------

class DistributionHead(nn.Module, ABC):

    def __init__(self, width: int, flow_dim: int):
        super().__init__()
        self.width = width
        self.flow_dim = flow_dim

    @abstractmethod
    def forward(self, cond: Float[torch.Tensor, "... d_model"]) -> Distribution:
        pass

class MixtureSameFamily(_MixtureSameFamily):
    def __getitem__(self, idx) -> "MixtureSameFamily":
        idx = idx if isinstance(idx, tuple) else (idx,)
        # Equivalent of MixtureSameFamily(self.mixture_distribution[idx], self.component_distribution[idx, :])
        return MixtureSameFamily(self.mixture_distribution[idx], self.component_distribution[idx + (slice(None),)])

class Categorical(_Categorical):
    def __getitem__(self, idx) -> "Categorical":
        idx = idx if isinstance(idx, tuple) else (idx,)
        # Equivalent of Categorical(logits=self.logits[idx, :])
        return Categorical(logits=self.logits[idx + (slice(None),)])  # type: ignore
    
class MultivariateNormal(_MultivariateNormal):
    def __getitem__(self, idx) -> "MultivariateNormal":
        idx = idx if isinstance(idx, tuple) else (idx,)
        # Equivalent of MultivariateNormal(self.loc[idx, :], scale_tril=self.scale_tril[idx, :, :])
        return MultivariateNormal(
            self.loc[idx + (slice(None),)],
            scale_tril=self.scale_tril[idx + (slice(None), slice(None))],  # type: ignore
        )

    @staticmethod
    def _gauss_legendre_nodes_weights(n: int, device, dtype):
        k = torch.arange(1, n, device=device, dtype=dtype)
        beta = k / torch.sqrt(4 * k * k - 1)

        J = torch.zeros((n, n), device=device, dtype=dtype)
        J.diagonal(1).copy_(beta)
        J.diagonal(-1).copy_(beta)

        eigvals, eigvecs = torch.linalg.eigh(J)
        x = eigvals
        w = 2 * (eigvecs[0, :] ** 2)
        return x, w

    @staticmethod
    def _bvn_cdf(a, b, rho, n_nodes: int):
        device = a.device
        dtype = torch.float64
        a = a.to(dtype)
        b = b.to(dtype)
        rho = rho.to(dtype)

        stdn = torch.distributions.Normal(
            loc=torch.tensor(0.0, device=device, dtype=dtype), scale=torch.tensor(1.0, device=device, dtype=dtype)
        )
        base = stdn.cdf(a) * stdn.cdf(b)

        r = rho.abs().clamp_max(1.0 - 1e-12)
        small = r < 1e-12
        if small.all():
            return base

        x, w = MultivariateNormal._gauss_legendre_nodes_weights(n_nodes, device=device, dtype=dtype)

        bro_shape = torch.broadcast_shapes(a.shape, b.shape, r.shape)
        expand_shape = (n_nodes,) + (1,) * len(bro_shape)
        xN = x.view(*expand_shape)
        wN = w.view(*expand_shape)

        s = torch.sign(rho)
        t = (0.5 * r).unsqueeze(0) * (xN + 1.0)
        t_signed = s.unsqueeze(0) * t

        one = torch.tensor(1.0, device=device, dtype=dtype)
        denom = torch.sqrt(one - t_signed**2)
        num = torch.exp(
            -(a.unsqueeze(0) ** 2 - 2 * t_signed * a.unsqueeze(0) * b.unsqueeze(0) + b.unsqueeze(0) ** 2)
            / (2 * (one - t_signed**2))
        )
        integrand = num / (2 * torch.pi * denom)

        I = (0.5 * r).unsqueeze(0) * (wN * integrand).sum(dim=0)
        out = base + s * I
        out = torch.where(small, base, out)
        return out

    def cdf(self, value, n_nodes: int = 32):
        if value.shape[-1] != 2:
            raise NotImplementedError("CDF approximation is only implemented for bivariate normal.")

        dtype_work = torch.float64
        value = value.to(dtype_work)
        mu = self.loc.to(dtype_work)
        Sigma = self.covariance_matrix.to(dtype_work)

        var_x = Sigma[..., 0, 0].clamp_min(0)
        var_y = Sigma[..., 1, 1].clamp_min(0)
        sig_x = torch.sqrt(var_x + 0.0)
        sig_y = torch.sqrt(var_y + 0.0)
        rho = (Sigma[..., 0, 1] / (sig_x * sig_y)).clamp(-1 + 1e-12, 1 - 1e-12)

        a = (value[..., 0] - mu[..., 0]) / sig_x
        b = (value[..., 1] - mu[..., 1]) / sig_y

        F = MultivariateNormal._bvn_cdf(a, b, rho, n_nodes=n_nodes)
        return F.to(self.loc.dtype)

class GMMDistributionHead(DistributionHead):

    def __init__(self, width: int, flow_dim: int, n_components: int = 4, dist_dtype: torch.dtype = torch.float64):
        super().__init__(width, flow_dim)
        self.n_components = n_components
        self.dist_dtype = dist_dtype

        self.proj = nn.Linear(width, n_components * (1 + flow_dim + (flow_dim * (flow_dim + 1)) // 2))

    def forward(self, cond: Float[torch.Tensor, "... d_model"]) -> MixtureSameFamily:
        x_out = self.proj(cond)
        B, L, _ = x_out.shape
        x = x_out.to(self.dist_dtype)
        logits, x = x[..., : self.n_components], x[..., self.n_components :]
        locs, scale_vals = (
            x[..., : self.n_components * self.flow_dim],
            x[..., self.n_components * self.flow_dim :],
        )
        locs = einops.rearrange(locs, "b l (n d) -> b l n d", d=self.flow_dim)
        scales_tril = scale_vals.new_zeros((B, L, self.n_components, self.flow_dim, self.flow_dim))
        # Map scale params to lower-triangular matrix
        scales_tril[..., *torch.tril_indices(self.flow_dim, self.flow_dim, device=x.device)] = scale_vals.view(
            B, L, self.n_components, (self.flow_dim * (self.flow_dim + 1)) // 2
        )
        # Softclip diagonal of covariance matrix to be > 0
        diag_mask = torch.eye(self.flow_dim, device=scales_tril.device, dtype=scales_tril.dtype)[None, None, None]
        scales_tril = scales_tril * (1 - diag_mask) + diag_mask * (squareplus(scales_tril) + 1e-4)
        return MixtureSameFamily(
            Categorical(logits=logits),
            MultivariateNormal(
                loc=locs,
                scale_tril=scales_tril,
            ),
        )

class RegressionDistribution(Distribution):
    def __init__(self, x: torch.Tensor):
        super().__init__()
        self.x = x

    def __getattribute__(self, name: str):
        if name != "log_prob":
            return super().__getattribute__(name)
        raise AttributeError()

    def __getitem__(self, idx) -> "RegressionDistribution":
        idx = idx if isinstance(idx, tuple) else (idx,)
        return RegressionDistribution(x=self.x[idx + (slice(None),)])  # type: ignore

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.x, x.to(self.x.dtype))

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        x = self.x
        if len(sample_shape) > 0:
            for _ in range(len(sample_shape)):
                x = x[None]
            x = x.expand(*sample_shape, *self.x.shape)
        return x

class RegressionHead(DistributionHead):

    def __init__(
        self,
        width: int,
        flow_dim: int,
        depth: int = 2,
        expansion_factor: int = 1,
        relu: bool = False,
        **kwargs,
    ):
        super().__init__(width, flow_dim)
        self.depth = depth
        d_ff = int(width * expansion_factor)
        act = nn.ReLU(inplace=True) if relu else nn.GELU(approximate="tanh")

        self.in_proj = nn.Linear(width, width, bias=False)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width, eps=1e-6),
                    nn.Linear(width, d_ff, bias=False),
                    act,
                    zero_init(nn.Linear(d_ff, width, bias=False)),
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(width, eps=1e-6).to(torch.float32)
        self.out_proj = zero_init(nn.Linear(width, flow_dim, bias=False))

    def forward(self, cond: Float[torch.Tensor, "... d_model"]) -> Distribution:
        x = self.in_proj(cond)
        for layer in self.layers:
            x = x + layer(x)
        norm_dtype = torch.promote_types(x.dtype, torch.float32)
        x = _layer_norm_wrapper(x, self.out_norm, norm_dtype, x.dtype)
        x = self.out_proj(x)
        return RegressionDistribution(x=x)

# ---------------------------------------------------------------------------------------------------------------------
# Step-by-Step Myriad Model
# ---------------------------------------------------------------------------------------------------------------------

class MyriadStepByStep(nn.Module):

    def __init__(
        self,
        # general parameters
        width: int,
        # transformer parameters
        depth: int,
        transformer_params: dict[str, Any] = {},
        # distribution head parameters
        flow_dim: int = 2,
        distribution_type: Literal["fm", "gmm", "regression"] = "fm",
        distribution_head_params: dict[str, Any] = {},
        # image feature extractor parameters
        train_image_feature_extractor: bool = True,
        image_feature_extractor_params: dict[str, Any] = {},
        # other parameters
        cache_train_attn_mask: bool = True,
    ) -> None:

        super().__init__()

        # backbone consists of three components:
        # 1) a DINO feature extractor
        # 2) a Transformer motion model
        # 3) a RF Distribution head

        # 1) Feature extractor
        self.train_image_feature_extractor = train_image_feature_extractor
        self.image_embedder = DinoFeatureExtractor(
            requires_grad=train_image_feature_extractor, **image_feature_extractor_params
        )
        d_img_feat = self.image_embedder.embed_dim

        # 2) Transformer motion model
        self.transformer = FusedTransformer(
            width=width,
            depth=depth,
            aux_feat_dim=d_img_feat,
            **transformer_params
        )

        # 3) Distribution head
        self.flow_dim = flow_dim
        if distribution_type == "fm":
            self.distribution_head = RFHeadDistributionOutput(
                denoiser_width=width, flow_dim=flow_dim, **distribution_head_params
            )
        elif distribution_type == "gmm":
            self.distribution_head = GMMDistributionHead(width=width, flow_dim=flow_dim, **distribution_head_params)
        elif distribution_type == "regression":
            self.distribution_head = RegressionHead(width=width, flow_dim=flow_dim, **distribution_head_params)
        else:
            raise ValueError(f"Invalid distribution_type: {distribution_type}")

        self.cache_train_attn_mask = cache_train_attn_mask
        self._cached_train_attn_mask = None

    def embed_image(self, img: Float[torch.Tensor, "b c h w"]) -> dict[str, torch.Tensor]:
        """
        Get the image features, pass to forward() etc. Image is expected to be square and in [-1, 1].
        If the resolution is not the expected one, it will be resized automatically inside the image embedder.
        """
        with nullcontext() if self.train_image_feature_extractor else torch.no_grad():
            features: Float[torch.Tensor, "b d h w"] = self.image_embedder(img)
            B, D, H, W = features.shape
            feature_pos: Float[torch.Tensor, "(h w) 2"] = make_axial_pos_2d(H, W, device=features.device)
        return {
            "feat": features,
            "pos": einops.repeat(feature_pos, "(h w) d -> b h w d", b=B, h=H, w=W),
            "L": H * W,
        }
    
    def forward(
        self,
        flow: Float[torch.Tensor, "b l 2"],
        pos: Float[torch.Tensor, "b l 2"],
        pos_orig: Float[torch.Tensor, "b l 2"],
        t: Float[torch.Tensor, "b l"],
        is_query: Bool[torch.Tensor, "b l"],
        camera_static: Bool[torch.Tensor, "b"],
        mask,
        d_img: dict[str, torch.Tensor],
        track_id: Int[torch.Tensor, "b l"],
        track_id_emb_table: Float[torch.Tensor, "n_tracks d"] | None = None,
        i_kv: int | None = None,
        compute_cross: bool = True,
    ) -> Distribution:

        aux_feat = d_img["feat"]
        aux_pos = d_img["pos"]

        x = self.transformer(
            x=flow,
            x_cross=einops.rearrange(aux_feat, "b d h w -> b (h w) d"),
            pos=pos,
            pos_orig=pos_orig,
            pos_cross=aux_pos,
            is_query=is_query,
            track_id=track_id,
            camera_static=camera_static,
            time=t,
            block_mask=mask,
            i_kv=i_kv,
            compute_cross=compute_cross,
            track_id_emb_table=track_id_emb_table,
        )

        return self.distribution_head(cond=x)

    @torch.no_grad()
    def predict_parallel(
        self,
        poke_pos: Float[torch.Tensor, "b l c"],
        poke_flow: Float[torch.Tensor, "b l c"],
        poke_pos_orig: Float[torch.Tensor, "b l c"],
        poke_t: Float[torch.Tensor, "b l"],
        poke_id: Int[torch.Tensor, "b l"],
        query_pos: Float[torch.Tensor, "b l_q c"],
        query_pos_orig: Float[torch.Tensor, "b l_q c"],
        query_t: Float[torch.Tensor, "b l_q"],
        query_id: Int[torch.Tensor, "b l_q"],
        camera_static: Bool[torch.Tensor, "b"] | bool,
        d_img: dict[str, torch.Tensor],
        track_id_emb_table: Float[torch.Tensor, "n_tracks d"] | None = None,
        mask: BlockMask | None = None,
    ) -> Distribution:
        B, L_P, C = poke_pos.shape
        L_Q, L_prefix = query_pos.shape[1], d_img["L"]

        if isinstance(camera_static, bool):
            camera_static = torch.full((B,), camera_static, dtype=torch.bool, device=poke_pos.device)

        if mask is None:
            mask_mod = inference_query_causal_mask_mod(l_prefix=L_prefix, l_seq=L_P)
            mask = create_block_mask(
                mask_mod, B=1, H=1, Q_LEN=L_prefix + L_P + L_Q, KV_LEN=L_prefix + L_P + L_Q, device=poke_pos.device
            )

        return self.forward(
            flow=torch.cat([poke_flow, poke_flow.new_zeros((B, L_Q, C))], dim=1),
            pos=torch.cat([poke_pos, query_pos], dim=1),
            pos_orig=torch.cat([poke_pos_orig, query_pos_orig], dim=1),
            t=torch.cat([poke_t, query_t], dim=1),
            track_id=torch.cat([poke_id, query_id], dim=1),
            is_query=torch.cat(
                [poke_pos.new_zeros((B, L_P), dtype=torch.bool), poke_pos.new_ones((B, L_Q), dtype=torch.bool)], dim=1
            ),
            camera_static=camera_static,
            mask=mask,
            d_img=d_img,
            track_id_emb_table=track_id_emb_table,
        )[:, L_P:]

    @torch.no_grad()
    def predict_simulate(
        self,
        n_traj: int,
        ts: Float[torch.Tensor, "b t"],
        given_pos: Float[torch.Tensor, "b (t n_traj) 2"],
        camera_static: Bool[torch.Tensor, "b"] | bool,
        d_img: dict[str, torch.Tensor],
        verbose: bool = True,
        d_steps: int | None = None,
        teacher_forcing_pos: Float[torch.Tensor, "b t n_traj 2"] | None = None,
    ) -> Float[torch.Tensor, "b t n_traj 2"]:
        is_fm = isinstance(self.distribution_head, RFHeadDistributionOutput)

        B, L_known, C = given_pos.shape
        device = given_pos.device
        assert L_known >= n_traj, "At least the original positions must be given"
        assert given_pos.shape[0] == B, f"{given_pos.shape=} != ({B}, *, {self.flow_dim})"
        assert given_pos.shape[2] == self.flow_dim, f"{given_pos.shape=} != ({B}, *, {self.flow_dim})"
        assert ts.shape[0] == B, f"{ts.shape=} != ({B}, T)"
        assert ts.device == device, f"{ts.device=} != {device=}"

        N_traj, T = n_traj, ts.shape[1]
        L_known_steps = L_known - N_traj
        L_prefix = d_img["L"]

        if isinstance(camera_static, bool):
            camera_static = torch.full((B,), camera_static, dtype=torch.bool, device=device)
        assert camera_static.shape == (B,), f"{camera_static.shape=} != ({B},)"
        assert camera_static.device == device, f"{camera_static.device=} != {device=}"
        for k in (k for k in d_img if k != "L"):
            assert d_img[k].shape[0] == B, f"{d_img[k].shape=} != ({B}, *, {d_img[k].shape[1:]})"
            assert d_img[k].device == device, f"{d_img[k].device=} != {device=}"

        # precompute embedding table to use for all rollouts
        track_id_emb_table = self.transformer.embedder.get_id_embedding_table(device=device, dtype=given_pos.dtype)

        pos: Float[torch.Tensor, "b t n c"] = given_pos.new_zeros((B, T, N_traj, C))
        # prefill known positions
        pos.view(B, T * N_traj, C)[:, :L_known] = given_pos
        prediction = pos.clone() if teacher_forcing_pos is not None else pos
        if teacher_forcing_pos is not None:
            if teacher_forcing_pos.shape != pos.shape:
                raise ValueError(
                    f"{teacher_forcing_pos.shape=} must match rollout shape {pos.shape}"
                )
            if teacher_forcing_pos.device != device:
                raise ValueError(
                    f"{teacher_forcing_pos.device=} must match rollout device {device}"
                )

        # Helper tensors
        pos_orig: Float[torch.Tensor, "b (t n) c"] = einops.repeat(pos[:, 0], "b n c -> b (t n) c", t=T)
        track_id: Int[torch.Tensor, "b (t n)"] = einops.repeat(
            torch.arange(N_traj, device=device), "n -> b (t n)", b=B, t=T
        )
        t: Float[torch.Tensor, "b (t n)"] = einops.repeat(ts, "b t -> b (t n)", n=N_traj)

        # Build RFHead timestep embedding cache (FM only)
        rf_steps = None
        if is_fm:
            rf_steps = d_steps if d_steps is not None else self.distribution_head.steps
            self.distribution_head.head.build_timestep_cache(
                steps=rf_steps, dims=(B, 1), device=device, dtype=given_pos.dtype
            )

        # Grow KV cache to accommodate the full sequence (image prefix + all motion tokens)
        self.transformer.grow_kv_cache(b=B, l=(L_prefix + T * N_traj))

        for i_step in trange(T * N_traj - L_known, desc="Simulating steps", disable=not verbose):
            i_total = i_step + L_known_steps
            i_t = i_total // N_traj
            i_traj = i_total % N_traj

            flow: Float[torch.Tensor, "b (t-1 n) c"] = einops.rearrange(
                pos[:, 1:] - pos[:, :-1],
                "b t n c -> b (t n) c",
            )

            pos_flat = einops.rearrange(pos, "b t n c -> b (t n) c")

            if i_step == 0:
                # Prefill: process all known pokes + first query in one pass
                # Use BlockMask so FlexAttention handles all-masked image rows gracefully
                mask_mod = inference_query_causal_mask_mod(l_prefix=L_prefix, l_seq=L_known_steps)
                block_mask = create_block_mask(
                    mask_mod,
                    B=1,
                    H=1,
                    Q_LEN=L_prefix + L_known_steps + 1,
                    KV_LEN=L_prefix + L_known_steps + 1,
                    device=device,
                )

                dist = self.forward(
                    flow=torch.cat([flow[:, :L_known_steps], flow.new_zeros((B, 1, C))], dim=1),
                    pos=torch.cat([pos_flat[:, :L_known_steps], pos_flat[:, i_total : i_total + 1]], dim=1),
                    pos_orig=torch.cat([pos_orig[:, :L_known_steps], pos_orig[:, i_total : i_total + 1]], dim=1),
                    t=torch.cat([t[:, :L_known_steps], t[:, i_total : i_total + 1]], dim=1),
                    is_query=torch.cat(
                        [
                            pos_flat.new_zeros((B, L_known_steps), dtype=torch.bool),
                            pos_flat.new_ones((B, 1), dtype=torch.bool),
                        ],
                        dim=1,
                    ),
                    camera_static=camera_static,
                    mask=block_mask,
                    d_img=d_img,
                    track_id=torch.cat([track_id[:, :L_known_steps], track_id[:, i_total : i_total + 1]], dim=1),
                    track_id_emb_table=track_id_emb_table,
                    i_kv=0,
                    compute_cross=True,
                )[:, -1:]
            else:
                # Incremental step: 2 tokens [prev_query_as_poke, new_query]
                i_kv = L_prefix + L_known_steps + i_step - 1
                mask = torch.ones((1, 1, 2, i_kv + 2), dtype=torch.bool, device=device)
                mask[:, :, 0, i_kv + 1] = False

                dist = self.forward(
                    flow=torch.cat([flow[:, i_total - 1 : i_total], flow.new_zeros((B, 1, C))], dim=1),
                    pos=pos_flat[:, i_total - 1 : i_total + 1],
                    pos_orig=pos_orig[:, i_total - 1 : i_total + 1],
                    t=t[:, i_total - 1 : i_total + 1],
                    is_query=torch.cat(
                        [
                            pos_flat.new_zeros((B, 1), dtype=torch.bool),
                            pos_flat.new_ones((B, 1), dtype=torch.bool),
                        ],
                        dim=1,
                    ),
                    camera_static=camera_static,
                    mask=mask,
                    d_img=d_img,
                    track_id=track_id[:, i_total - 1 : i_total + 1],
                    track_id_emb_table=track_id_emb_table,
                    i_kv=i_kv,
                    compute_cross=False,
                )[:, -1:]

            if is_fm:
                dist.steps = rf_steps
            delta = dist.sample().squeeze(1)
            next_pos = pos[:, i_t, i_traj] + delta
            prediction[:, i_t + 1, i_traj] = next_pos
            if teacher_forcing_pos is None:
                pos[:, i_t + 1, i_traj] = next_pos
            else:
                # Oracle-history diagnostic: retain the actual one-step prediction,
                # but put ground truth into the autoregressive cache for later tokens.
                pos[:, i_t + 1, i_traj] = teacher_forcing_pos[:, i_t + 1, i_traj]
        return prediction

# ---------------------------------------------------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------------------------------------------------

MyriadStepByStep_Large = partial(
    MyriadStepByStep,
    width=1024,
    depth=24,
    train_image_feature_extractor=True,
    distribution_type="fm",
    transformer_params={
        "d_head": 128,
        "out_mlp_depth": 2,
        "ff_expand": 4,
        "ada_norm_size": 1024,
        "time_norm_size": -1,
        "input_scale": 10,
        "emb_depth": 3,
        "track_id_embedding": True,
        "max_num_track_ids": 256,
        "scaled_cosine_sim": True,
    },
    image_feature_extractor_params={
        "model_version": "dinov3_vitl16",
        "image_size": 384,  # uses cropped in images
    },
    distribution_head_params={
        "depth": 3,
        "d_cond": 1024,
        "expansion_factor": 1,
        "internal_value_scale": 1,
        "value_scale_cascade": True,
        "value_scale_cascade_steps": 512,
        "cat_embs": False,
        "relu": False,
        "cond_dropout": 0.,
        "steps": 50,
        "cfg_scale": 1.0,
    },
)

MyriadStepByStep_Large_Billiard = partial(
    MyriadStepByStep,
    width=1024,
    depth=24,
    train_image_feature_extractor=True,
    distribution_type="fm",
    transformer_params={
        "d_head": 128,
        "out_mlp_depth": 2,
        "ff_expand": 4,
        "ada_norm_size": 1024,
        "time_norm_size": -1,
        "input_scale": 10,
        "emb_depth": 3,
        "track_id_embedding": True,
        "max_num_track_ids": 256,
        "scaled_cosine_sim": True,
    },
    image_feature_extractor_params={
        "model_version": "dinov3_vitl16",
        "image_size": 512,  # uses full size images
    },
    distribution_head_params={
        "depth": 3,
        "d_cond": 1024,
        "expansion_factor": 1,
        "internal_value_scale": 500,   # only difference is that we scale up values to make Billiard motion larger
        "value_scale_cascade": True,
        "value_scale_cascade_steps": 512,
        "cat_embs": False,
        "relu": False,
        "cond_dropout": 0.,
        "steps": 50,
        "cfg_scale": 1.0,
    },
)

MyriadStepByStep_Large_Billiard_PhysicsBias = partial(
    MyriadStepByStep,
    width=1024,
    depth=24,
    train_image_feature_extractor=True,
    distribution_type="fm",
    transformer_params={
        "d_head": 128,
        "out_mlp_depth": 2,
        "ff_expand": 4,
        "ada_norm_size": 1024,
        "time_norm_size": -1,
        "input_scale": 10,
        "emb_depth": 3,
        "track_id_embedding": True,
        "max_num_track_ids": 256,
        "scaled_cosine_sim": True,
        "use_physics_bias": True,
        "physics_bias_hidden_dim": 64,
        "physics_bias_depth": 2,
        "physics_bias_time_scale": 50.0,
        "physics_bias_max_abs": 1.0,
        "physics_bias_ball_radius": 0.033,
        "physics_bias_num_layers": 4,
        "physics_bias_initial_scale": 0.5,
        "physics_bias_history_window": 8,
        "physics_bias_history_decay": 4.0,
        "physics_bias_query_chunk_size": 64,
        "physics_bias_checkpoint_chunks": False,
    },
    image_feature_extractor_params={
        "model_version": "dinov3_vitl16",
        "image_size": 512,
    },
    distribution_head_params={
        "depth": 3,
        "d_cond": 1024,
        "expansion_factor": 1,
        "internal_value_scale": 500,
        "value_scale_cascade": True,
        "value_scale_cascade_steps": 512,
        "cat_embs": False,
        "relu": False,
        "cond_dropout": 0.0,
        "steps": 50,
        "cfg_scale": 1.0,
    },
)

MyriadStepByStep_Large_Billiard_LongHistoryBias = partial(
    MyriadStepByStep,
    width=1024,
    depth=24,
    train_image_feature_extractor=True,
    distribution_type="fm",
    transformer_params={
        "d_head": 128,
        "out_mlp_depth": 2,
        "ff_expand": 4,
        "ada_norm_size": 1024,
        "time_norm_size": -1,
        "input_scale": 10,
        "emb_depth": 3,
        "track_id_embedding": True,
        "max_num_track_ids": 256,
        "scaled_cosine_sim": True,
        "use_physics_bias": False,
        "use_long_history_bias": True,
        "long_history_bias_hidden_dim": 64,
        "long_history_bias_depth": 2,
        "long_history_bias_time_scale": 50.0,
        "long_history_bias_max_abs": 0.25,
        "long_history_bias_num_layers": 4,
        "long_history_bias_initial_scale": 0.25,
        "long_history_bias_window": 8,
        "long_history_bias_query_chunk_size": 64,
        "long_history_bias_cross_track_initial_scale": 0.1,
        "long_history_bias_checkpoint_chunks": False,
    },
    image_feature_extractor_params={
        "model_version": "dinov3_vitl16",
        "image_size": 512,
    },
    distribution_head_params={
        "depth": 3,
        "d_cond": 1024,
        "expansion_factor": 1,
        "internal_value_scale": 500,
        "value_scale_cascade": True,
        "value_scale_cascade_steps": 512,
        "cat_embs": False,
        "relu": False,
        "cond_dropout": 0.0,
        "steps": 50,
        "cfg_scale": 1.0,
    },
)

MyriadStepByStep_Large_Billiard_CombinedBias = partial(
    MyriadStepByStep,
    width=1024,
    depth=24,
    train_image_feature_extractor=True,
    distribution_type="fm",
    transformer_params={
        "d_head": 128,
        "out_mlp_depth": 2,
        "ff_expand": 4,
        "ada_norm_size": 1024,
        "time_norm_size": -1,
        "input_scale": 10,
        "emb_depth": 3,
        "track_id_embedding": True,
        "max_num_track_ids": 256,
        "scaled_cosine_sim": True,
        "use_physics_bias": True,
        "use_long_history_bias": True,
        "physics_bias_hidden_dim": 64,
        "physics_bias_depth": 2,
        "physics_bias_time_scale": 50.0,
        "physics_bias_max_abs": 1.0,
        "physics_bias_ball_radius": 0.033,
        "physics_bias_num_layers": 4,
        "physics_bias_initial_scale": 0.5,
        "physics_bias_history_window": 8,
        "physics_bias_history_decay": 4.0,
        "physics_bias_query_chunk_size": 64,
        "long_history_bias_hidden_dim": 64,
        "long_history_bias_depth": 2,
        "long_history_bias_time_scale": 50.0,
        "long_history_bias_max_abs": 0.25,
        "long_history_bias_num_layers": 4,
        "long_history_bias_initial_scale": 0.25,
        "long_history_bias_window": 8,
        "long_history_bias_query_chunk_size": 64,
        "long_history_bias_cross_track_initial_scale": 0.1,
        "use_combined_bias_gates": True,
        "combined_physics_initial_scale": 0.5,
        "combined_long_history_initial_scale": 0.25,
        "combined_gate_hidden_dim": 32,
    },
    image_feature_extractor_params={
        "model_version": "dinov3_vitl16",
        "image_size": 512,
    },
    distribution_head_params={
        "depth": 3,
        "d_cond": 1024,
        "expansion_factor": 1,
        "internal_value_scale": 500,
        "value_scale_cascade": True,
        "value_scale_cascade_steps": 512,
        "cat_embs": False,
        "relu": False,
        "cond_dropout": 0.0,
        "steps": 50,
        "cfg_scale": 1.0,
    },
)
