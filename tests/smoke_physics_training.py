"""Two-step small-model optimizer smoke test for eager/compiled physics bias."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from myriad.model import FusedTransformer
from train import configure_physics_training


class SmallPhysicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_embedder = nn.Identity()
        self.transformer = FusedTransformer(
            width=48, depth=2, aux_feat_dim=32, d_head=16, out_mlp_depth=1,
            ff_expand=2, track_id_embedding=False, use_physics_bias=True,
            physics_bias_query_chunk_size=2,
        )
        self.distribution_head = nn.Linear(48, 2)
        for layer in self.transformer.mid_level:
            nn.init.normal_(layer.out_proj.weight, std=0.01)

    def forward(self, batch, mask):
        cond = self.transformer(**batch, block_mask=mask)
        return self.distribution_head(cond)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPhysicsModel().to(device).train()
    trainable = configure_physics_training(model, "physics-only", 1)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    b, l_motion, l_cross = 2, 4, 4
    batch = dict(
        x=torch.randn(b, l_motion, 2, device=device),
        x_cross=torch.randn(b, l_cross, 32, device=device),
        pos=torch.rand(b, l_motion, 2, device=device),
        pos_orig=torch.rand(b, l_motion, 2, device=device),
        pos_cross=torch.rand(b, 2, 2, 2, device=device),
        is_query=torch.tensor([[False, False, True, True]], device=device).expand(b, -1),
        track_id=torch.arange(l_motion, device=device).expand(b, -1) % 2,
        camera_static=torch.ones(b, dtype=torch.bool, device=device),
        time=torch.arange(l_motion, device=device).expand(b, -1).float(),
    )
    target = torch.randn(b, l_motion, 2, device=device)
    if device.type == "cuda":
        mask = create_block_mask(lambda bi, h, q, k: q >= l_cross, B=1, H=1,
                                 Q_LEN=l_cross + l_motion, KV_LEN=l_cross + l_motion, device=device)
    else:
        mask = torch.ones(1, 1, l_cross + l_motion, l_cross + l_motion, dtype=torch.bool, device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(batch, mask)
            loss = (prediction.float() - target).square().mean()
        loss.backward()
        grad = model.transformer.physics_bias_generator.mlp[-1].weight.grad
        if grad is None or not torch.isfinite(grad).all() or torch.count_nonzero(grad) == 0:
            raise AssertionError("physics bias did not receive a finite non-zero gradient")
        optimizer.step()
        return loss.detach()

    losses = [float(step()) for _ in range(2)]
    print({"compiled": False, "device": str(device), "losses": losses,
           "trainable_parameters": sum(p.numel() for p in trainable)})


if __name__ == "__main__":
    main()
