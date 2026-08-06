# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2026 Stefan Baumann et al., CompVis @ LMU Munich

import os
import atexit
import json
import math
from pathlib import Path
import logging
import random
import subprocess
from datetime import datetime

import click
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import create_block_mask
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange, repeat


def set_requires_grad(module, value: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def configure_physics_training(model, train_mode: str, unfreeze_last_n_layers: int = 6):
    """Freeze a relation-bias model for a stable, process-level training stage."""
    set_requires_grad(model, False)
    generator = model.transformer.physics_bias_generator
    if generator is None:
        raise ValueError("Relation-bias training requires a model with an attention-bias generator")
    set_requires_grad(generator, True)
    if train_mode == "physics-only":
        pass
    elif train_mode == "finetune":
        depth = len(model.transformer.mid_level)
        if unfreeze_last_n_layers <= 0 or unfreeze_last_n_layers > depth:
            raise ValueError(f"unfreeze_last_n_layers must be in [1, {depth}], got {unfreeze_last_n_layers}")
        for layer in model.transformer.mid_level[-unfreeze_last_n_layers:]:
            set_requires_grad(layer, True)
        set_requires_grad(model.transformer.out_proj, True)
        set_requires_grad(model.distribution_head, True)
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise AssertionError("No trainable parameters configured")
    if train_mode == "physics-only":
        bad = [name for name, p in model.named_parameters()
               if p.requires_grad and not name.startswith("transformer.physics_bias_generator.")]
        if bad:
            raise AssertionError(f"Non-physics parameters are trainable in physics-only mode: {bad}")
    return trainable


def configure_combined_bias_training(model, train_mode: str, unfreeze_last_n_layers: int = 6):
    """Train both physical and temporal relation-bias branches plus source gates."""
    set_requires_grad(model, False)
    physics = model.transformer.physics_bias_generator
    temporal = model.transformer.long_history_bias_generator
    if physics is None or temporal is None:
        raise ValueError("Combined bias training requires both relation-bias generators")
    set_requires_grad(physics, True)
    set_requires_grad(temporal, True)
    for name in ("physics_source_logit", "long_history_source_logit"):
        parameter = getattr(model.transformer, name, None)
        if parameter is None:
            raise ValueError(f"Combined model is missing {name}")
        parameter.requires_grad_(True)
    if train_mode == "finetune":
        depth = len(model.transformer.mid_level)
        if unfreeze_last_n_layers <= 0 or unfreeze_last_n_layers > depth:
            raise ValueError(f"unfreeze_last_n_layers must be in [1, {depth}], got {unfreeze_last_n_layers}")
        for layer in model.transformer.mid_level[-unfreeze_last_n_layers:]:
            set_requires_grad(layer, True)
        set_requires_grad(model.transformer.out_proj, True)
        set_requires_grad(model.distribution_head, True)
    elif train_mode != "physics-only":
        raise ValueError(f"Unknown train_mode: {train_mode}")
    return [p for p in model.parameters() if p.requires_grad]


def _strip_uniform_module_prefix(state_dict):
    keys = list(state_dict)
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _remap_modelscope_dinov3_layer_keys(model, state_dict):
    """Align the ModelScope DINOv3 wrapper's extra ``model`` level safely.

    The official MYRIAD checkpoint stores DINO blocks as
    ``image_embedder.model.model.layer.*``.  The compatible ModelScope
    Transformers wrapper stores those same blocks below
    ``image_embedder.model.model.model.layer.*`` while keeping embeddings and
    final norm at the original paths.  Rewrite only when the destination is
    an actual parameter of the instantiated model, so normal HF checkpoints
    are unchanged.
    """
    source = "image_embedder.model.model.layer."
    destination = "image_embedder.model.model.model.layer."
    model_keys = set(model.state_dict())
    remapped = dict(state_dict)
    for key in list(state_dict):
        if not key.startswith(source):
            continue
        mapped_key = destination + key[len(source):]
        if mapped_key in model_keys and key not in model_keys:
            remapped[mapped_key] = remapped.pop(key)
    return remapped


def load_init_checkpoint(model, path):
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = _strip_uniform_module_prefix(state_dict)
    state_dict = _remap_modelscope_dinov3_layer_keys(model, state_dict)

    # Preserve older relation columns and zero-initialize newly added causal
    # kinematics so 5- and 11-feature physics checkpoints remain loadable.
    physics_input_key = "transformer.physics_bias_generator.mlp.0.weight"
    target_state = model.state_dict()
    if physics_input_key in state_dict and physics_input_key in target_state:
        source = state_dict[physics_input_key]
        target = target_state[physics_input_key]
        if (source.shape[:-1] == target.shape[:-1]
                and source.shape[-1] in (5, 11)
                and source.shape[-1] < target.shape[-1]):
            expanded = source.new_zeros(target.shape)
            expanded[..., :source.shape[-1]] = source
            state_dict[physics_input_key] = expanded

    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "transformer.physics_bias_generator.",
        "transformer.long_history_bias_generator.",
        "transformer.physics_source_logit",
        "transformer.long_history_source_logit",
    )
    invalid_missing = [key for key in incompatible.missing_keys
                       if not key.startswith(allowed_missing_prefixes)]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unsafe init checkpoint: invalid missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible


def current_git_commit():
    project_dir = Path(__file__).resolve().parent
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={project_dir}", "-C", str(project_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_torch_save(payload, destination: Path) -> None:
    """Save without exposing a partial checkpoint at the final path."""
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def endless_iter(iterable):
    while True:
        yield from iterable


def make_scheduler(optimizer, lr, warmup_steps, max_steps, scheduler_type="linear"):
    """Build a LR scheduler.

    scheduler_type="linear":  linear warmup → linear decay to 0
    scheduler_type="cosine":  linear warmup → cosine annealing to 1e-8
    """
    has_warmup = warmup_steps > 0
    has_decay = max_steps is not None

    if scheduler_type == "cosine":
        if has_warmup and has_decay:
            warmup = LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
            decay = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=1e-8)
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        elif has_warmup:
            return LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
        elif has_decay:
            return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-8)
        else:
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:  # linear
        if has_warmup and has_decay:
            warmup = LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
            decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, max_steps - warmup_steps))
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        elif has_warmup:
            return LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
        elif has_decay:
            return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        else:
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


# ---------------------------------------------------------------------------
# Model-specific step protocols
# Each factory receives (model, real_model, device, device_type, is_distributed)
# and returns (init_caches, compute_step).
# ---------------------------------------------------------------------------

def fpt_make_train_fns(model, real_model, device, device_type, is_distributed):
    """Step protocol for the original FlowPokeTransformer (flow_poke.model)."""
    from flow_poke.model import query_causal_mask_mod as fpt_mask_mod

    def init_caches(batch):
        B = batch["pos_poke"].size(0)
        L_poke = batch["pos_poke"].size(1)
        L_query = math.prod(batch["pos_query"].shape[1:3])
        block_mask = create_block_mask(
            fpt_mask_mod(sequence_length=L_poke, n_query=batch["pos_query"].size(2)),
            B=1, H=1,
            Q_LEN=L_poke + L_query,
            KV_LEN=L_poke + L_query,
            device=device,
        )
        is_query = repeat(
            torch.cat([
                torch.zeros(L_poke, dtype=torch.bool, device=device),
                torch.ones(L_query, dtype=torch.bool, device=device),
            ]),
            "l -> b l", b=B,
        )
        return block_mask, is_query, L_poke

    def compute_step(batch, block_mask, is_query, L_poke, compute_metrics=False, flow_mask=None):
        pos = torch.cat([
            batch["pos_poke"],
            rearrange(batch["pos_query"], "b n_p n_q c -> b (n_p n_q) c"),
        ], dim=1)
        flow_target = rearrange(batch["flow_query"], "b n_p n_q c -> b (n_p n_q) c")
        flow = torch.cat([batch["flow_poke"], torch.zeros_like(flow_target)], dim=1)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            d_img = real_model.embed_image(batch["x"])
            distribution = model(
                pos=pos, flow=flow, is_query=is_query,
                camera_static=batch["camera_static"],
                mask=block_mask, d_img=d_img,
            )

        loss = -distribution[:, L_poke:].log_prob(flow_target).mean()

        if not compute_metrics:
            return loss

        with torch.no_grad():
            metrics = {}
            samples = distribution[:, L_poke:].sample()
            epe = (samples - flow_target).norm(p=2, dim=-1)
            metrics["epe"] = epe.mean().detach()
            for alpha in [0.1, 0.01, 0.001]:
                metrics[f"pck@{alpha}"] = epe.le(alpha).float().mean().detach()
            metrics["flow_mag_gt"] = flow_target.norm(p=2, dim=-1).mean().detach()
            metrics["flow_mag_pred"] = samples.norm(p=2, dim=-1).mean().detach()
            metrics["frac_static_camera"] = batch["camera_static"].float().mean().detach()

        return loss, metrics

    return init_caches, compute_step


def myriad_make_train_fns(model, real_model, device, device_type, is_distributed):
    """Step protocol for the Myriad MyriadStepByStep models (myriad.model)."""
    from myriad.model import query_causal_mask_mod as myriad_mask_mod

    def init_caches(batch):
        B = batch["pos_poke"].size(0)
        L_poke = batch["pos_poke"].size(1)
        L_query = batch["pos_query"].size(1)
        assert L_query == L_poke, f"Expected L_query == L_poke, got {L_query} vs {L_poke}"

        d_img = real_model.embed_image(batch["x"].to(device))
        L_prefix = d_img["L"]
        block_mask = create_block_mask(
            myriad_mask_mod(l_prefix=L_prefix, l_seq=L_poke, n_query=1),
            B=1, H=1,
            Q_LEN=L_prefix + L_poke + L_query,
            KV_LEN=L_prefix + L_poke + L_query,
            device=device,
        )
        is_query = repeat(
            torch.cat([
                torch.zeros(L_poke, dtype=torch.bool, device=device),
                torch.ones(L_query, dtype=torch.bool, device=device),
            ]),
            "l -> b l", b=B,
        )
        return block_mask, is_query, L_poke

    def compute_step(batch, block_mask, is_query, L_poke, compute_metrics=False, flow_mask=None):
        pos_poke, pos_query = batch["pos_poke"], batch["pos_query"]
        if pos_query.ndim > pos_poke.ndim:
            pos_query = rearrange(pos_query, "b n_p n_q ... -> b (n_p n_q) ...")
        pos = torch.cat([pos_poke, pos_query], dim=1)

        pos_orig = (
            torch.cat([batch["pos_orig_poke"], batch["pos_orig_query"]], dim=1)
            if "pos_orig_poke" in batch and "pos_orig_query" in batch else None
        )
        t = (
            torch.cat([batch["t_poke"], batch["t_query"]], dim=1)
            if "t_poke" in batch and "t_query" in batch else None
        )
        track_id = (
            torch.cat([batch["id_poke"], batch["id_query"]], dim=1)
            if "id_poke" in batch and "id_query" in batch else None
        )

        flow_poke, flow_query = batch["flow_poke"], batch["flow_query"]
        if flow_query.ndim > flow_poke.ndim:
            flow_query = rearrange(flow_query, "b n_p n_q ... -> b (n_p n_q) ...")
        flow_target = flow_query
        flow = torch.cat([flow_poke, torch.zeros_like(flow_target)], dim=1)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            d_img = real_model.embed_image(batch["x"])
            distribution = model(
                pos=pos, pos_orig=pos_orig, t=t, track_id=track_id,
                flow=flow, is_query=is_query,
                camera_static=batch["camera_static"],
                mask=block_mask, d_img=d_img, track_id_emb_table=None,
            )
            per_token_loss = distribution[:, L_poke:].loss(flow_target)
            if flow_mask is not None:
                valid = flow_mask.sum()
                loss = (
                    (per_token_loss * flow_mask).sum() / valid
                    if valid > 0 else per_token_loss.new_zeros(())
                )
            else:
                loss = per_token_loss.mean()

        if not compute_metrics:
            return loss

        with torch.no_grad():
            metrics = {}
            samples = distribution[:, L_poke:].sample()
            epe = (samples - flow_target).norm(dim=-1)
            metrics["epe"] = epe.mean().detach()
            for alpha in [0.1, 0.01, 0.001]:
                metrics[f"pck@{alpha}"] = epe.le(alpha).float().mean().detach()
            metrics["flow_mag_gt"] = flow_target.norm(p=2, dim=-1).mean().detach()
            metrics["flow_mag_pred"] = samples.norm(p=2, dim=-1).mean().detach()
            metrics["frac_static_camera"] = batch["camera_static"].float().mean().detach()
            collision_mask = batch.get("collision_token_mask")
            if collision_mask is not None:
                collision_mask = collision_mask.bool()
                if collision_mask.shape != epe.shape:
                    raise AssertionError(
                        f"Collision token mask {collision_mask.shape} does not match EPE {epe.shape}"
                    )
                collision_count = collision_mask.sum()
                noncollision_mask = ~collision_mask
                noncollision_count = noncollision_mask.sum()
                metrics["collision_token_fraction"] = collision_mask.float().mean().detach()
                metrics["collision_loss"] = (
                    (per_token_loss * collision_mask).sum() / collision_count.clamp_min(1)
                ).detach()
                metrics["collision_epe"] = (
                    (epe * collision_mask).sum() / collision_count.clamp_min(1)
                ).detach()
                metrics["noncollision_loss"] = (
                    (per_token_loss * noncollision_mask).sum() / noncollision_count.clamp_min(1)
                ).detach()
                metrics["noncollision_epe"] = (
                    (epe * noncollision_mask).sum() / noncollision_count.clamp_min(1)
                ).detach()
            if flow_mask is not None:
                metrics["mean_loss_weight"] = flow_mask.float().mean().detach()

        return loss, metrics

    return init_caches, compute_step


# ---------------------------------------------------------------------------
# Shared training infrastructure
# ---------------------------------------------------------------------------

def _train(
    data,
    model_cls,
    make_train_fns,
    out_dir,
    max_steps,
    checkpoint_freq,
    clip_grad_norm,
    load_checkpoint,
    init_checkpoint,
    ckpt_load_optim,
    ckpt_load_scheduler,
    lr,
    weight_decay,
    warmup_steps,
    scheduler_type,
    wandb_enabled,
    wandb_project,
    tensorboard_enabled,
    tensorboard_dir,
    config_dict,
    configure_model=None,
    train_mode=None,
    unfreeze_last_n_layers=6,
):
    if load_checkpoint is not None and init_checkpoint is not None:
        raise ValueError("--load-checkpoint and --init-checkpoint are mutually exclusive")
    git_commit = current_git_commit()
    config_dict = config_dict | {"git_commit": git_commit, "init_checkpoint": init_checkpoint,
                                 "train_mode": train_mode,
                                 "unfreeze_last_n_layers": unfreeze_last_n_layers}
    # Output & logging setup
    slurm_id = os.environ.get("SLURM_JOB_ID")
    timestamp = datetime.now().strftime("%H-%M-%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_id = slurm_id if slurm_id is not None else timestamp
    out_path = Path(out_dir) / date_str / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_path / "train.log"),
        ],
    )

    # Distributed init & single-GPU fallback
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_type = "cuda"
        device = torch.device(f"{device_type}:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Running distributed. Local rank: {local_rank}, World size: {world_size}")
        rank0logger = logging.getLogger(__name__)
        if rank != 0:
            rank0logger.disabled = True
        barrier = dist.barrier
    else:
        rank = 0
        device_type = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        logger.info(f"Running non-distributed on {device_type}")
        rank0logger = logger
        barrier = lambda: None

    # Save config
    if rank == 0:
        config_path = out_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        rank0logger.info(f"Saved config to {config_path}")

    # WandB setup
    if wandb_enabled and rank == 0:
        import wandb
        wandb.init(
            project=wandb_project,
            config=config_dict | {"global_batch_size": config_dict.get("batch_size", 1) * world_size},
            dir=out_path,
        )

    tensorboard_writer = None
    if tensorboard_enabled and rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        if tensorboard_dir is None:
            tensorboard_path = out_path / "tensorboard"
        else:
            tensorboard_path = Path(tensorboard_dir) / config_dict.get("model", "train") / date_str / run_id
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_path), flush_secs=10)
        tensorboard_writer.add_text("run/config", json.dumps(config_dict, indent=2), global_step=0)
        atexit.register(tensorboard_writer.close)
        rank0logger.info(f"TensorBoard events: {tensorboard_path}")

    # Checkpoint loading pt1: read step counter before seeding
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, weights_only=False, map_location=device)
        start_step = checkpoint["step"]
        if train_mode is not None and checkpoint.get("train_mode") != train_mode:
            raise ValueError("Checkpoint train_mode differs; use --init-checkpoint for a new training stage")
        rank0logger.info(f"Loaded checkpoint from {load_checkpoint} @ step {start_step}.")
    else:
        checkpoint = None
        start_step = 0

    # Seeding
    seed = 42 + rank + start_step
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model_cls().to(device)
    if load_checkpoint is not None:
        model.load_state_dict(_strip_uniform_module_prefix(checkpoint["model"]), strict=True)
    elif init_checkpoint is not None:
        incompatible = load_init_checkpoint(model, init_checkpoint)
        rank0logger.info(f"Initialized model from {init_checkpoint}; allowed missing keys: {incompatible.missing_keys}")

    trainable_parameters = configure_model(model, train_mode, unfreeze_last_n_layers) if configure_model else [
        p for p in model.parameters() if p.requires_grad
    ]
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    rank0logger.info(model)
    rank0logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M "
                     f"({sum(p.numel() for p in trainable_parameters) / 1e6:.3f}M trainable)")
    for name in trainable_names:
        rank0logger.info(f"Trainable: {name}")

    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    scheduler_max_steps = max_steps
    if load_checkpoint is not None and not ckpt_load_scheduler and max_steps is not None:
        scheduler_max_steps = max_steps - start_step
        if scheduler_max_steps < 1:
            raise ValueError(
                f"max_steps ({max_steps}) must exceed resumed step ({start_step})"
            )
        rank0logger.info(
            f"Reset scheduler over {scheduler_max_steps} remaining steps "
            f"({start_step} -> {max_steps})."
        )
    scheduler = make_scheduler(
        optimizer,
        lr=lr,
        warmup_steps=min(warmup_steps, scheduler_max_steps),
        max_steps=scheduler_max_steps,
        scheduler_type=scheduler_type,
    )
    if load_checkpoint is not None:
        if ckpt_load_optim:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if ckpt_load_scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        rank0logger.info("Checkpoint state loaded strictly.")

    real_model = model

    # DDP wrapping (after state load so we wrap the restored model)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore

    # Build model-specific step functions
    init_caches, compute_step = make_train_fns(model, real_model, device, device_type, is_distributed)

    rank0logger.info("Training step uses eager mode (torch.compile disabled).")

    barrier()

    train_loader = data.train_dataloader()

    # Training loop
    # Caches (block_mask, is_query, L_poke) are initialized once from the first batch,
    # assuming consistent batch shapes throughout training.
    caches_initialized = False
    block_mask = is_query = L_poke = None

    if rank == 0:
        logger.info("Starting training...")

    done = False
    for i, batch in enumerate(
        pbar := tqdm(endless_iter(train_loader), desc="Training", disable=rank != 0, initial=start_step)
    ):
        global_step = start_step + i + 1
        try:
            if not caches_initialized:
                block_mask, is_query, L_poke = init_caches(batch)
                caches_initialized = True

            flow_mask = batch.get("flow_loss_mask", None)
            optimizer.zero_grad(set_to_none=True)
            device_batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            step_kwargs = dict(
                block_mask=block_mask,
                is_query=is_query,
                L_poke=L_poke,
                compute_metrics=True,
                flow_mask=flow_mask.to(device) if flow_mask is not None else None,
            )
            loss, metrics = compute_step(device_batch, **step_kwargs)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, clip_grad_norm)
            optimizer.step()
            scheduler.step()

            avg_loss = (
                dist_nn.all_reduce(loss.detach().clone(), op=dist.ReduceOp.SUM) / world_size
                if is_distributed else loss.detach()
            )
            metrics = {
                k: (
                    dist_nn.all_reduce(v.detach(), op=dist.ReduceOp.SUM) / world_size
                    if is_distributed else v.detach()
                ).item()
                for k, v in metrics.items()
            }
            train_meta = {
                "loss": avg_loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
            } | metrics

            if device_type == "cuda":
                train_meta |= {
                    "gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
                    "gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
                }

            pbar.set_postfix(train_meta)
            if tensorboard_writer is not None:
                for key, value in train_meta.items():
                    tensorboard_writer.add_scalar(f"train/{key}", value, global_step)
                tensorboard_writer.add_scalar("system/torch_compile_active", 0, global_step)
            if wandb_enabled and rank == 0:
                import wandb
                wandb.log({f"train/{k}": v for k, v in train_meta.items()}, step=global_step)

            done = max_steps is not None and global_step >= max_steps
            if done:
                rank0logger.info(f"Reached max steps: {global_step} >= {max_steps}. Stopping training...")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping training...")
            done = True

        checkpoint_due = done or (global_step % checkpoint_freq == 0)
        if checkpoint_due:
            save_failed = False
            if rank == 0:
                checkpoint = {
                    "model": real_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "train_mode": train_mode,
                    "git_commit": git_commit,
                }
                ckpt_dir = out_path / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                destination = ckpt_dir / f"checkpoint_{global_step:07}.pt"
                try:
                    atomic_torch_save(checkpoint, destination)
                    rank0logger.info(f"Saved checkpoint at step {global_step}.")
                except (OSError, RuntimeError) as error:
                    save_failed = True
                    rank0logger.error(
                        f"Checkpoint save failed at step {start_step + i}; stopping with the previous "
                        f"checkpoint intact: {error}"
                    )
            if is_distributed:
                failed_tensor = torch.tensor(int(save_failed), device=device)
                dist.broadcast(failed_tensor, src=0)
                save_failed = bool(failed_tensor.item())
            if save_failed:
                done = True

        if done:
            break

    barrier()
    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()
    rank0logger.info("Training stopped.")


# ---------------------------------------------------------------------------
# Common CLI options shared by all subcommands
# ---------------------------------------------------------------------------

def common_options(func):
    options = [
        click.option("--out-dir", default="outputs", show_default=True, help="Root output directory."),
        click.option("--max-steps", default=None, type=int, help="Stop after this many steps (None = run forever)."),
        click.option("--checkpoint-freq", default=25_000, show_default=True, help="Save a checkpoint every N steps."),
        click.option("--clip-grad-norm", default=1.0, show_default=True, type=float, help="Gradient clipping norm."),
        click.option("--load-checkpoint", default=None, type=click.Path(exists=True), help="Checkpoint to resume from."),
        click.option("--init-checkpoint", default=None, type=click.Path(exists=True), help="Load model weights only for a new stage."),
        click.option("--ckpt-load-optim/--no-ckpt-load-optim", default=True, show_default=True, help="Restore optimizer state."),
        click.option("--ckpt-load-scheduler/--no-ckpt-load-scheduler", default=True, show_default=True, help="Restore scheduler state."),
        click.option("--lr", default=1e-4, show_default=True, type=float, help="Peak learning rate."),
        click.option("--weight-decay", default=1e-2, show_default=True, type=float, help="AdamW weight decay."),
        click.option("--warmup-steps", default=1_000, show_default=True, type=int, help="Linear LR warmup steps."),
        click.option("--scheduler", "scheduler_type", default="linear", show_default=True,
                     type=click.Choice(["linear", "cosine"]), help="LR decay schedule after warmup."),
        click.option("--wandb/--no-wandb", "wandb_enabled", default=False, show_default=True, help="Enable W&B logging."),
        click.option("--wandb-project", default="flow-poke-reasoner", show_default=True, help="W&B project name."),
        click.option("--tensorboard/--no-tensorboard", "tensorboard_enabled", default=True, show_default=True,
                     help="Write per-step TensorBoard metrics."),
        click.option("--tensorboard-dir", default=None, type=click.Path(file_okay=False),
                     help="TensorBoard root (default: the run output directory)."),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


# ---------------------------------------------------------------------------
# CLI group & subcommands
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Flow Poke Transformer / Myriad training."""
    pass


@cli.command("fpt")
@common_options
@click.option("--tar-base", default="data", show_default=True, help="Base directory containing .tar shards.")
@click.option("--batch-size", default=32, show_default=True, type=int, help="Training batch size.")
def train_fpt(tar_base, batch_size, **common_kwargs):
    """Train the original FlowPokeTransformer (flow_poke.model)."""
    from flow_poke.model import FlowPokeTransformer_Base
    from flow_poke.data import TrackerShardsDataModule

    data = TrackerShardsDataModule(tar_base=tar_base, batch_size=batch_size, shuffle=1000)

    config_dict = dict(model="fpt", tar_base=tar_base, batch_size=batch_size, **common_kwargs)
    _train(
        data=data,
        model_cls=FlowPokeTransformer_Base,
        make_train_fns=fpt_make_train_fns,
        config_dict=config_dict,
        **common_kwargs,
    )


@cli.command("myriad")
@common_options
@click.option("--tar-base", required=True, multiple=True, help="Base directory/directories containing .tar shards.")
@click.option("--train-shards", default=None, help="Glob pattern for training shards (default: all *.tar).")
@click.option("--batch-size", default=8, show_default=True, type=int, help="Training batch size.")
@click.option("--num-workers", default=4, show_default=True, type=int, help="DataLoader worker count.")
@click.option("--num-tracks", default=16, show_default=True, type=int, help="Number of tracks per sample.")
@click.option("--num-steps", default=16, show_default=True, type=int, help="Number of time steps per sequence.")
@click.option("--shuffle", default=500, show_default=True, type=int, help="WebDataset shuffle buffer (0 = off).")
def train_2d(tar_base, train_shards, batch_size, num_workers, num_tracks, num_steps, shuffle, **common_kwargs):
    """Train on 2-D tracker shards (myriad.data_2d.TrackerShardsDataModule)."""
    from myriad.model import MyriadStepByStep_Large
    from myriad.data_2d import TrackerShardsDataModule

    data = TrackerShardsDataModule(
        tar_base=list(tar_base),
        batch_size=batch_size,
        num_workers=num_workers,
        num_tracks=num_tracks,
        num_steps=num_steps,
        train={"shards": train_shards, "shuffle": shuffle},
    )

    config_dict = dict(
        model="large", dataset="2d",
        tar_base=list(tar_base), batch_size=batch_size,
        num_workers=num_workers, num_tracks=num_tracks, num_steps=num_steps,
        train_shards=train_shards,
        **common_kwargs,
    )
    _train(
        data=data,
        model_cls=MyriadStepByStep_Large,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        **common_kwargs,
    )


@cli.command("billiards")
@common_options
@click.option("--batch-size", default=8, show_default=True, type=int, help="Training batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="DataLoader worker count.")
@click.option("--nr-balls", default=16, show_default=True, type=int, help="Number of balls in the simulation.")
@click.option("--frame-size", default=512, show_default=True, type=int, help="Rendered frame size in pixels.")
@click.option("--duration", default=0.5, show_default=True, type=float, help="Simulation duration in seconds.")
@click.option("--dt", default=0.01, show_default=True, type=float, help="Simulation time step.")
def train_billiards(batch_size, num_workers, nr_balls, frame_size, duration, dt, **common_kwargs):
    """Train on procedurally generated billiards data (myriad.data_billiards.BilliardSimDataModule)."""
    from myriad.model import MyriadStepByStep_Large_Billiard
    from myriad.data_billiards import BilliardSimDataModule

    data = BilliardSimDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train={"dataset_config": dict(nr_balls=nr_balls, frame_size=frame_size, duration=duration, dt=dt)},
    )

    config_dict = dict(
        model="billiard", dataset="billiards",
        batch_size=batch_size, num_workers=num_workers,
        nr_balls=nr_balls, frame_size=frame_size, duration=duration, dt=dt,
        **common_kwargs,
    )
    _train(
        data=data,
        model_cls=MyriadStepByStep_Large_Billiard,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        **common_kwargs,
    )


@cli.command("billiards-physics")
@common_options
@click.option("--train-mode", default="physics-only", show_default=True,
              type=click.Choice(["physics-only", "finetune"]))
@click.option("--unfreeze-last-n-layers", default=6, show_default=True, type=int)
@click.option("--physics-bias-checkpoint-chunks/--no-physics-bias-checkpoint-chunks",
              default=False, show_default=True)
@click.option("--physics-kinematics-mode", default="long", show_default=True,
              type=click.Choice(["long", "short", "collision-gated", "collision-smooth"]))
@click.option("--collision-long-history-distance", default=0.04, show_default=True, type=float)
@click.option("--collision-long-history-temperature", default=0.008, show_default=True, type=float)
@click.option("--batch-size", default=1, show_default=True, type=int)
@click.option("--num-workers", default=4, show_default=True, type=int)
@click.option("--nr-balls", default=16, show_default=True, type=int)
@click.option("--frame-size", default=512, show_default=True, type=int)
@click.option("--duration", default=0.5, show_default=True, type=float)
@click.option("--dt", default=0.01, show_default=True, type=float)
@click.option("--collision-loss-weight", default=3.0, show_default=True, type=float,
              help="Loss multiplier for balls affected by a collision.")
@click.option("--collision-window-steps", default=10, show_default=True, type=int,
              help="Also weight this many flow steps after each collision.")
def train_billiards_physics(train_mode, unfreeze_last_n_layers, physics_bias_checkpoint_chunks,
                            physics_kinematics_mode, collision_long_history_distance,
                            collision_long_history_temperature,
                            batch_size, num_workers, nr_balls, frame_size, duration, dt,
                            collision_loss_weight, collision_window_steps, **common_kwargs):
    """Train the separate MYRIAD billiards model with relation-MLP attention bias."""
    from myriad.model import MyriadStepByStep_Large_Billiard_PhysicsBias
    from myriad.data_billiards import BilliardSimDataModule

    data = BilliardSimDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train={"dataset_config": dict(
            nr_balls=nr_balls,
            frame_size=frame_size,
            duration=duration,
            dt=dt,
            collision_loss_weight=collision_loss_weight,
            collision_window_steps=collision_window_steps,
        )},
    )
    physics_config = {
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
        "physics_bias_checkpoint_chunks": physics_bias_checkpoint_chunks,
        "physics_kinematics_mode": physics_kinematics_mode,
        "collision_long_history_distance": collision_long_history_distance,
        "collision_long_history_temperature": collision_long_history_temperature,
    }

    def model_cls():
        model = MyriadStepByStep_Large_Billiard_PhysicsBias()
        generator = model.transformer.physics_bias_generator
        generator.checkpoint_chunks = physics_bias_checkpoint_chunks
        generator.set_kinematics_mode(
            physics_kinematics_mode,
            collision_distance=collision_long_history_distance,
            collision_temperature=collision_long_history_temperature,
        )
        return model

    config_dict = dict(
        model="billiard-physics", dataset="billiards", batch_size=batch_size,
        num_workers=num_workers, nr_balls=nr_balls, frame_size=frame_size,
        duration=duration, dt=dt, train_mode=train_mode,
        unfreeze_last_n_layers=unfreeze_last_n_layers, **physics_config, **common_kwargs,
        collision_loss_weight=collision_loss_weight,
        collision_window_steps=collision_window_steps,
    )
    _train(
        data=data,
        model_cls=model_cls,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        configure_model=configure_physics_training,
        train_mode=train_mode,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        **common_kwargs,
    )


@cli.command("billiards-long-history")
@common_options
@click.option("--train-mode", default="physics-only", show_default=True,
              type=click.Choice(["physics-only", "finetune"]))
@click.option("--unfreeze-last-n-layers", default=6, show_default=True, type=int)
@click.option("--long-history-checkpoint-chunks/--no-long-history-checkpoint-chunks",
              default=False, show_default=True)
@click.option("--batch-size", default=1, show_default=True, type=int)
@click.option("--num-workers", default=4, show_default=True, type=int)
@click.option("--nr-balls", default=16, show_default=True, type=int)
@click.option("--frame-size", default=512, show_default=True, type=int)
@click.option("--duration", default=0.5, show_default=True, type=float)
@click.option("--dt", default=0.01, show_default=True, type=float)
@click.option("--collision-loss-weight", default=3.0, show_default=True, type=float)
@click.option("--collision-window-steps", default=10, show_default=True, type=int)
def train_billiards_long_history(train_mode, unfreeze_last_n_layers,
                                 long_history_checkpoint_chunks, batch_size, num_workers,
                                 nr_balls, frame_size, duration, dt,
                                 collision_loss_weight, collision_window_steps, **common_kwargs):
    """Train a temporal-history attention bias with no physical-state inputs."""
    from myriad.model import MyriadStepByStep_Large_Billiard_LongHistoryBias
    from myriad.data_billiards import BilliardSimDataModule

    data = BilliardSimDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train={"dataset_config": dict(
            nr_balls=nr_balls,
            frame_size=frame_size,
            duration=duration,
            dt=dt,
            collision_loss_weight=collision_loss_weight,
            collision_window_steps=collision_window_steps,
        )},
    )

    def model_cls():
        model = MyriadStepByStep_Large_Billiard_LongHistoryBias()
        model.transformer.physics_bias_generator.checkpoint_chunks = long_history_checkpoint_chunks
        return model

    config_dict = dict(
        model="billiard-long-history", dataset="billiards", batch_size=batch_size,
        num_workers=num_workers, nr_balls=nr_balls, frame_size=frame_size,
        duration=duration, dt=dt, train_mode=train_mode,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        use_physics_bias=False, use_long_history_bias=True,
        long_history_bias_hidden_dim=64, long_history_bias_depth=2,
        long_history_bias_time_scale=50.0, long_history_bias_max_abs=0.25,
        long_history_bias_num_layers=4, long_history_bias_initial_scale=0.25,
        long_history_bias_window=8,
        long_history_checkpoint_chunks=long_history_checkpoint_chunks,
        collision_loss_weight=collision_loss_weight,
        collision_window_steps=collision_window_steps,
        **common_kwargs,
    )
    _train(
        data=data,
        model_cls=model_cls,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        configure_model=configure_physics_training,
        train_mode=train_mode,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        **common_kwargs,
    )


@cli.command("billiards-combined-gated")
@common_options
@click.option("--train-mode", default="physics-only", show_default=True,
              type=click.Choice(["physics-only", "finetune"]))
@click.option("--unfreeze-last-n-layers", default=6, show_default=True, type=int)
@click.option("--combined-bias-checkpoint-chunks/--no-combined-bias-checkpoint-chunks",
              default=False, show_default=True)
@click.option("--batch-size", default=1, show_default=True, type=int)
@click.option("--num-workers", default=4, show_default=True, type=int)
@click.option("--nr-balls", default=16, show_default=True, type=int)
@click.option("--frame-size", default=512, show_default=True, type=int)
@click.option("--duration", default=0.5, show_default=True, type=float)
@click.option("--dt", default=0.01, show_default=True, type=float)
@click.option("--collision-loss-weight", default=3.0, show_default=True, type=float)
@click.option("--collision-window-steps", default=10, show_default=True, type=int)
def train_billiards_combined_gated(train_mode, unfreeze_last_n_layers,
                                    combined_bias_checkpoint_chunks, batch_size, num_workers,
                                    nr_balls, frame_size, duration, dt,
                                    collision_loss_weight, collision_window_steps, **common_kwargs):
    """Train jointly gated physical and temporal relation-bias branches."""
    from myriad.model import MyriadStepByStep_Large_Billiard_CombinedBias
    from myriad.data_billiards import BilliardSimDataModule

    data = BilliardSimDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train={"dataset_config": dict(
            nr_balls=nr_balls,
            frame_size=frame_size,
            duration=duration,
            dt=dt,
            collision_loss_weight=collision_loss_weight,
            collision_window_steps=collision_window_steps,
        )},
    )

    def model_cls():
        model = MyriadStepByStep_Large_Billiard_CombinedBias()
        model.transformer.physics_bias_generator.checkpoint_chunks = combined_bias_checkpoint_chunks
        model.transformer.long_history_bias_generator.checkpoint_chunks = combined_bias_checkpoint_chunks
        model.transformer.physics_bias_generator.set_kinematics_mode(
            "collision-smooth", collision_distance=0.04, collision_temperature=0.008
        )
        return model

    config_dict = dict(
        model="billiard-combined-gated", dataset="billiards", batch_size=batch_size,
        num_workers=num_workers, nr_balls=nr_balls, frame_size=frame_size,
        duration=duration, dt=dt, train_mode=train_mode,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        use_physics_bias=True, use_long_history_bias=True,
        use_combined_bias_gates=True,
        combined_physics_initial_scale=0.5,
        combined_long_history_initial_scale=0.25,
        physics_kinematics_mode="collision-smooth",
        long_history_checkpoint_chunks=combined_bias_checkpoint_chunks,
        collision_loss_weight=collision_loss_weight,
        collision_window_steps=collision_window_steps,
        **common_kwargs,
    )
    _train(
        data=data,
        model_cls=model_cls,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        configure_model=configure_combined_bias_training,
        train_mode=train_mode,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        **common_kwargs,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    try:
        cli()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
