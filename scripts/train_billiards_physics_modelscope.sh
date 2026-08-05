#!/usr/bin/env bash
# Train the MYRIAD physics-bias model from the official billiards baseline
# using a locally downloaded ModelScope DINOv3 ViT-L/16 snapshot.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/lhy/flow-poke-transformer-repro}"
VENV_DIR="${VENV_DIR:-/mnt/lhy/fpt}"
DINO_DIR="${MYRIAD_DINO_PATH:-/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master}"
BASELINE_CKPT="${MYRIAD_CKPT:-${PROJECT_DIR}/checkpoints/myriad_billiard.pt}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/physics_bias_stage1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/root/tf-logs/flow-poke}"
COLLISION_LOSS_WEIGHT="${COLLISION_LOSS_WEIGHT:-3.0}"
COLLISION_WINDOW_STEPS="${COLLISION_WINDOW_STEPS:-10}"

[[ -f "${DINO_DIR}/config.json" ]] || { echo "Missing DINO config: ${DINO_DIR}/config.json" >&2; exit 1; }
[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing MYRIAD checkpoint: ${BASELINE_CKPT}" >&2; exit 1; }
[[ -f "${VENV_DIR}/bin/activate" ]] || { echo "Missing venv: ${VENV_DIR}" >&2; exit 1; }

cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"

export MYRIAD_DINO_PATH="${DINO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

exec python -u train.py billiards-physics \
  --init-checkpoint "${BASELINE_CKPT}" \
  --train-mode physics-only \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --nr-balls 16 \
  --frame-size 512 \
  --duration 0.5 \
  --dt 0.01 \
  --collision-loss-weight "${COLLISION_LOSS_WEIGHT}" \
  --collision-window-steps "${COLLISION_WINDOW_STEPS}" \
  --max-steps 50000 \
  --checkpoint-freq 500 \
  --lr 1e-4 \
  --warmup-steps 200 \
  --scheduler cosine \
  --tensorboard \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --physics-bias-checkpoint-chunks \
  --out-dir "${OUT_DIR}" \
  "$@"
