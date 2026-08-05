#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: uv venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements-demo.txt" >&2
  exit 1
fi

exec .venv/bin/python -m scripts.demo.app \
  --device cuda \
  --compile False \
  --warmup_compiled_paths False \
  --server_name 0.0.0.0 \
  --server_port "${FPT_PORT:-55555}" \
  "$@"
