#!/usr/bin/env bash
# Create and run each one-factor-at-a-time sweep sequentially (from the repo root).
set -u
export PATH="$(pwd)/.venv/Scripts:$PATH"   # so the sweep's `python` is the venv one
PROJECT=cs6886w-a1-vgg6
for f in activation optimizer batch_size lr_adam lr_sgd; do
  out=$(python -m wandb sweep --project "$PROJECT" "configs/sweep_$f.yaml" 2>&1)
  echo "$out"
  agent=$(echo "$out" | grep -o 'wandb agent [^ ]*' | tail -1 | sed 's/wandb agent //')
  echo "== sweep $f -> $agent"
  python -m wandb agent "$agent"
done
echo ALL_SWEEPS_DONE
