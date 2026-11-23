#!/usr/bin/env bash
# All arms, 30 epochs, seeds 0-2 (seed 0 for every arm first). Skips runs that already finished.
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for arm in supervised A B AB; do
    [ -e "runs/${arm}_s${seed}/preds.pt" ] && continue
    rm -rf "runs/${arm}_s${seed}"
    uv run python -m discipulus.train "$arm" --epochs 30 --seed "$seed" > /dev/null
  done
done
echo ALL_DONE
