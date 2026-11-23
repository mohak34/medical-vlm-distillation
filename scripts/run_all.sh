#!/usr/bin/env bash
# Every arm for seeds 0-2 (seed 0 for every arm first), same epoch budget for all. Skips finished runs.
# Main budget is 5 epochs (runs/<arm>_s<seed>); EPOCHS=30 gives the long-schedule ablation (runs/<arm>_e30_s<seed>).
# Usage: scripts/run_all.sh  |  EPOCHS=30 SEEDS=0 scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-5}
TAG=$([ "$EPOCHS" = 5 ] && echo "" || echo "_e$EPOCHS")
for seed in ${SEEDS:-0 1 2}; do
  for arm in supervised A B AB; do
    name="${arm}${TAG}_s${seed}"
    [ -e "runs/$name/preds.pt" ] && continue
    rm -rf "runs/$name"
    uv run python -m discipulus.train "$arm" --epochs "$EPOCHS" --seed "$seed" --name "$name" > /dev/null
  done
done
echo ALL_DONE
