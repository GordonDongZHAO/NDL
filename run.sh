#!/usr/bin/env bash
# # Run the main NDL configurations
# Usage examples:
#   bash run.sh movielens 0
#   bash run.sh yelp 0
#   bash run.sh book 0

set -euo pipefail

DATASET=${1:-movielens}
GPU=${2:-0}
EPOCHS=${EPOCHS:-500}

COMMON_ARGS=(
  --gpu_id "${GPU}"
  --epoch "${EPOCHS}"
  --context_hops 2
  --dim 32
  --cl_lambda 0.001
  --cl_temp 0.2
  --cl_drop 0.1
  --warmup_epochs 10
)

case "${DATASET}" in
  movielens)
    DATA_ARGS=(
      --dataset movielens
      --tau_coh 0.3
      --tau_loss 5.0
      --ema_decay 0.9
      --min_activity 70.0
    )
    ;;
  yelp)
    DATA_ARGS=(
      --dataset yelp
      --tau_coh 0.5
      --tau_loss 10.0
      --ema_decay 0.9
      --min_activity 30.0
    )
    ;;
  book)
    DATA_ARGS=(
      --dataset book
      --tau_coh 0.5
      --tau_loss 5.0
      --ema_decay 0.5
      --min_activity 25.0
    )
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Choose from: movielens, yelp, book." >&2
    exit 1
    ;;
esac

python main_ndl.py "${COMMON_ARGS[@]}" "${DATA_ARGS[@]}"
