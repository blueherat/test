#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

EPOCH_INDEX=${EPOCH_INDEX:-4}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-1251}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-$((EPOCH_INDEX * STEPS_PER_EPOCH))}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-10000}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
SAMPLE_BATCH=${SAMPLE_BATCH:-8}
WAIT_FOR_FREE_GPUS=${WAIT_FOR_FREE_GPUS:-0}

ORIG_RUN=${ORIG_RUN:-fair_ditdh_qtr56_dinov2_original_gbs1024_epochwise_4gpu}
ADAPTER_RUN=${ADAPTER_RUN:-fair_ditdh_qtr56_dinov2_adapter_gbs1024_epochwise_4gpu}
ORIG_CONFIG=${ORIG_CONFIG:-$EQVAE_ROOT/experiments/configs/fair_ditdh_qtr56_dinov2_original_gbs1024_ep80.yaml}
ADAPTER_CONFIG=${ADAPTER_CONFIG:-$EQVAE_ROOT/experiments/configs/fair_ditdh_qtr56_dinov2_adapter_gbs1024_ep80.yaml}

export EPOCH_INDEX
export STEPS_PER_EPOCH
export MAX_TRAIN_STEPS
export NUM_FID_SAMPLES
export TRAIN_GPUS
export NPROC
export SAMPLE_BATCH
export WAIT_FOR_FREE_GPUS
export ORIG_RUN
export ADAPTER_RUN
export ORIG_CONFIG
export ADAPTER_CONFIG

bash "$EQVAE_ROOT/experiments/run_fair_epoch_compare.sh"
