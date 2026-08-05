#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zhoushunyu/eqvae
DATA=/data/users/zhoushunyu/eqvae
PYTHON="$DATA/envs/raev2/bin/python"

CONFIG="$REPO/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
SOURCE="$DATA/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"
INDEX_MAP="$DATA/datasets/raev2_imagenet_train_lexicographic_indices.npy"
DINO_REPO="$DATA/models/RAEv2/dinov3_repo"

RESULTS_ROOT="$DATA/experiments/raev2_flow50_matched_detach_protocol"
EXPERIMENT="flow50_matched_detach_protocol"
EXPERIMENT_DIR="$RESULTS_ROOT/$EXPERIMENT"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"

FLOW10="$CHECKPOINT_DIR/branch-0000010-global-0100090.pt"
FLOW50="$CHECKPOINT_DIR/branch-0000050-global-0100130.pt"

for path in \
    "$PYTHON" \
    "$CONFIG" \
    "$SOURCE" \
    "$INDEX_MAP"
do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
done

if [[ ! -d "$DINO_REPO" ]]; then
    echo "Missing DINOv3 repository: $DINO_REPO" >&2
    exit 1
fi

mkdir -p "$RESULTS_ROOT"
cd "$REPO"

common_args=(
    experiments/train_raev2_strict_lpl.py
    --config "$CONFIG"
    --data-path /data/shared/imagenet-1k/data
    --packed-data-path /data/shared/imagenet-1k/random_access_v1
    --index-map "$INDEX_MAP"
    --results-dir "$RESULTS_ROOT"
    --experiment-name "$EXPERIMENT"
    --source-checkpoint "$SOURCE"
    --objective flow
    --save-every 10
    --precision bf16
    --ema-device cpu
    --global-seed 42
    --num-workers 4
    --min-free-gib 0.5
    --dino-repo-dir "$DINO_REPO"
)

echo "============================================================"
echo "Matched Flow50 continuation"
echo "Results: $EXPERIMENT_DIR"
echo "============================================================"

# Stage A: official -> Flow10
if [[ ! -f "$FLOW10" ]]; then
    echo "[$(date --iso-8601=seconds)] Training official -> Flow10"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        "${common_args[@]}" \
        --max-updates 10 \
        2>&1 | tee "$RESULTS_ROOT/flow_0_to_10.log"
else
    echo "Flow10 already exists: $FLOW10"
fi

if [[ ! -f "$FLOW10" ]]; then
    echo "Flow10 was not produced: $FLOW10" >&2
    exit 1
fi

# Stage B: resume the latest available Flow checkpoint -> Flow50.
# This also supports safe restart after interruption at update 20/30/40.
if [[ ! -f "$FLOW50" ]]; then
    shopt -s nullglob
    checkpoints=("$CHECKPOINT_DIR"/branch-*-global-*.pt)
    shopt -u nullglob

    if ((${#checkpoints[@]} == 0)); then
        echo "No Flow checkpoint available for resume" >&2
        exit 1
    fi

    IFS=$'\n' checkpoints=($(printf '%s\n' "${checkpoints[@]}" | sort))
    unset IFS
    RESUME="${checkpoints[-1]}"

    echo "[$(date --iso-8601=seconds)] Resuming from:"
    echo "  $RESUME"
    echo "Training to Flow50"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        "${common_args[@]}" \
        --resume "$RESUME" \
        --max-updates 50 \
         --resume "$RESUME" \
        --max-updates 50 \
        2>&1 | tee -a "$RESULTS_ROOT/flow_10_to_50.log"
else
    echo "Flow50 already exists: $FLOW50"
fi

if [[ ! -f "$FLOW50" ]]; then
    echo "Flow50 was not produced: $FLOW50" >&2
    exit 1
fi

echo
echo "============================================================"
echo "Flow50 completed"
echo "$FLOW50"
ls -lh "$FLOW50"
echo "============================================================"
