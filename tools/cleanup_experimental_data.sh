#!/usr/bin/env bash
set -euo pipefail

ROOT="${EQVAE_DATA_ROOT:-/data/users/zhoushunyu/eqvae}"
MODE="${1:---dry-run}"
STAMP="20260731"
ARCHIVE="$ROOT/retained/cleanup_$STAMP"
EXPERIMENTS="$ROOT/experiments"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
  echo "usage: $0 [--dry-run|--apply]" >&2
  exit 2
fi

if [[ "$(realpath -m "$ROOT")" != "/data/users/zhoushunyu/eqvae" ]]; then
  echo "refusing unexpected data root: $ROOT" >&2
  exit 2
fi

DELETE_REL=(
  "cache/rae_layerwise_path_streams"
  "cache/rae_layerwise_path_smoke"
  "cache/rae_decoder_risk_phase0"
  "cache/gauge_large"
  "stage2_samples"
  "stage2_training"
  "artifacts/rae_stage2_samples"
  "artifacts/rae_stage2_training"
  "artifacts/latent_adapter"
  "artifacts/decoder_inverse_adapter"
  "experiments/rae_strict_lpl"
  "experiments/rae_strict_lpl_cross_tokenizer"
  "experiments/rae_lpl_authenticity"
  "experiments/rae_lpl_detach_factorial"
  "experiments/rae_lpl_detach_cross_tokenizer"
  "experiments/rae_lpl_detach_factorial_impl_replay"
  "experiments/rae_lpl_checkpoint_curve_seed4102"
  "experiments/rae_strict_lpl_compute_control"
  "experiments/rae_dinov2_large"
  "experiments/rae_layerwise_path_train"
  "experiments/rae_layerwise_path_samples"
  "experiments/rae_spectral_tiny"
  "experiments/rae_spc_multiseed_v1"
  "experiments/rae_path_schedule_train"
  "experiments/rae_path_schedule_train_smoke"
  "experiments/rae_path_crossover_train"
  "experiments/rae_path_crossover_train_v2"
  "experiments/rae_path_schedule_closure"
  "experiments/rae_dual_stream_full"
  "experiments/rae_dual_stream_full_smoke"
  "experiments/rae_dual_stream_full_detailshift"
  "experiments/rae_dual_stream_gate"
  "experiments/rae_dual_stream_gate_smoke"
  "experiments/rae_dual_stream_gate_detailshift"
  "experiments/rae_decoder_risk_phase0"
  "experiments/raev2_lpl_pilot"
  "experiments/raev2_guidance_aware_10step"
  "experiments/raev2_flow_parallel_10step"
  "experiments/raev2_raw_10step"
  "experiments/raev2_prediction_detach_10step"
  "experiments/raev2_detach_raw_50step_evaluation"
  "experiments/raev2_target_normalized_10step"
  "experiments/raev2_symmetric_10step"
)

# These compact experiment directories remain, but their generated images/arrays do not.
STRIP_REL=(
  "experiments/rae_strict_lpl_official_source_5k"
  "experiments/raev2_common_adapter_model_10step"
  "experiments/raev2_common_adapter_full_target_control"
  "experiments/raev2_head_swap_full10"
)

# Preserve a minimal set of exact checkpoints needed to reproduce the decisive claims.
KEEP_MOVES=(
  "experiments/rae_strict_lpl/seed3_flow_from_official_s0_u2000/checkpoints/step-0002000.pt|old_rae_dinov2_b/flow_seed3_step2000.pt"
  "experiments/rae_strict_lpl/seed3_lpl_from_official_s0_u2000/checkpoints/step-0002000.pt|old_rae_dinov2_b/lpl_seed3_step2000.pt"
  "experiments/rae_strict_lpl_cross_tokenizer/mae_seed3407_flow_u500/checkpoints/step-0000500.pt|old_rae_mae_b/flow_seed3407_step500.pt"
  "experiments/rae_strict_lpl_cross_tokenizer/mae_seed3407_lpl_u500/checkpoints/step-0000500.pt|old_rae_mae_b/lpl_seed3407_step500.pt"
  "experiments/rae_strict_lpl_cross_tokenizer/siglip2_seed3407_flow_u500/checkpoints/step-0000500.pt|old_rae_siglip2_b/flow_seed3407_step500.pt"
  "experiments/rae_strict_lpl_cross_tokenizer/siglip2_seed3407_lpl_u500/checkpoints/step-0000500.pt|old_rae_siglip2_b/lpl_seed3407_step500.pt"
  "experiments/rae_lpl_detach_factorial/seed3_detach_from_official_s0_u2000/checkpoints/step-0002000.pt|old_rae_dinov2_b/detach_seed3_step2000.pt"
  "experiments/rae_lpl_detach_factorial/seed0_raw_from_official_s0_u2000/checkpoints/step-0000050.pt|old_rae_dinov2_b/raw_seed0_step50.pt"
  "experiments/raev2_lpl_pilot/flow_official_150_strict_from30/checkpoints/branch-0000100-global-0100180.pt|raev2_dinov3_l/flow_branch100.pt"
  "experiments/raev2_lpl_pilot/lpl_official_800_strict_from10/checkpoints/branch-0000100-global-0100180.pt|raev2_dinov3_l/lpl_branch100.pt"
  "experiments/raev2_prediction_detach_10step/lpl10_full_base_prediction_detach_final/checkpoints/branch-0000050-global-0100130.pt|raev2_dinov3_l/detach_branch50.pt"
  "experiments/raev2_raw_10step/raw10_gradcal_full_base/checkpoints/branch-0000050-global-0100130.pt|raev2_dinov3_l/raw_branch50.pt"
  "experiments/rae_spectral_tiny/seed3407_baseline_from_s5000/checkpoints/step-0010000.pt|negative_baselines/spectral_baseline_seed3407_step10000.pt"
  "experiments/rae_spectral_tiny/seed3407_partial_from_s5000/checkpoints/step-0010000.pt|negative_baselines/spectral_partial_seed3407_step10000.pt"
  "experiments/rae_spc_multiseed_v1/seed1201_static_s0_to5000/checkpoints/step-0005000.pt|negative_baselines/spc_static_seed1201_step5000.pt"
  "experiments/rae_spc_multiseed_v1/seed1201_spc_floor020_p2_rank16_switch2000_s0_to5000/checkpoints/step-0005000.pt|negative_baselines/spc_seed1201_step5000.pt"
  "experiments/rae_layerwise_path_train/seed3407_static_rank16_s0_to_10000/checkpoints/step-0010000.pt|negative_baselines/layerwise_static_seed3407_step10000.pt"
  "experiments/rae_layerwise_path_train/seed3407_annealed_rank16_s0_to_10000/checkpoints/step-0010000.pt|negative_baselines/layerwise_annealed_seed3407_step10000.pt"
)

validate_targets() {
  local protected=(
    "/data/shared"
    "$ROOT/datasets"
    "$ROOT/models"
    "$ROOT/external"
    "$ROOT/cache/huggingface"
  )
  local rel path keep
  for rel in "${DELETE_REL[@]}" "${STRIP_REL[@]}"; do
    path="$(realpath -m "$ROOT/$rel")"
    if [[ "$path" != "$ROOT"/* ]]; then
      echo "refusing path outside experiment root: $path" >&2
      exit 4
    fi
    for keep in "${protected[@]}"; do
      if [[ "$path" == "$keep" || "$path" == "$keep"/* ]]; then
        echo "refusing protected path: $path" >&2
        exit 4
      fi
    done
  done
}

validate_keep_sources() {
  local item src_rel dst_rel src dst
  for item in "${KEEP_MOVES[@]}"; do
    src_rel="${item%%|*}"
    dst_rel="${item#*|}"
    src="$ROOT/$src_rel"
    dst="$ARCHIVE/checkpoints/$dst_rel"
    if [[ ! -e "$src" && ! -e "$dst" ]]; then
      echo "missing checkpoint selected for retention: $src" >&2
      exit 5
    fi
  done
}

size_bytes() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sb "$path" 2>/dev/null | cut -f1
  else
    echo 0
  fi
}

archive_metadata() {
  local src="$1"
  local rel="$2"
  [[ -d "$src" ]] || return 0
  while IFS= read -r -d '' file; do
    local sub="${file#"$src"/}"
    local name="${file##*/}"
    case "$file" in
      *.json|*.jsonl|*.yaml|*.yml|*.csv|*.tsv|*.txt|*.md|*.log|*.pdf|*.svg|*.html) ;;
      *.png)
        case "$name" in
          *grid*.png|*plot*.png|*curve*.png|*heatmap*.png|*summary*.png|*comparison*.png|*atlas*.png) ;;
          *) continue ;;
        esac
        ;;
      *) continue ;;
    esac
    case "$sub" in
      */generation/*|generation/*|*/samples*/*|samples*/*)
        case "$file" in
          *.json|*.jsonl|*.yaml|*.yml|*.csv|*.tsv|*.txt|*.md|*.log) ;;
          *) continue ;;
        esac
        ;;
    esac
    local out="$ARCHIVE/metadata/$rel/$sub"
    mkdir -p "$(dirname "$out")"
    cp -p -- "$file" "$out"
  done < <(
    find "$src" -type f -size -200M \
      \( -name '*.json' -o -name '*.jsonl' -o -name '*.yaml' -o -name '*.yml' \
         -o -name '*.csv' -o -name '*.tsv' -o -name '*.txt' -o -name '*.md' \
         -o -name '*.log' -o -name '*.pdf' -o -name '*.svg' -o -name '*.html' \
         -o -name '*.png' \) -print0
  )
}

total=0
echo "mode=$MODE"
echo "root=$ROOT"
echo "archive=$ARCHIVE"
validate_targets
validate_keep_sources

for rel in "${DELETE_REL[@]}"; do
  path="$ROOT/$rel"
  bytes="$(size_bytes "$path")"
  total=$((total + bytes))
  printf 'delete\t%12d\t%s\n' "$bytes" "$rel"
done

for rel in "${STRIP_REL[@]}"; do
  path="$ROOT/$rel"
  bytes=0
  if [[ -d "$path" ]]; then
    while IFS= read -r -d '' file; do
      bytes=$((bytes + $(stat -c '%s' "$file")))
    done < <(find "$path" -type f \( -name '*.png' -o -name '*.npz' \) -print0)
  fi
  total=$((total + bytes))
  printf 'strip \t%12d\t%s\n' "$bytes" "$rel"
done

printf 'candidate_total_bytes=%d\n' "$total"
printf 'candidate_total_gib=%.2f\n' "$(awk -v n="$total" 'BEGIN { print n / 1024 / 1024 / 1024 }')"

if [[ "$MODE" == "--dry-run" ]]; then
  exit 0
fi

mkdir -p "$ARCHIVE/metadata" "$ARCHIVE/checkpoints"
manifest_candidate="$total"
if [[ -f "$ARCHIVE/MANIFEST.txt" ]]; then
  saved_candidate="$(awk -F= '/^candidate_total_bytes=/{print $2; exit}' "$ARCHIVE/MANIFEST.txt")"
  if [[ "$saved_candidate" =~ ^[0-9]+$ && "$saved_candidate" -gt "$manifest_candidate" ]]; then
    manifest_candidate="$saved_candidate"
  fi
fi
{
  printf 'cleanup_date=%s\n' "$STAMP"
  printf 'root=%s\n' "$ROOT"
  printf 'candidate_total_bytes=%d\n' "$manifest_candidate"
  printf 'policy=preserve metadata, decisive checkpoints, and representative negative baselines; remove generated samples and superseded runs\n'
} > "$ARCHIVE/MANIFEST.txt"

for item in "${KEEP_MOVES[@]}"; do
  src_rel="${item%%|*}"
  dst_rel="${item#*|}"
  src="$ROOT/$src_rel"
  dst="$ARCHIVE/checkpoints/$dst_rel"
  if [[ -e "$src" ]]; then
    if [[ -e "$dst" ]]; then
      echo "refusing to overwrite retained checkpoint: $dst" >&2
      exit 3
    fi
    mkdir -p "$(dirname "$dst")"
    mv -- "$src" "$dst"
  elif [[ ! -e "$dst" ]]; then
    echo "missing required checkpoint: $src" >&2
    exit 3
  fi
  printf 'retained_checkpoint=%s\n' "$dst_rel" >> "$ARCHIVE/MANIFEST.txt"
done

for rel in "${DELETE_REL[@]}"; do
  path="$ROOT/$rel"
  [[ -e "$path" ]] || continue
  archive_metadata "$path" "$rel"
  rm -rf -- "$path"
done

for rel in "${STRIP_REL[@]}"; do
  path="$ROOT/$rel"
  [[ -d "$path" ]] || continue
  archive_metadata "$path" "$rel"
  find "$path" -type f \( -name '*.png' -o -name '*.npz' \) -delete
  find "$path" -depth -type d -empty -delete
done

echo "cleanup complete"
