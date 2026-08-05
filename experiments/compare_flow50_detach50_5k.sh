#!/usr/bin/env bash
set -euo pipefail

# Flow50 vs prediction-detach LPL50: matched 5k IG-scale comparison.
#
# Default:
#   - branches: flow50, detach50
#   - 5,000 samples per branch/scale
#   - scales: 1.0, 1.4, 1.6, 1.78, 2.1
#   - seed: 20260805
#   - online "model" weights
#
# Usage:
#   bash experiments/compare_flow50_detach50_5k.sh
#   bash experiments/compare_flow50_detach50_5k.sh audit
#   bash experiments/compare_flow50_detach50_5k.sh summarize
#
# Optional overrides:
#   SEEDS="20260805 20260806" SAMPLE_COUNT=5000 \
#   SCALES="1.0,1.4,1.6,1.78,2.1" \
#   INCLUDE_OFFICIAL=1 \
#   bash experiments/compare_flow50_detach50_5k.sh

ACTION="${1:-all}"
case "$ACTION" in
  all|audit|run|summarize) ;;
  *) echo "Usage: $0 [all|audit|run|summarize]" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/zhoushunyu/eqvae}"
DATA="${DATA:-/data/users/zhoushunyu/eqvae}"
PYTHON="${PYTHON:-$DATA/envs/raev2/bin/python}"

CONFIG="${CONFIG:-$REPO/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml}"
DINO_REPO="${DINO_REPO:-$DATA/models/RAEv2/dinov3_repo}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-$REPO/experiments/run_raev2_ig_scale_sweep.py}"

OFFICIAL="${OFFICIAL:-$DATA/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt}"
FLOW50="${FLOW50:-$DATA/experiments/raev2_flow50_matched_detach_protocol/flow50_matched_detach_protocol/checkpoints/branch-0000050-global-0100130.pt}"
DETACH50="${DETACH50:-$DATA/retained/cleanup_20260731/checkpoints/raev2_dinov3_l/detach_branch50.pt}"

FLOW_AUDIT="${FLOW_AUDIT:-$DATA/experiments/raev2_flow50_matched_detach_protocol/flow50_matched_detach_protocol/first_batch_audit.json}"
DETACH_AUDIT="${DETACH_AUDIT:-$DATA/retained/cleanup_20260731/metadata/experiments/raev2_prediction_detach_10step/lpl10_full_base_prediction_detach_final/first_batch_audit.json}"

OUTPUT_BASE="${OUTPUT_BASE:-$DATA/experiments/raev2_ig_scale_sweep/flow50_vs_detach50_5k}"
SAMPLE_COUNT="${SAMPLE_COUNT:-5000}"
SCALES="${SCALES:-1.0,1.4,1.6,1.78,1.82,1.9,2.1}"
SEEDS="${SEEDS:-20260805}"
PER_RANK_BATCH="${PER_RANK_BATCH:-8}"
DEVICES="${DEVICES:-0,1,2,3}"
INCLUDE_OFFICIAL="${INCLUDE_OFFICIAL:-0}"
WAIT_TIMEOUT_HOURS="${WAIT_TIMEOUT_HOURS:-12}"

mkdir -p "$OUTPUT_BASE"
cd "$REPO"

require_file() {
  [[ -f "$1" ]] || { echo "Missing $2: $1" >&2; exit 1; }
}

wait_for_flow50() {
  if [[ -f "$FLOW50" ]]; then return; fi
  echo "Waiting for Flow50: $FLOW50"
  local deadline=$(( $(date +%s) + WAIT_TIMEOUT_HOURS * 3600 ))
  while [[ ! -f "$FLOW50" ]]; do
    (( $(date +%s) < deadline )) || {
      echo "Timed out waiting for Flow50." >&2
      exit 1
    }
    sleep 60
  done
}

audit_pair() {
  "$PYTHON" - "$FLOW50" "$DETACH50" "$OUTPUT_BASE/checkpoint_fairness_audit.json" <<'PY'
import json, sys
from pathlib import Path
import torch

flow_path, detach_path, out_path = map(Path, sys.argv[1:4])

def read(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = ckpt["raev2_lpl"]
    row = {
        "path": str(path.resolve()),
        "step": int(ckpt["step"]),
        "epoch": int(ckpt["epoch"]),
        "objective": meta.get("objective"),
        "lpl_variant": meta.get("lpl_variant"),
        "branch_update": int(meta["branch_update"]),
        "source_step": int(meta["source_step"]),
        "source_epoch": int(meta["source_epoch"]),
        "source_steps_per_epoch": int(meta["source_steps_per_epoch"]),
        "source_sha256": meta["source_sha256"],
        "config_sha256": meta["config_sha256"],
        "data_indices_sha256": meta["data_indices_sha256"],
        "paired_branch_stream_is_exact": bool(meta["paired_branch_stream_is_exact"]),
    }
    del ckpt
    return row

flow, detach = read(flow_path), read(detach_path)

if flow["objective"] != "flow":
    raise SystemExit(f"Expected Flow objective, got {flow['objective']!r}")
if detach["objective"] != "lpl" or detach["lpl_variant"] != "prediction_detach":
    raise SystemExit(
        f"Expected prediction-detach LPL checkpoint, got "
        f"objective={detach['objective']!r}, variant={detach['lpl_variant']!r}"
    )

keys = [
    "step", "branch_update", "source_step", "source_epoch",
    "source_steps_per_epoch", "source_sha256", "config_sha256",
    "data_indices_sha256", "paired_branch_stream_is_exact",
]
matches = {key: flow[key] == detach[key] for key in keys}
payload = {
    "flow50": flow,
    "detach50": detach,
    "required_matches": matches,
    "fairness_passed": all(matches.values()),
}
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nCheckpoint fairness audit")
for key, same in matches.items():
    print(f"{key:31s}: {'MATCH' if same else 'DIFFER'}")
print(f"flow objective              : {flow['objective']}")
print(f"detach objective/variant    : {detach['objective']} / {detach['lpl_variant']}")

if not all(matches.values()):
    raise SystemExit(f"Fairness audit failed; see {out_path}")
print("CHECKPOINT FAIRNESS: PASS")
PY

  if [[ -f "$FLOW_AUDIT" && -f "$DETACH_AUDIT" ]]; then
    "$PYTHON" - "$FLOW_AUDIT" "$DETACH_AUDIT" "$OUTPUT_BASE/first_batch_pairing_audit.json" <<'PY'
import json, sys
from pathlib import Path

flow_path, detach_path, out_path = map(Path, sys.argv[1:4])
flow = json.loads(flow_path.read_text(encoding="utf-8"))
detach = json.loads(detach_path.read_text(encoding="utf-8"))
keys = [
    "rank", "indices", "image_sha256", "label_sha256",
    "latent_sha256", "noise_sha256", "time_sha256", "cfg_mask_sha256",
]
rows, ok = [], True
for left, right in zip(flow, detach, strict=True):
    matches = {k: left.get(k) == right.get(k) for k in keys}
    rows.append({"rank": left.get("rank"), "matches": matches})
    ok &= all(matches.values())
out_path.write_text(
    json.dumps({"exact_match": ok, "ranks": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("\nFirst resumed-batch audit")
for row in rows:
    print(f"rank {row['rank']}: " + (
        "EXACT MATCH" if all(row["matches"].values()) else "MISMATCH"
    ))
if not ok:
    raise SystemExit(f"First-batch pairing failed; see {out_path}")
print("FIRST RESUMED BATCH: EXACT MATCH")
PY
  else
    echo "Warning: first_batch_audit.json missing; checkpoint metadata audit still passed."
  fi
}

run_seed() {
  local seed="$1"
  local out="$OUTPUT_BASE/n${SAMPLE_COUNT}_seed${seed}"
  mkdir -p "$out"

  local branch_args=(
    --branch "flow50=$FLOW50"
    --branch "detach50=$DETACH50"
  )
  if [[ "$INCLUDE_OFFICIAL" == "1" ]]; then
    branch_args=(
      --branch "official=$OFFICIAL"
      "${branch_args[@]}"
    )
  fi

  echo
  echo "Running seed=$seed, n=$SAMPLE_COUNT, scales=$SCALES"
  "$PYTHON" "$SWEEP_SCRIPT" \
    --config "$CONFIG" \
    "${branch_args[@]}" \
    --baseline-branch flow50 \
    --output-root "$out" \
    --scales "$SCALES" \
    --sample-count "$SAMPLE_COUNT" \
    --per-rank-batch "$PER_RANK_BATCH" \
    --sampling-seed "$seed" \
    --metric-seed "$seed" \
    --precision bf16 \
    --state-key model \
    --devices "$DEVICES" \
    --dino-repo-dir "$DINO_REPO" \
    2>&1 | tee "$out/run.log"
}

summarize() {
  "$PYTHON" - "$OUTPUT_BASE" "$SAMPLE_COUNT" $SEEDS <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
n = int(sys.argv[2])
seeds = [int(x) for x in sys.argv[3:]]

frames = []
for seed in seeds:
    path = root / f"n{n}_seed{seed}" / "ig_scale_sweep_summary.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    else:
        print(f"Missing: {path}")

if not frames:
    raise SystemExit("No completed sweep summaries found.")

df = pd.concat(frames, ignore_index=True)
df = df.sort_values(["seed", "ig_scale", "branch"])
df.to_csv(root / "combined_metrics.csv", index=False)

pair = df[df["branch"].isin(["flow50", "detach50"])].copy()
fid = pair.pivot(
    index=["seed", "ig_scale"],
    columns="branch",
    values="frechet_inception_distance",
).reset_index()
fid["detach_minus_flow_fid"] = fid["detach50"] - fid["flow50"]
fid["winner_same_scale"] = fid["detach_minus_flow_fid"].map(
    lambda x: "detach50" if x < 0 else ("flow50" if x > 0 else "tie")
)
fid.to_csv(root / "paired_same_scale_fid.csv", index=False)

best = pair.loc[
    pair.groupby(["seed", "branch"])["frechet_inception_distance"].idxmin()
].sort_values(["seed", "branch"])
best.to_csv(root / "best_scale_by_seed.csv", index=False)

tuned_rows = []
for seed in sorted(pair["seed"].unique()):
    f = best[(best.seed == seed) & (best.branch == "flow50")].iloc[0]
    d = best[(best.seed == seed) & (best.branch == "detach50")].iloc[0]
    delta = d.frechet_inception_distance - f.frechet_inception_distance
    tuned_rows.append({
        "seed": int(seed),
        "flow50_best_scale": float(f.ig_scale),
        "flow50_best_fid": float(f.frechet_inception_distance),
        "detach50_best_scale": float(d.ig_scale),
        "detach50_best_fid": float(d.frechet_inception_distance),
        "detach_minus_flow_tuned_fid": float(delta),
        "winner_tuned": "detach50" if delta < 0 else ("flow50" if delta > 0 else "tie"),
    })
tuned = pd.DataFrame(tuned_rows)
tuned.to_csv(root / "tuned_best_comparison_by_seed.csv", index=False)

agg = pair.groupby(["branch", "ig_scale"]).agg(
    fid_mean=("frechet_inception_distance", "mean"),
    fid_std=("frechet_inception_distance", "std"),
    kid_mean=("kernel_inception_distance_mean", "mean"),
    is_mean=("inception_score_mean", "mean"),
    seeds=("seed", "nunique"),
).reset_index()
agg.to_csv(root / "aggregate_by_scale.csv", index=False)

paired_agg = fid.groupby("ig_scale").agg(
    flow50_fid_mean=("flow50", "mean"),
    flow50_fid_std=("flow50", "std"),
    detach50_fid_mean=("detach50", "mean"),
    detach50_fid_std=("detach50", "std"),
    detach_minus_flow_fid_mean=("detach_minus_flow_fid", "mean"),
    detach_minus_flow_fid_std=("detach_minus_flow_fid", "std"),
    seeds=("seed", "nunique"),
).reset_index()
paired_agg.to_csv(root / "paired_aggregate_by_scale.csv", index=False)

agg_best = agg.loc[agg.groupby("branch")["fid_mean"].idxmin()].sort_values("branch")
agg_best.to_csv(root / "aggregate_best_scale.csv", index=False)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for branch in ["flow50", "detach50"]:
        p = agg[agg.branch == branch].sort_values("ig_scale")
        ax.plot(p.ig_scale, p.fid_mean, marker="o", label=branch)
    ax.set_xlabel("Internal Guidance scale")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title(f"Flow50 vs Detach50, n={n}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "fid_scale_curves.png", dpi=180)
    plt.close(fig)
except Exception as exc:
    print(f"Plot skipped: {exc}")

summary = [
    "# Flow50 vs Detach50",
    "",
    f"- samples per branch/scale: `{n}`",
    f"- completed seeds: `{sorted(pair.seed.unique().tolist())}`",
    "- `detach_minus_flow < 0` means Detach50 is better.",
    "",
    "## Same-scale FID",
    "",
    fid.to_markdown(index=False, floatfmt=".6f"),
    "",
    "## Independently tuned best",
    "",
    tuned.to_markdown(index=False, floatfmt=".6f"),
    "",
    "## Aggregate by scale",
    "",
    paired_agg.to_markdown(index=False, floatfmt=".6f"),
]
(root / "comparison_summary.md").write_text("\n".join(summary), encoding="utf-8")

print("\nSame-scale paired FID")
print(fid.to_string(index=False))
print("\nIndependently tuned best")
print(tuned.to_string(index=False))
print("\nAggregate best scale")
print(agg_best.to_string(index=False))
print(f"\nSummary: {root / 'comparison_summary.md'}")
PY
}

if [[ "$ACTION" != "summarize" ]]; then
  require_file "$PYTHON" "Python"
  require_file "$CONFIG" "config"
  require_file "$SWEEP_SCRIPT" "sweep script"
  require_file "$DETACH50" "Detach50 checkpoint"
  [[ -d "$DINO_REPO" ]] || { echo "Missing DINO repo: $DINO_REPO" >&2; exit 1; }
  if [[ "$INCLUDE_OFFICIAL" == "1" ]]; then
    require_file "$OFFICIAL" "official checkpoint"
  fi
  wait_for_flow50
  audit_pair
fi

[[ "$ACTION" == "audit" ]] && exit 0

if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
  for seed in $SEEDS; do
    run_seed "$seed"
  done
fi

summarize