#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
experiment="$data_root/experiments/raev2_lpl_pilot/lpl_official_800_strict_from10"
source_checkpoint="$data_root/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"
step100_checkpoint="$experiment/checkpoints/branch-0000100-global-0100180.pt"
step150_checkpoint="$experiment/checkpoints/branch-0000150-global-0100230.pt"
config="$repo/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
dino_repo="$data_root/models/RAEv2/dinov3_repo"
reference=/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz
diagnostic="$experiment/diagnostics/nonema_step100"
samples="$diagnostic/samples_n5000_seed0"
metrics="$diagnostic/metrics.csv"
audit="$diagnostic/same_noise_audit.json"
log="$diagnostic/run.log"
orchestrator_pid="${1:?usage: $0 ORCHESTRATOR_PID}"
parent_stopped=0

mkdir -p "$diagnostic"
exec >>"$log" 2>&1

timestamp() {
  date --iso-8601=seconds
}

resume_parent() {
  if (( parent_stopped )) && kill -0 "$orchestrator_pid" 2>/dev/null; then
    kill -CONT "$orchestrator_pid"
    printf '%s resumed_orchestrator=%s\n' "$(timestamp)" "$orchestrator_pid"
    parent_stopped=0
  fi
}
trap resume_parent EXIT

printf '%s diagnostic_start orchestrator=%s\n' "$(timestamp)" "$orchestrator_pid"
if ! kill -0 "$orchestrator_pid" 2>/dev/null; then
  echo "orchestrator PID does not exist: $orchestrator_pid" >&2
  exit 1
fi
orchestrator_command=$(tr '\0' ' ' <"/proc/$orchestrator_pid/cmdline")
if [[ "$orchestrator_command" != *"run_raev2_lpl_checkpoint_cycle.py"* ]] ||
   [[ "$orchestrator_command" != *"--target-step 1000"* ]] ||
   [[ "$orchestrator_command" != *"--experiment-name lpl_official_800_strict_from10"* ]]; then
  echo "refusing to stop an unexpected process: $orchestrator_command" >&2
  exit 1
fi
for required in \
  "$python" \
  "$source_checkpoint" \
  "$step100_checkpoint" \
  "$config" \
  "$dino_repo" \
  "$reference"; do
  if [[ ! -e "$required" ]]; then
    echo "required path is missing: $required" >&2
    exit 1
  fi
done
if [[ -e "$metrics" ]]; then
  echo "diagnostic already completed: $metrics"
  exit 0
fi
if pgrep -f '[s]ample_raev2_threeway.py' >/dev/null; then
  echo "another RAEv2 sampler is already active" >&2
  exit 1
fi

# Stop only the sequential scheduler. Its current torchrun child is intentionally
# left running until it has completely finished the step-150 checkpoint.
kill -STOP "$orchestrator_pid"
parent_stopped=1
sleep 2
parent_state=$(ps -o stat= -p "$orchestrator_pid" | tr -d ' ')
if [[ "$parent_state" != *T* ]]; then
  echo "orchestrator did not enter the stopped state: $parent_state" >&2
  exit 1
fi
printf '%s orchestrator_stopped state=%s\n' "$(timestamp)" "$parent_state"

while pgrep -f '[t]rain_raev2_strict_lpl.py.*--max-updates 150' >/dev/null; do
  printf '%s waiting_for_step150\n' "$(timestamp)"
  sleep 30
done

if [[ ! -f "$step150_checkpoint" ]]; then
  echo "step-150 trainer exited without the expected checkpoint" >&2
  exit 1
fi
cd "$repo"
"$python" -c \
  "from pathlib import Path; from experiments.run_raev2_long_pipeline import validate_checkpoint; print(validate_checkpoint(Path('$step150_checkpoint'), objective='lpl'))"

if pgrep -f '[t]rain_raev2_strict_lpl.py' >/dev/null; then
  echo "a training process is still active after step-150 validation" >&2
  exit 1
fi
if pgrep -f '[s]ample_raev2_threeway.py' >/dev/null; then
  echo "another sampler became active before the diagnostic" >&2
  exit 1
fi
printf '%s step150_validated_and_gpus_released\n' "$(timestamp)"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  experiments/sample_raev2_threeway.py \
  --config "$config" \
  --branch "official_model=$source_checkpoint" \
  --branch "lpl_s0100_model=$step100_checkpoint" \
  --results-dir "$samples" \
  --sample-count 5000 \
  --per-rank-batch 16 \
  --sampling-seed 0 \
  --precision bf16 \
  --state-key model \
  --dino-repo-dir "$dino_repo"
printf '%s nonema_sampling_complete\n' "$(timestamp)"

"$python" -c \
  "import json; from pathlib import Path; from experiments.run_raev2_long_pipeline import verify_same_noise_protocol; p=verify_same_noise_protocol({'official_model': Path('$samples/official_model'), 'lpl_s0100_model': Path('$samples/lpl_s0100_model')}); Path('$audit').write_text(json.dumps(p, indent=2, ensure_ascii=False)+'\\n', encoding='utf-8'); print(json.dumps(p, ensure_ascii=False))"

CUDA_VISIBLE_DEVICES=0 "$python" experiments/evaluate_raev2_samples.py \
  --branch "official_model=$samples/official_model/samples.npz" \
  --branch "lpl_s0100_model=$samples/lpl_s0100_model/samples.npz" \
  --reference "$reference" \
  --output "$metrics" \
  --batch-size 64 \
  --seed 0
printf '%s nonema_evaluation_complete metrics=%s\n' "$(timestamp)" "$metrics"

resume_parent
trap - EXIT
printf '%s diagnostic_complete\n' "$(timestamp)"
