"""Detached resource holder for one train/evaluate action, with visible receipts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import common as c
from experiments.weak_reference_loss_20260914 import idle


def hold(args):
    action = c.ROOT / args.run / args.action_id
    action.mkdir(parents=True, exist_ok=True)
    leases, devices = [], []
    for row in idle.gpu_snapshot():
        if not idle.eligible(row):
            continue
        lease = Path("/tmp", f"eqvae_idle_{row['uuid']}.lock").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        latest = {value["uuid"]: value for value in idle.gpu_snapshot()}
        if row["uuid"] not in latest or not idle.eligible(latest[row["uuid"]]):
            lease.close()
            continue
        leases.append(lease)
        devices.append(row)
    if len(devices) < args.minimum_gpus:
        raise RuntimeError(f"Need {args.minimum_gpus} idle GPUs, found {len(devices)}")
    if (args.stop_file or c.ROOT / "STOP_AFTER_CURRENT").exists():
        raise RuntimeError("The NEW endpoint-training queue has a stop request")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(row["uuid"] for row in devices),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONUNBUFFERED="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    command = [c.PYTHON, "-u", "-m", "torch.distributed.run", "--standalone", "--virtual-local-rank",
               f"--nproc-per-node={len(devices)}", "-m", "--",
               "experiments.adversarial_weak_training_20260915." + args.mode,
               "--run", args.run, *args.arguments]
    if args.stop_file is not None:
        command += ["--stop-file", str(args.stop_file)]
    receipt = dict(mode=args.mode, run=args.run, action_id=args.action_id,
        started_utc=c.now(), holder_pid=os.getpid(), devices=devices, command=command,
        log=str(action / "worker.log"), sources=c.source_receipts())
    c.atomic(action / "launch.json", receipt)
    with (action / "worker.log").open("a") as stream:
        process = subprocess.Popen(command, cwd=c.WORK, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt["pid"] = process.pid
        c.atomic(action / "launch.json", receipt)
        while process.poll() is None:
            c.atomic(action / "status.json", dict(phase="running", updated_utc=c.now(),
                pid=process.pid, holder_pid=os.getpid(), gpu_indices=[row["index"] for row in devices]))
            time.sleep(5)
    c.atomic(action / "status.json", dict(phase="complete" if process.returncode == 0 else "failed",
        exit_code=process.returncode, updated_utc=c.now(), pid=process.pid, holder_pid=os.getpid(),
        gpu_indices=[]))
    for lease in leases:
        lease.close()
    if process.returncode:
        raise SystemExit(process.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=("train", "train_moments", "train_energy", "train_binary", "evaluate"), default="train")
    parser.add_argument("--action-id", default="training")
    parser.add_argument("--minimum-gpus", type=int, default=2)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--stop-file", type=Path,
                        help="Separate stop marker, permitted only for evaluation")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stop_file is not None and args.mode != "evaluate":
        parser.error("--stop-file is only supported for evaluation")
    if args.arguments[:1] == ["--"]:
        args.arguments = args.arguments[1:]
    if args.hold:
        hold(args)
        return
    run = c.ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "train_moments", "train_energy", "train_binary"):
        path = run / "request.json"
        if path.exists():
            raise RuntimeError("A training request already exists; use a new run ID")
        initial_name = (args.arguments[args.arguments.index("--initial-head") + 1]
                        if "--initial-head" in args.arguments else
                        'guided_weak' if args.mode in ('train_energy', 'train_binary') else 'real')
        initial = c.original.model_root("sit_small") / "training" / initial_name / "head.pt"
        request = dict(created_utc=c.now(), authorization="Start and iteratively optimize endpoint-adversarial weak learning",
            initial_head=str(initial), initial_head_sha256=c.sha(initial),
            stopped_original_queue_preserved=True, objective="Four-source D CE; generator P/R pair loss through full guided sampler",
            real_target="Continuous shared VAE decode of dynamically sampled real training posteriors",
            strong_bank_samples=126689, current_weak="fresh standalone sample every update",
            train_feature="frozen Inception2048 with learned nonlinear conditional spectral-normalized critic",
            validation="same 1K and 5K inputs, ADM reference and coefficient search as historical full-data queue",
            target_best_5k_fid=36.688847797154665, arguments=args.arguments,
            sources=c.source_receipts())
        if args.mode == "train_moments":
            request["objective"] = "Four-source D CE; generator real-calibrated Frechet in CURRENT critic features through full guided sampler"
            request["moment_feedback"] = "Rolling fixed-Inception feature memory, re-encoded with current critic; fresh-row gradient surrogate"
            request["not_full_advfd"] = "D maximizes source classification, not Frechet distance"
            if '--conditional-moments' in args.arguments:
                request['objective'] = 'Four-source D CE; generator joint image-feature/condition Frechet through full guided sampler'
                request['condition_features'] = 'Paired one-hot class, centered and divided by sqrt(memory class prior); no tuned weight'
        if args.mode == 'train_energy':
            request['objective'] = 'Four-source D CE on previous batch; fresh conditional energy score through full guided sampler'
            request['generator_pairs'] = 'Two independent P and R draws per sampled class; both R members differentiated'
            request['gradient_scope'] = 'Unbiased fresh-pair estimate for fixed current critic; not an optimal-critic or optimizer unbiasedness claim'
            request['critic_lag'] = 'One previous detached batch, including preceding weak and guided endpoints'
            request['not_full_advfd'] = 'D optimizes CE, not FD or energy distance'
        if args.mode == 'train_binary':
            request['instruction'] = 'D only judges real versus strong/weak-extrapolated final images; W fools D'
            request['objective'] = 'Binary logistic GAN: D real/fake loss plus feature-input R1; W non-saturating fake-as-real loss'
            request['data_protocol'] = 'real_rgb_official_continuous_v2'
            request['real_target'] = 'Actual ImageNet-100 TRAINING RGB center crops; all 126689 images; no VAE reconstruction'
            request['generated_image_preprocessing'] = 'Official SiT clamp(127.5 * VAE_decode + 128, 0, 255) / 255; continuous before uint8 truncation'
            request['discriminator_outputs'] = 1
            request['train_feature'] = 'Frozen Inception2048 followed by learned class-conditional binary spectral-normalized critic'
            request['fake_source'] = 'Fresh actual full guided-sampler endpoints only'
            request.pop('strong_bank_samples')
            request.pop('current_weak')
        if "--resume" in args.arguments:
            checkpoint = Path(args.arguments[args.arguments.index("--resume") + 1]).resolve()
            request["resume_checkpoint"] = str(checkpoint)
            request["resume_checkpoint_sha256"] = c.sha(checkpoint)
        c.atomic(path, request)
        for filename, digest in request["sources"].items():
            source = Path(filename)
            saved = run / "source_snapshot" / source.relative_to(c.WORK)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(source.read_bytes())
            assert c.sha(saved) == digest
    action = run / args.action_id
    action.mkdir(parents=True, exist_ok=True)
    if (action / "launcher.json").exists():
        raise RuntimeError("This action ID was already launched")
    command = [c.PYTHON, "-u", "-m", "experiments.adversarial_weak_training_20260915.launch",
        "--hold", "--run", args.run, "--mode", args.mode, "--action-id", args.action_id,
        "--minimum-gpus", str(args.minimum_gpus)]
    if args.stop_file is not None:
        command += ["--stop-file", str(args.stop_file)]
    command += ["--", *args.arguments]
    with (action / "holder.log").open("a") as stream:
        process = subprocess.Popen(command, cwd=c.WORK, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    c.atomic(action / "launcher.json", dict(pid=process.pid, command=command, started_utc=c.now()))
    print(json.dumps(dict(pid=process.pid, run=str(run), action=str(action)), ensure_ascii=False))


if __name__ == "__main__":
    main()
