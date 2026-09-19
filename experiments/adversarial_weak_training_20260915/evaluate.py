"""Original sampling inputs, solver, batch shape and ADM metric; trained head only."""

import argparse
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from experiments.guidance_dynamic_50k_20260915.models import Adapter
from experiments.guidance_dynamic_50k_20260915.sampling import integrate
from . import common as c
from .reuse_strong import reuse_strong_one_k


@torch.inference_mode()
def main(args):
    stop_file = getattr(args, "stop_file", None) or c.ROOT / "STOP_AFTER_CURRENT"
    stopped = stop_file.exists
    rank, world = c.setup()
    adapter = Adapter("sit_small")
    head = adapter.loaded_head("real")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    digest = c.sha(args.checkpoint)
    branch = f"step{state['step']:06d}_{args.weights}"
    run = c.ROOT / args.run
    bank = c.original.model_root("sit_small") / "quality_inputs"
    noise = np.load(bank / "noise.npy", mmap_mode="r")
    labels = np.load(bank / "labels.npy")
    if rank == 0:
        # Check the evaluation path against one preserved baseline batch without
        # spending another 1K evaluation or touching the old queue.
        start_noise = torch.from_numpy(np.array(noise[:8])).cuda()
        start_labels = torch.from_numpy(labels[:8]).long().cuda()
        baseline, _ = integrate(adapter, head, "real", 1., start_noise, start_labels)
        previous = c.original.model_root("sit_small") / "points/real__c0040/batches/00000.npz"
        with np.load(previous) as batch:
            np.testing.assert_array_equal(baseline.cpu().numpy(), batch["latents"])
            np.testing.assert_array_equal(adapter.pixels(baseline), batch["arr_0"])
        c.atomic(run / "quality" / branch / "baseline_parity.json", dict(passed=True,
            preserved_baseline_batch=str(previous), checkpoint_sha256=digest,
            same_noise=True, same_solver=True, same_pixels=True, checked_utc=c.now()))
    head.load_state_dict(state[args.weights])
    c.barrier()
    scores = []
    gpu_policy = c.ROOT / 'ADM_GPU_VALIDATED.json'
    use_gpu_score = gpu_policy.exists() and c.read(gpu_policy).get('enabled', False)
    for tick in args.ticks:
        if stopped():
            break
        point = run / "quality" / branch / f"c{tick:04d}"
        stage = point / f"n{args.samples}"
        if tick == 0 and args.samples == 1000:
            if rank == 0:
                reuse_strong_one_k(point, args.checkpoint, digest, args.run,
                    state["step"], args.weights, noise, labels)
            c.barrier()
            continue
        for start in tuple(range(0, args.samples, 8))[rank::world]:
            if stopped():
                break
            path = point / "batches" / f"{start:05d}.npz"
            receipt = path.with_suffix(".json")
            if receipt.exists():
                record = c.read(receipt)
                assert record["checkpoint_sha256"] == digest
                assert record["sha256"] == c.sha(path)
                assert record["tick"] == tick and record["coefficient"] == tick / 40.
                assert record["noise_sha256"] == c.original.array_sha(noise[start:start + record["samples"]])
                continue
            z = torch.from_numpy(np.array(noise[start:start + 8])).cuda()
            y = torch.from_numpy(labels[start:start + 8]).long().cuda()
            output, counts = integrate(adapter, head, "real", tick / 40., z, y)
            pixels = adapter.pixels(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, arr_0=pixels, latents=output.cpu().numpy(),
                    labels=y.cpu().numpy(), start=start)
            temporary.replace(path)
            c.atomic(receipt, dict(sha256=c.sha(path), checkpoint_sha256=digest,
                checkpoint=str(args.checkpoint), counts=counts, start=start,
                noise_sha256=c.original.array_sha(noise[start:start + len(y)]),
                tick=tick, coefficient=tick / 40., samples=len(y)))
            c.atomic(stage / f"progress_rank{rank}.json", dict(start=start, n=args.samples, updated_utc=c.now()))
        c.barrier()
        if stopped():
            break
        if rank == 0:
            images, records = [], []
            for start in range(0, args.samples, 8):
                path = point / "batches" / f"{start:05d}.npz"
                record = c.read(path.with_suffix(".json"))
                assert record["checkpoint_sha256"] == digest and record["sha256"] == c.sha(path)
                assert record["noise_sha256"] == c.original.array_sha(noise[start:start + record["samples"]])
                with np.load(path) as batch:
                    np.testing.assert_array_equal(batch["labels"], labels[start:start + len(batch["labels"])])
                    assert int(batch["start"]) == start
                    images.append(batch["arr_0"])
                records.append(dict(file=str(path), sha256=record["sha256"]))
            earlier_path = point / "n1000/summary.json"
            first_1000_reused = args.samples == 5000 and earlier_path.exists()
            if first_1000_reused:
                earlier = c.read(earlier_path)
                assert earlier["n"] == 1000 and earlier["checkpoint_sha256"] == digest
                assert len(earlier["records"]) == 125
                assert records[:len(earlier["records"])] == earlier["records"]
            sample_path = stage / "samples.npz"
            stage.mkdir(parents=True, exist_ok=True)
            with sample_path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, arr_0=np.concatenate(images))
            sample_path.with_suffix(".tmp").replace(sample_path)
            summary = dict(complete=True, valid=True, n=args.samples, tick=tick, coefficient=tick / 40.,
                step=state["step"], weights=args.weights, run=args.run, checkpoint_sha256=digest,
                records=records, samples_path=str(sample_path), samples_sha256=c.sha(sample_path),
                first_1000_reused=first_1000_reused, completed_utc=c.now())
            c.atomic(stage / "summary.json", summary)
            scores = [process for process in scores if process.poll() is None]
            while len(scores) >= (1 if use_gpu_score else 4):
                time.sleep(2)
                scores = [process for process in scores if process.poll() is None]
            with (stage / "score_holder.log").open("a") as stream:
                gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0] if use_gpu_score else None
                score_command = [c.PYTHON, "-u", "-m",
                    "experiments.adversarial_weak_training_20260915.score", "--stage", str(stage)]
                if gpu:
                    score_command += ['--shared-gpu', gpu]
                process = subprocess.Popen(score_command,
                    cwd=c.WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            c.atomic(stage / "score_launch.json", dict(pid=process.pid, started_utc=c.now(),
                backend='validated ADM CUDA FP32' if gpu else 'CPU ADM', shared_gpu=gpu))
            scores.append(process)
            print("samples_complete", tick / 40., args.samples, "score_pid", process.pid, flush=True)
        c.barrier()
    if use_gpu_score:
        if rank == 0:
            for process in scores:
                if process.wait():
                    raise RuntimeError('ADM scorer failed; inspect score_holder.log')
        c.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "head"), default="ema")
    parser.add_argument("--ticks", type=int, nargs="+", default=[40])
    parser.add_argument("--samples", type=int, choices=(1000, 5000), default=1000)
    parser.add_argument("--stop-file", type=Path,
                        help="Stop marker for a separately authorized evaluation queue; default preserves the original queue")
    main(parser.parse_args())
