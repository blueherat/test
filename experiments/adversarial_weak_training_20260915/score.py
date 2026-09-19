"""Original ADM evaluator, with an explicitly validated optional CUDA backend."""

import argparse
import os
from pathlib import Path
import subprocess
import math
from . import common as c


def main(stage, shared_gpu=None):
    summary = c.read(stage / "summary.json")
    assert summary["complete"] and summary["valid"]
    assert c.sha(summary["samples_path"]) == summary["samples_sha256"]
    reference = c.original.EXPS.parent / "imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
    entry = str(c.WORK / 'experiments/compute_adm_fid.py')
    if shared_gpu:
        policy = c.read(c.ROOT / 'ADM_GPU_VALIDATED.json')
        assert policy['enabled']
        for filename, digest in policy['validation_receipts'].items():
            assert c.sha(filename) == digest and c.read(Path(filename))['passed']
        entry = str(c.WORK / 'experiments/adversarial_weak_training_20260915/adm_gpu_entry.py')
    command = ["/data/shared/envs/adm-fid/bin/python", entry,
        "--reference", str(reference), "--samples", summary["samples_path"],
        "--output", str(stage / "adm.json"), "--activations-output", str(stage / "activations.npz"),
        "--batch-size", "32"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=shared_gpu or "", OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="1")
    backend = 'ADM CUDA FP32, TF32 disabled' if shared_gpu else 'unchanged CPU ADM'
    with (stage / "evaluation.log").open("a") as log:
        try:
            subprocess.run(command, cwd=c.WORK, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            if not shared_gpu:
                raise
            c.atomic(stage / 'gpu_score_fallback.json', dict(error=repr(error), gpu=shared_gpu, utc=c.now()))
            command[1] = str(c.WORK / 'experiments/compute_adm_fid.py')
            environment['CUDA_VISIBLE_DEVICES'] = ''
            subprocess.run(command, cwd=c.WORK, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            backend = 'unchanged CPU ADM after recorded CUDA failure'
    result = c.read(stage / "adm.json")
    assert math.isfinite(result["fid"]) and math.isfinite(result["inception_score"])
    c.atomic(stage / "metrics.json", dict(summary, fid=result["fid"], inception_score=result["inception_score"],
        reference=str(reference), reference_sha256=c.sha(reference), evaluator=backend,
        shared_evaluation_gpu=shared_gpu,
        scored_utc=c.now()))
    print(stage, result["fid"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument('--shared-gpu', help='GPU already leased by the parent sampling action; no independent admission')
    args = parser.parse_args()
    main(args.stage, args.shared_gpu)
