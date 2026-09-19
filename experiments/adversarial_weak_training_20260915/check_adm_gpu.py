"""Run the unchanged ADM graph on a leased GPU and compare a saved CPU result.

This writes a separate validation artifact and never replaces formal metrics.
Run with the existing adm-fid environment, which has TensorFlow's CUDA build.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main(args):
    from experiments.weak_reference_loss_20260914 import idle
    lease = Path('/tmp', f'eqvae_idle_{args.gpu}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert idle.eligible(next(r for r in idle.gpu_snapshot() if r['uuid'] == args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    assert len(devices) == 1, 'CUDA runtime unavailable: do not silently call CPU a GPU check'
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.stage / 'adm.json').read_text())
    metrics = args.output / 'adm.json'
    command = [str(Path(__file__).resolve().parents[1] / 'compute_adm_fid.py'),
        '--reference', previous['reference'], '--samples', previous['samples'],
        '--output', str(metrics), '--activations-output', str(args.output / 'activations.npz'),
        '--batch-size', str(previous['batch_size'])]
    sys.argv = command
    before = time.perf_counter()
    runpy.run_path(command[0], run_name='__main__')
    elapsed = time.perf_counter() - before
    actual = json.loads(metrics.read_text())
    thresholds = dict(fid=1e-4, inception_score=1e-4, sfid=1e-3)
    differences = {key: actual[key] - previous[key] for key in thresholds}
    passed = all(abs(differences[key]) <= thresholds[key] for key in thresholds)
    result = dict(passed=passed, gpu=args.gpu, pid=os.getpid(), seconds=elapsed,
        tf_version=tf.__version__, tf32=False, source_stage=str(args.stage),
        cpu={key: previous[key] for key in thresholds},
        gpu_metrics={key: actual[key] for key in thresholds},
        differences=differences, thresholds=thresholds,
        samples_unchanged=True, original_metrics_unchanged=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
