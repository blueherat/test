"""One bounded AA inverse refinement of the existing 8+8 preflight samples.

The frozen CFG64 map and Anderson implementation are unchanged. Tight inverse
residual tolerance is 1e-7 with the same 16-update cap. This measures sensitivity
to a numerical tolerance; it is not an exact preimage-error bound for real data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.cfg_inverse_prior_20260913 import bank_anderson as bank


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def stats(a, b):
    result = bank.error_stats(a, b)
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    target_rms = float(np.sqrt(np.mean(np.asarray(b, dtype=np.float64) ** 2)))
    result.update(global_rms=float(np.sqrt(np.mean(delta ** 2))), target_rms=target_rms)
    result['relative_global_rms'] = result['global_rms'] / max(target_rms, 1e-30)
    return result


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=bank.ROOT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = args.out or args.root / 'refine_tol1e_7'
    request, data = bank.load_frozen(args.root)
    preflight = bank.read(args.root / 'preflight.json')
    if not preflight['passed']:
        raise ValueError('Requires the completed original Anderson preflight')
    for filename, digest in preflight['files_sha256'].items():
        if bank.sha(args.root / 'preflight' / filename) != digest:
            raise ValueError('Frozen preflight array hash mismatch')
    out.mkdir(parents=True, exist_ok=False)
    rows = bank.PREFLIGHT_ROWS
    metadata = dict(operation='one fixed-tolerance numerical refinement, no parameter search',
        pid=os.getpid(), device=args.device, rows=rows.tolist(), labels=data['labels'][rows].tolist(),
        source_ids=data['source_ids'][rows].tolist(), dtype='float32', tf32=False, autocast=False,
        tolerance=1e-7, max_anderson_updates=16, original_tolerance=request['inverse']['rms_tolerance'],
        script_sha256=bank.sha(__file__), bank_sources_sha256=bank.source_hashes(),
        bank_request_sha256=bank.sha(args.root / 'request.json'),
        preflight_sha256=bank.sha(args.root / 'preflight.json'),
        scope='tolerance sensitivity, not proof of exact inverse or downstream quality')
    bank.write_new(out / 'request.json', document=metadata)
    print(json.dumps(metadata), flush=True)
    rt = None
    stage = 'loading'
    start = time.monotonic()
    report = dict(passed=False, request_sha256=bank.sha(out / 'request.json'))
    try:
        rt = bank.NativeRuntime(args.device)
        labels = torch.from_numpy(data['labels'][rows]).to(args.device)
        with torch.inference_mode(), rt.context():
            for stage, filename in (('known', 'known_noise_inverse.npz'), ('real', 'real_inverse.npz')):
                old = load_npz(args.root / 'preflight' / filename)
                target = old['generated'] if stage == 'known' else data['clean'][rows]
                if stage == 'real' and not np.array_equal(target, old['clean']):
                    raise ValueError('Real source array mismatch')
                before = rt.branch_image_evaluations
                tight = bank.inverse(rt, torch.from_numpy(target).to(args.device), labels,
                    tol=1e-7, max_updates=16, save_states=True)
                inverse_cost = rt.branch_image_evaluations - before
                endpoint, states = bank.forward(rt, torch.from_numpy(tight['noise']).to(args.device),
                    labels, save_states=True)
                endpoint = endpoint.cpu().numpy()
                stage_report = dict(inverse=bank.inverse_stats(tight),
                    tight_noise_vs_loose_noise=stats(tight['noise'], old['noise']),
                    loose_roundtrip=stats(old['reconstructed'], target),
                    tight_roundtrip=stats(endpoint, target),
                    tight_endpoint_vs_loose_endpoint=stats(endpoint, old['reconstructed']),
                    inverse_branch_image_evaluations=inverse_cost,
                    forward_branch_image_evaluations=224 * len(labels))
                if inverse_cost != stage_report['inverse']['branch_image_evaluations']:
                    raise RuntimeError('Inverse cost mismatch')
                if stage == 'known':
                    stage_report['loose_preimage_error'] = stats(old['noise'], old['original_noise'])
                    stage_report['tight_preimage_error'] = stats(tight['noise'], old['original_noise'])
                report[stage] = stage_report
                arrays = {**tight, 'target': target, 'reconstructed': endpoint,
                    'forward_states': states, 'loose_noise': old['noise'],
                    'loose_reconstructed': old['reconstructed'], 'rows': rows,
                    'labels': data['labels'][rows], 'source_ids': data['source_ids'][rows]}
                if stage == 'known':
                    arrays['original_noise'] = old['original_noise']
                bank.write_new(out / f'{stage}.npz', arrays=arrays)
                print(json.dumps({stage: stage_report}), flush=True)
        report.update(passed=True, branch_image_evaluations=rt.branch_image_evaluations,
            seconds=time.monotonic() - start,
            files_sha256={p.name: bank.sha(p) for p in out.glob('*.npz')})
    except BaseException as error:
        report.update(passed=False, failed_stage=stage, error=str(error),
            branch_image_evaluations=rt.branch_image_evaluations if rt else 0,
            seconds=time.monotonic() - start)
        bank.save_failure(out, error, report)
        bank.write_new(out / 'summary.json', document=report)
        raise
    bank.write_new(out / 'summary.json', document=report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
