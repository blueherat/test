"""Summarize a completed search and check its paired evaluation provenance on CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


EXPERIMENTS = Path('/home/zhoushunyu/data/eqvae/experiments')
OLD = EXPERIMENTS / 'guidance_dynamic_50k_20260915/sit_small'
NEW = EXPERIMENTS / 'adversarial_weak_training_20260915'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main(args):
    run = NEW / args.run
    branch = f'step{args.step:06d}_{args.weights}'
    search = run / f'search_{branch}_full'
    result = read(search / 'result.json')
    assert result['complete']
    chosen = read(search / 'selected.json')
    candidates = {row['tick']: row for stage in result['stages'] for row in stage['rows']}
    expected = sorted(candidates.values(), key=lambda row: (row['fid'], row['tick']))[:2]
    assert [r['tick'] for r in expected] == [r['tick'] for r in chosen['selected']]
    assert len({r['tick'] for r in chosen['selected']}) == 2
    for index, stage in enumerate(result['stages'][:-1]):
        intervals = stage['selected_intervals']
        left = min(i['left'] for i in intervals)
        right = max(i['right'] for i in intervals)
        width = round(result['stages'][index + 1]['width'] * 40)
        assert stage['next_ticks'] == list(range(left, right + 1, width))
    parity = read(run / 'quality' / branch / 'baseline_parity.json')
    assert parity['passed'] and parity['same_noise'] and parity['same_solver'] and parity['same_pixels']
    noise = np.load(OLD / 'quality_inputs/noise.npy', mmap_mode='r')
    baseline = read(OLD / 'points/guided_weak__c0042/n5000/metrics.json')
    old_adm = read(OLD / 'points/guided_weak__c0042/n5000/adm.json')
    comparison = []
    for selected in chosen['selected']:
        point = run / 'quality' / branch / f"c{selected['tick']:04d}"
        small = read(point / 'n1000/metrics.json')
        large = read(point / 'n5000/metrics.json')
        adm = read(point / 'n5000/adm.json')
        assert small['n'] == 1000 and large['n'] == 5000 and large['first_1000_reused']
        assert large['records'][:125] == small['records'] and len(large['records']) == 625
        assert large['reference'] == old_adm['reference']
        assert large['reference_sha256'] == digest(Path(old_adm['reference']))
        assert large['checkpoint_sha256'] == parity['checkpoint_sha256']
        assert digest(Path(large['samples_path'])) == large['samples_sha256']
        counts = None
        for start, record in zip(range(0, 5000, 8), large['records']):
            receipt = read(Path(record['file']).with_suffix('.json'))
            assert record['sha256'] == receipt['sha256']
            assert receipt['checkpoint_sha256'] == large['checkpoint_sha256']
            assert receipt['start'] == start and receipt['samples'] == 8
            assert receipt['noise_sha256'] == hashlib.sha256(np.ascontiguousarray(noise[start:start + 8]).tobytes()).hexdigest()
            assert receipt['tick'] == selected['tick']
            counts = receipt['counts'] if counts is None else counts
            assert receipt['counts'] == counts
        assert counts == baseline['counts']
        comparison.append(dict(coefficient=selected['tick'] / 40,
            fid_1k=small['fid'], fid_5k=large['fid'], is_5k=large['inception_score'],
            sfid_5k=adm.get('sfid'), delta_from_previous_best_5k=large['fid'] - baseline['fid'],
            evaluator=large.get('evaluator', 'legacy CPU ADM'),
            counts=counts, metrics=str(point / 'n5000/metrics.json'),
            checkpoint_sha256=large['checkpoint_sha256']))
    audit = dict(complete=True, run=args.run, step=args.step, weights=args.weights,
        checked_utc=datetime.now(timezone.utc).isoformat(), rows=comparison,
        previous_best_5k=dict(fid=baseline['fid'], inception_score=baseline['inception_score'],
            sfid=old_adm.get('sfid'), coefficient=1.05),
        same_noise_bank=True, same_reference=True, same_call_budget=True,
        exact_original_1k_prefix_reused=True, connected_search=True,
        coefficients_selected_before_5k=chosen['frozen_before_5k_utc'],
        five_k_is_independent_holdout=False,
        scope='Same SiT model, stored inputs, 64 Heun steps and ADM protocol; no statistical significance claim')
    output = search / 'comparison_audit.json'
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--weights', choices=('head', 'ema'), default='head')
    main(parser.parse_args())
