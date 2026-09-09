"""Independently check recorded energy identities, provenance and unchanged probes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'experiments/results/terminal_defect_20260908'

def main():
    output = []
    for model in ['sit', 'raev2']:
        d = json.loads((R / f'pfr_head_kinematic_response_{model}.json').read_text())
        old = json.loads((R / f'pfr_head_time_response_{model}.json').read_text())
        assert d['complete'] and d['unchanged_endpoint_parity']
        for k in ['noise_sha256', 'endpoint_sha256', 'checkpoint_sha256', 'labels', 'grid', 'full_calls', 'prefix_calls']:
            assert d[k] == old[k], k
        assert len(d['rows']) == len(old['rows']) == 128
        capture = Path('/home/zhoushunyu/data/eqvae/experiments/pfr_head_kinematic_response_20260908') / model
        for i, (p, digest) in enumerate(d['sources'].items()):
            assert hashlib.sha256((capture / f'{i}_{Path(p).name}').read_bytes()).hexdigest() == digest
        for row, prior in zip(d['rows'], old['rows']):
            for k, v in prior.items():
                assert row[k] == v, (model, k)
            a = row['explicit_energy']
            for head in ['weak', 'full']:
                e = row[f'{head}_energy']
                p = row[f'prediction_{head}_energy']
                c = row[f'explicit_{head}_prediction_cross']
                assert abs(e - (a + p + 2*c)) < 1e-10
                assert 0 <= row[f'transverse_{head}_energy'] <= e + 1e-12
            assert abs(row['cross'] - (a + row['explicit_weak_prediction_cross'] + row['explicit_full_prediction_cross'] + row['prediction_cross'])) < 1e-10
        for summary in d['summary']:
            group = [x for x in d['rows'] if x['step'] == summary['step']]
            assert len(group) == 32
            total = lambda k: sum(x[k] for x in group)
            for name, key in [('prediction_cosine', 'prediction'), ('transverse_cosine', 'transverse')]:
                value = total(f'{key}_cross') / (total(f'{key}_weak_energy') * total(f'{key}_full_energy'))**.5
                assert abs(value-summary[name]) < 1e-12
            output.append(dict(model=model, **summary))
        print(model, 'passed', d['seconds'], flush=True)
    with (R / 'pfr_head_kinematic_response_audit.json').open('x') as f:
        json.dump(dict(passed=True, summaries=output), f, indent=2)

if __name__ == '__main__':
    main()
