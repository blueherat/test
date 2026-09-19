"""Summarize measured refinement probes, replay audits and repeat variability."""
import itertools
import json
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance')
OLD, NEW = ROOT/'performance_20260919', ROOT/'refinement_20260919'
DOC = Path(__file__).resolve().parents[1]/'docs/classifier_guidance'
REFERENCES = dict(sit_small='train_graph_checkpoint', jit='jit_train_optimized', raev2='raev2_train_optimized')
PROBES = dict(
    sit_trajectory='reject_no_stable_speed_gain', jit_trajectory='reject_slower_more_memory',
    raev2_trajectory='reject_no_gain_more_memory', sit_channels_last='reject_gradient_difference',
    sit_compile='reject_gradient_difference', sit_time_cache='reject_no_stable_gain',
    raev2_invariants='reject_no_stable_gain', sit_staged='reject_slower_small_memory_gain',
    raev2_checkpoint='optional_memory_speed_tradeoff',
)


def read(path):
    value = json.loads(path.read_text())
    if not value.get('complete'):
        raise ValueError(f'Incomplete evidence: {path}')
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return dict(relative_error=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))))


def main():
    probes, batches = {}, {}
    for name, decision in PROBES.items():
        path = NEW/name
        record = read(path/'result.json')
        reference = OLD/REFERENCES[record['model']]
        before = read(reference/'result.json')
        assert before['batch'] == record['batch']
        gradient = difference(np.load(reference/'head_gradient.npy'), np.load(path/'head_gradient.npy'))
        probes[name] = dict(decision=decision, record=record, reference=str(reference),
            vs_previous_optimized_clipped_gradient=gradient,
            speed_ratio=before['median_seconds']/record['median_seconds'],
            reserved_memory_ratio=record['peak_reserved_gib']/before['peak_reserved_gib'],
            gradient_sha256=digest(path/'head_gradient.npy'), result_sha256=digest(path/'result.json'))
    for name in ('sit_batch12', 'jit_batch12', 'raev2_batch4'):
        path = NEW/name
        record = read(path/'result.json')
        before = read(OLD/REFERENCES[record['model']]/'result.json')
        speed = record['batch']/record['median_seconds']
        batches[name] = dict(record=record, images_per_second=speed,
            previous_batch=before['batch'], previous_images_per_second=before['batch']/before['median_seconds'],
            throughput_ratio=speed/(before['batch']/before['median_seconds']),
            comparable_loss=False, note='Different local batch; preserve global batch and update cadence in deployment')
    audits = {name: read(NEW/f'{name}_replay_audit.json') for name in ('sit', 'jit', 'raev2')}
    for record in audits.values():
        for row in record['checks']:
            assert row['endpoint_max_error'] == 0
            assert row['gradient_relative_error'] < .005 and row['gradient_cosine'] > .9999
    repeat = {}
    gradients = {}
    for kind in ('baseline', 'optimized'):
        path = NEW/f'raev2_{kind}_repeat'
        record = read(path/'result.json')
        gradients[kind] = [np.load(path/f'head_gradient_repeat_{i}.npy') for i in range(1, 4)]
        within = [dict(left=i+1, right=j+1, **difference(gradients[kind][i], gradients[kind][j]))
                  for i, j in itertools.combinations(range(3), 2)]
        repeat[kind] = dict(record=record, within=within,
            gradient_sha256=[digest(path/f'head_gradient_repeat_{i}.npy') for i in range(1, 4)])
    repeat['cross'] = [dict(baseline=i+1, optimized=j+1, **difference(a, b))
                       for i, a in enumerate(gradients['baseline']) for j, b in enumerate(gradients['optimized'])]
    smoke = NEW/'sit_global24_smoke'
    complete = read(smoke/'complete.json')
    assert json.loads((smoke/'exit.json').read_text())['exit_code'] == 0
    latest = json.loads((smoke/'latest.json').read_text())
    assert latest['replicas_identical']
    tests = json.loads((NEW/'tests_final.json').read_text())
    assert tests['exit_code'] == 0
    result = dict(complete=True, probes=probes, batch_profiles=batches, replay_audits=audits,
        tests=tests,
        final_sources={str(p.resolve()): digest(p) for p in Path(__file__).parent.glob('*.py')},
        probe_patch_sha256=digest(DOC.parents[1]/'archive/manifests/20260919/classifier_refinement_probes.patch'),
        raev2_gradient_repeats=repeat, sit_global24_smoke=dict(complete=complete, latest=latest,
            training=[json.loads(line) for line in (smoke/'train.jsonl').read_text().splitlines()],
            resume=json.loads((smoke/'resume.json').read_text())),
        evidence_root=str(NEW), numerical_scope='Sampled configurations; no long training or FID guarantee')
    output = DOC/'refinement_20260919.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(output)
    for name, value in batches.items():
        print(name, value['record']['median_seconds'], value['images_per_second'], value['throughput_ratio'])
    for kind in ('baseline', 'optimized'):
        values = [v['relative_error'] for v in repeat[kind]['within']]
        print('RAE repeat', kind, min(values), max(values))
    values = [v['relative_error'] for v in repeat['cross']]
    print('RAE cross', min(values), max(values))


if __name__ == '__main__':
    main()
