"""CPU policy checks: connected refinement, one-idea admission, and field semantics."""
import ast
from pathlib import Path
import tempfile
from unittest import mock
import torch
from . import config as k, planning, pipeline, worker


def main():
    k.verify()
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text())
    assert set(k.METHODS) == set(k.old.ARMS)
    assert not {'native', 'context', 'cfg_native'} & set(k.METHODS)
    assert [k.value(t) for t in k.STEPS] == [.4, .2, .1, .025]
    assert [k.value(t) for t in k.COARSE] == [0., .4, .8, 1.2, 1.6, 2.]
    scores = [dict(tick=t, fid=f, valid=True) for t, f in zip(k.COARSE, [100, 0, 0, 100, 0, 0])]
    intervals, ticks = planning.refine(scores, 16, 8)
    assert [(i['left'], i['right']) for i in intervals] == [(16, 32), (64, 80)]
    assert ticks == list(range(16, 81, 8)) and all(t in ticks for t in (40, 48, 56))
    scores[1]['valid'] = False
    assert all(i['left'] != 16 for i in planning.best_intervals(scores, 16))
    assert planning.refine([dict(tick=0, fid=1, valid=True)], 16, 8) == ([], [])
    assert k.canonical('guided_weak', 0) == k.canonical('self', 0) == k.key('strong', 0)
    assert k.canonical('ig_contrast', 0) == k.key('native', 32)
    assert k.canonical('cfg_contrast', 0) == k.key('cfg_native', 50)
    assert k.anchor(k.key('gaussian', 16)) == 'gaussian_half'
    for method in k.METHODS:
        tick = pipeline.Supervisor.default_tick(method)
        assert k.anchor(k.key(method, tick)) == method
        expected = k.old.inference_counts(method)
        assert k.counts(k.key(method, tick)) == expected

    # Algebra and floating-point order at default gains, including fixed IG/CFG bases.
    class Runtime:
        name = 'sit_small'
        labels = torch.zeros(2, dtype=torch.long)
        def field(self, z, t, kind):
            return z + (.1 if torch.all(self.labels == 100) else .3)
        def pair(self, z, t):
            full = self.field(z, t, 'full')
            return full, full - .2
    class Capture:
        values = {'context': torch.ones(2, 4, 2, 2), 'condition': torch.ones(2, 3)}
    head = lambda tokens, condition: tokens * .4
    z = torch.full((2, 4, 2, 2), .15)
    with mock.patch.object(worker.x.local, 'unpatchify', side_effect=lambda rt, value: value):
        for method in k.METHODS:
            point = k.key(method, pipeline.Supervisor.default_tick(method))
            for left in (.1, .3, .6, .8):
                t = z.new_tensor(left)
                actual = worker.field(Runtime(), head, Capture(), point, z, t, left)
                expected = worker.x.guided_field(Runtime(), head, Capture(), method, z, t, left)
                assert torch.equal(actual, expected), (method, left)
        first = worker.field(Runtime(), head, Capture(), k.key('guided_weak', 16), z, z.new_tensor(.3), .3)
        second = worker.field(Runtime(), head, Capture(), k.key('guided_weak', 32), z, z.new_tensor(.3), .3)
        full = Runtime().field(z, z.new_tensor(.3), 'full')
        torch.testing.assert_close(second - full, 2 * (first - full))

    # Exercise the real serial/adaptive controller without submitting any work.
    with tempfile.TemporaryDirectory(prefix='guidance_serial_policy_') as directory:
        root = Path(directory)
        (root / 'old').mkdir()
        (root / 'request.json').write_text('{}')
        with mock.patch.object(k, 'ROOT', root), mock.patch.object(k.old, 'ROOT', root / 'old'), \
             mock.patch.object(k, 'METHODS', ('guided_weak', 'self')), \
             mock.patch.object(planning, 'verify_output', return_value=False):
            pipeline.install()
            supervisor = pipeline.Supervisor()
            assert supervisor.advance() == 'guided_weak'
            assert 'train_self' not in supervisor.allowed_jobs('guided_weak')
            assert 'train_guided_weak' in supervisor.allowed_jobs('guided_weak')
            levels = []
            for _ in range(4):
                method = supervisor.current
                assert method == 'guided_weak'
                stage = supervisor.searches[method]['stages'][-1]
                levels.append(stage['step_ticks'])
                ticks_now = {*stage['ticks'], supervisor.default_tick(method)}
                for tick in ticks_now:
                    point = k.canonical(method, tick)
                    k.atomic(planning.point_metrics(point), dict(valid=True, fid=60 + (k.value(tick) - .7) ** 2,
                        inception_score=30., point=point))
                for name in supervisor.allowed_jobs(method):
                    supervisor.ledger[name] = dict(state='complete')
                supervisor.advance()
            assert levels == list(k.STEPS)
            assert supervisor.searches['guided_weak']['state'] == 'complete'
            assert supervisor.current == 'self'
            assert 'train_self' in supervisor.allowed_jobs('self')
            assert len({job['id'] for job in supervisor.jobs}) == len(supervisor.jobs)
            assert all(len(stage['ticks']) == len(set(stage['ticks']))
                for stage in supervisor.searches['guided_weak']['stages'])
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, connected_gap_filled=True, selected_intervals_not_disjoint_searches=True,
        single_idea_all_four_levels_before_next=True, no_previous_ig_or_mlp_sweep=True,
        all_original_19_heads_retained=True, existing_points_deduplicated=True,
        default_field_exact_for_all_19_methods=True, coefficient_scaling_checked=True,
        zero_points_reuse_correct_base=True, cpu_checks_initialized_cuda=False,
        request_sha256=k.sha(k.ROOT / 'request.json'))
    k.atomic(k.ROOT / 'cpu_preflight.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
