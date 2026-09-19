"""Deterministic numerical checks of output contracts and inference plumbing."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from experiments.sit_control_50_20260910 import core
from experiments import run_sit_control_50ideas_20260910 as catalog
from experiments.lifting_scale_sweep_20260909 import EXPS, atomic, sha


def audit_priority_trace(stats, config):
    if 'priority_trace_kind' not in stats:
        return dict(events=0)
    kind = str(stats['priority_trace_kind'])
    if kind == 'verified':
        cb, ca, fb, fa, bound, accepted = [np.asarray(stats['priority_trace_'+key])
            for key in ('coarse_before', 'coarse_after', 'fine_before', 'fine_after',
                        'discrepancy_bound', 'accepted')]
        np.testing.assert_allclose(bound, np.abs(fb-cb)+np.abs(fa-ca), rtol=2e-6, atol=1e-8)
        assert np.all((ca-cb)[accepted] > config['theta']*bound[accepted])
        assert np.all((fa-fb)[accepted] > 0)
        assert accepted.shape == cb.shape and cb.ndim == 2
        return dict(events=len(cb), accepted=int(accepted.sum()), strict_fine_improvement=True)
    assert kind == 'finite'
    losses, lengths, selected, attained = [np.asarray(stats['priority_trace_'+key])
        for key in ('losses', 'lengths', 'selected', 'attained')]
    assert losses.shape[-1] == 9 and losses.shape == lengths.shape
    for event in range(len(losses)):
        for sample in range(losses.shape[1]):
            values, norms = losses[event, sample], lengths[event, sample]
            eligible = values <= (1-config['theta'])*values[0]
            index = selected[event, sample]
            assert values[index] <= values[0] and norms[0] == 0
            assert attained[event, sample] == eligible.any()
            if eligible.any():
                assert eligible[index] and norms[index] == norms[eligible].min()
            else:
                assert values[index] == values.min()
    return dict(events=len(losses), exhaustive_choices=int(selected.size), nonincreasing=True,
                minimum_norm_among_sufficient_decreases=True)


class FakeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(101, 4, dtype=torch.float64)
        generator = torch.Generator().manual_seed(319)
        with torch.no_grad():
            self.embedding_table.weight.copy_(torch.randn((101, 4), generator=generator, dtype=torch.float64))
            self.embedding_table.weight[100].zero_()
    def forward(self, labels):
        return self.embedding_table(labels)


class FakeSemantic:
    def __init__(self):
        self.original_labels = torch.arange(100)
        self.decoded_images = self.classified_images = 0
        generator = torch.Generator().manual_seed(911)
        self.weights = torch.randn((4, 1000), generator=generator, dtype=torch.float64)
    def probabilities(self, x):
        self.decoded_images += len(x)
        self.classified_images += len(x)
        logits = x.mean((2, 3)) @ self.weights
        return logits.softmax(-1), logits[:, :100].softmax(-1)
    def target_probability(self, x, labels):
        return self.probabilities(x)[0].gather(1, labels[:, None]).squeeze(1)


class FakeRuntime:
    def __init__(self):
        self.model = SimpleNamespace(y_embedder=FakeEmbedding(),
            blocks=[SimpleNamespace(attn=SimpleNamespace(num_heads=2))])
        self.counts = dict(full=0, prefix=0)
        self.capture_weak = SimpleNamespace(value=None)
        self.portfolio_assets = {}
        self.pasted_semantic = FakeSemantic()
    def field(self, x, t, kind):
        self.counts['full' if kind == 'full' else 'prefix'] += 1
        embedding = self.model.y_embedder(self.labels)[:, :, None, None]
        result = .15*torch.tanh(.5*x)+.12*embedding*(1+.08*x)+.03*t*torch.sin(x)
        return result if kind == 'full' else .9*result
    def pair(self, x, t):
        full = self.field(x, t, 'full')
        weak = .9*full
        self.capture_weak.value = x.detach()
        return full, weak


@torch.inference_mode()
def cpu_checks():
    rows = catalog.configurations()
    assert len(rows) == 709 and sum(r['role'] == 'candidate' for r in rows) == 636
    assert len(catalog.IDEAS) == 53
    for path in core.source_assets():
        assert path.exists(), path
    generator = torch.Generator().manual_seed(5173)
    checks = {}

    # Compare RLS with an independently formed exponentially weighted LS solve.
    m = torch.zeros((3, 2, 2), dtype=torch.float64)
    p = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    gram, rhs = p.clone(), torch.zeros_like(m)
    for _ in range(25):
        a = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        y = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        m, p, _ = core.rls_update(m, p, a, y, .8)
        gram = .8*gram+a[:, :, None]*a[:, None, :]
        rhs = .8*rhs+y[:, :, None]*a[:, None, :]
        expected = torch.linalg.solve(gram, rhs.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(m, expected, rtol=1e-10, atol=1e-11)
    checks['rls_matches_weighted_batch_ls'] = True

    matrices = torch.tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]],
        [[1., 0.], [-1., 0.]], [[1., 0.], [2., 0.]],
        [[1., 1.], [-1., 1.]]], dtype=torch.float64)
    bounds = torch.tensor([[1., 2.], [1., 2.], [1., 1.], [1., 3.], [2., 2.]], dtype=torch.float64)
    solution, feasible = core.minimal_two_constraints(matrices, bounds)
    assert feasible.tolist() == [True, True, False, True, True]
    torch.testing.assert_close(solution, torch.tensor([[1., 2.], [-1., -2.], [0., 0.],
        [1.5, 0.], [0., 2.]], dtype=torch.float64), rtol=0, atol=1e-12)
    checks['qp_orientation_parallel_and_infeasible_cases'] = True

    values = torch.rand((20000, 4), generator=generator, dtype=torch.float64)
    cb, ca, fb, fa = values.unbind(1)
    bound = (fb-cb).abs()+(fa-ca).abs()
    assert ((fa-fb) >= (ca-cb)-bound-1e-15).all()
    for multiplier in (1., 1.5, 2.):
        accept = ca-cb > multiplier*bound
        assert (fa[accept] > fb[accept]).all() and accept.any()
    checks['coarse_fine_triangle_20000_cases'] = True

    losses = torch.rand((1000, 9), generator=generator, dtype=torch.float64)
    lengths = torch.rand((1000, 9), generator=generator, dtype=torch.float64)
    lengths[:, 0] = 0.
    losses[0] = 0.
    for reduction in (.05, .15, .3):
        selected, attained = core.select_finite(losses, lengths, reduction)
        trace = dict(priority_trace_kind='finite', priority_trace_losses=losses.numpy()[None],
            priority_trace_lengths=lengths.numpy()[None], priority_trace_selected=selected.numpy()[None],
            priority_trace_attained=attained.numpy()[None])
        audit_priority_trace(trace, dict(theta=reduction))
        assert selected[0] == 0
    checks['finite_selection_3000_exhaustive_cases'] = True

    # Joint output cannot vanish just because both future maps become identity.
    for residual, probability, expected_zero in [(0., .8, True), (0., .1, False), (1., .8, False)]:
        vector = torch.full((2, 64), residual, dtype=torch.float64)
        semantic = torch.full((2, 1), max(0., .6-probability), dtype=torch.float64)
        for theta in (.1, 1., 10.):
            joint = core.pack(vector, theta**.5*semantic)
            assert bool((joint == 0).all()) == expected_zero
    checks['semantic_golden_zero_set_and_terminal_noncollapse'] = True

    rt = FakeRuntime()
    z = torch.randn((2, 4, 8, 8), generator=generator, dtype=torch.float64)
    labels = torch.tensor([2, 9])
    rt.labels = labels
    checks['limiting_states'] = core.limiting_checks(rt, z, labels)
    probes, trajectories = [], []
    for method in catalog.IDEAS:
        selected = [r for r in rows if r['idea_id'] == method['id'] and r['strength'] == 2.75]
        for config in selected:
            history = {}
            x = z.clone()
            # Five-event controllers must exercise their histories; all other
            # objectives are checked at two physically different sample times.
            for step in method['events']:
                context = core.event_context(z, step, history)
                x, record = core.calibrate(rt, x, step/64, config, context, config['strength'])
                assert torch.isfinite(x).all() and all(np.isfinite(v) for v in record.values()), config['arm']
                assert record['after'] <= record['before']+1e-8, (config['arm'], record)
                assert rt.labels is labels and len(rt.model.y_embedder._forward_hooks) == 0
            probes.append(config['arm'])
        config = selected[-1]
        x, stats = core.sample(rt, z, labels, config)
        assert torch.isfinite(x).all(), config['arm']
        audit_priority_trace(stats, config)
        trajectories.append(config['arm'])
    checks['all_structural_settings_and_histories'] = probes
    checks['all_53_full_fake_trajectories'] = trajectories
    result = dict(passed=True, checks=checks, no_gpu=True, no_fid_used=True,
        sources={path: sha(path) for path in (__file__, core.__file__, catalog.__file__)})
    root = EXPS/'sit_control_output_50ideas_20260910'
    root.mkdir(parents=True, exist_ok=True)
    atomic(root/'cpu_checks.json', result)
    print(dict(cpu_checks_passed=True, structural_settings=len(probes), full_trajectories=len(trajectories)), flush=True)
    return result


if __name__ == '__main__':
    cpu_checks()
