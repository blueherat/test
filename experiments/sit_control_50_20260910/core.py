"""Counted, inference-only control-output candidates on the fixed small SiT."""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from experiments import run_sit_control_50ideas_20260910 as catalog
from experiments.sit_fsg_pasted_20260910 import core as seven
from experiments.sit_fsg_followup_20260910 import core as prior
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.lifting_scale_sweep_20260909 import array_sha

IDEAS = {row['key']: row for row in catalog.IDEAS}
FAMILIES = {f'i{row["id"]:02d}_{row["key"]}': row for row in catalog.IDEAS}
SPECIAL = set(IDEAS) | {'sde_semantic_direct'}
configurations = catalog.configurations
norm, inner, cap, projection = old.norm, old.inner, old.cap, old.projection
flat, energy, choose = prior.flat, prior.energy, prior.choose
EPS = 1e-10


def install():
    pass


def source_assets():
    return [seven.MANIFEST, seven.CLASSIFIER_WEIGHTS]


def pack(*pieces):
    return torch.cat([flat(x)/math.sqrt(x[0].numel()) for x in pieces], dim=1)


def ortho_basis(first, random, dimensions=2, *, tangent=None):
    values = [first]
    if dimensions > 2:
        values.append(old.lowpass(first))
    values += list(random.unbind(0))
    answer = []
    for value in values:
        if tangent is not None:
            value = value-projection(value, tangent)
        for previous in answer:
            value = value-projection(value, previous)
        # Degenerate directions become zero columns; regularization handles them.
        value = value/norm(value).clamp_min(EPS)
        answer.append(value)
        if len(answer) == dimensions:
            break
    assert len(answer) == dimensions
    return torch.stack([flat(value) for value in answer], dim=-1)


def combine(q, coefficient, like):
    with old.exact_matmul():
        return (q @ coefficient.unsqueeze(-1)).squeeze(-1).reshape_as(like)


def ridge_step(residual, jacobian, *, ridge=.001, metric=None):
    with old.exact_matmul():
        j, r = jacobian.double(), residual.double()
        gram = j.transpose(1, 2) @ j
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
        matrix = gram+ridge*scale[:, None, None]*torch.eye(j.shape[-1], device=j.device, dtype=j.dtype)
        if metric is not None:
            matrix = matrix+metric.double()
        rhs = -(j.transpose(1, 2) @ r.unsqueeze(-1))
        return torch.linalg.solve(matrix, rhs).squeeze(-1).to(residual.dtype)


def finite_jacobian(fn, z, q, epsilon, original=None):
    original = fn(z) if original is None else original
    columns = []
    for index in range(q.shape[-1]):
        changed = fn(z+epsilon*q[:, :, index].reshape_as(z))
        columns.append((changed-original)/epsilon.flatten(1))
    return original, torch.stack(columns, -1)


def event_context(noise, step, history=None):
    seed = int(array_sha(noise.detach().cpu().numpy())[:15], 16)
    generator = torch.Generator(device=noise.device).manual_seed((seed+104729+7919*step) % (2**63-1))
    random = torch.randn((16, *noise.shape), generator=generator, device=noise.device, dtype=noise.dtype)
    future = torch.randn((4, 16, *noise.shape), generator=generator, device=noise.device, dtype=noise.dtype)
    return SimpleNamespace(xi=random[0], random=random, future_noise=future,
                           history={} if history is None else history, step=step)


class Probe(seven.FutureProbe):
    def __init__(self, rt, z, t, context, *, steps=None, horizon=None):
        super().__init__(rt, z, t, context, steps=catalog.FUTURE_STEPS if steps is None else steps)
        self.end_time = 1. if horizon is None else min(1., self.t+horizon)
        self.grid = [self.t+(self.end_time-self.t)*i/self.steps for i in range(self.steps+1)]
        self.b = 1-self.t
        self.m = z+self.b*self.c
        self.noise = z-self.t*self.c
        self.q = ortho_basis(self.d, context.random)

    def field(self, z, t, condition):
        before = self.rt.labels
        embedding = None
        if isinstance(condition, str):
            self.rt.labels = torch.full_like(self.labels, 100) if condition == 'u' else self.labels
        elif condition.dtype == torch.long:
            self.rt.labels = condition
        else:
            self.rt.labels = self.labels
            embedding = condition
        try:
            with old.embedding_override(self.rt, embedding), old.exact_matmul():
                return self.rt.field(z, z.new_tensor(t), 'full')
        finally:
            self.rt.labels = before

    def end(self, z, condition='u', *, noise=None):
        if noise is None and z is self.z and isinstance(condition, str):
            key = 'end_'+condition
            if key not in self.cache:
                self.cache[key] = self.rollout(z, condition)[-1]
            return self.cache[key]
        return self.rollout(z, condition, noise=noise)[-1]

    def gap(self, z):
        return self.end(z, 'c')-self.end(z, 'u')

    def scale(self):
        if 'scale' not in self.cache:
            self.cache['scale'] = energy(self.gap(self.z)).sqrt().clamp_min(1e-5)[:, None]
        return self.cache['scale']

    def probabilities(self, z):
        return self.semantic().probabilities(z)

    def target_probability(self, z):
        return self.semantic().target_probability(z, self.labels)

    def probability_margin(self, z):
        full, _ = self.probabilities(z)
        logs = full.clamp_min(1e-12).log()
        original = self.semantic().original_labels[self.labels]
        target = logs.gather(1, original[:, None]).squeeze(1)
        competitors = logs.clone()
        competitors.scatter_(1, original[:, None], -torch.inf)
        return target-competitors.max(1).values

    def rivals(self, count=1):
        if not hasattr(self.rt, 'control50_rivals'):
            table = F.normalize(self.rt.model.y_embedder.embedding_table.weight[:100], dim=-1)
            with old.exact_matmul():
                values = table @ table.T
            values.fill_diagonal_(-torch.inf)
            self.rt.control50_rivals = values.topk(3, dim=-1).indices
        return self.rt.control50_rivals[self.labels, :count]

    def clean_proxy(self, x, t, mix=0.):
        uncond = x+(1-t)*self.field(x, t, 'u')
        pu = self.target_probability(uncond)
        if mix == 0.:
            return pu
        cond = x+(1-t)*self.field(x, t, 'c')
        pc = self.target_probability(cond)
        return (1-mix)*pu+mix*pc

    def reverse_null(self, terminal):
        state = terminal
        for index in range(self.steps, 0, -1):
            time, end = self.grid[index], self.grid[index-1]
            h = end-time
            first = self.field(state, time, 'u')
            second = self.field(state+h*first, end, 'u')
            state = state+(h/2)*(first+second)
        return state

    def planned_null(self, start, second_shift):
        state = start
        split = max(1, min(self.steps-1, round(.25/max(self.end_time-self.t, .25)*self.steps)))
        for index in range(self.steps):
            if index == split:
                state = state+second_shift
            t, end = self.grid[index:index+2]
            h = end-t
            first = self.field(state, t, 'u')
            second = self.field(state+h*first, end, 'u')
            state = state+(h/2)*(first+second)
        return state


def channel_moments(x):
    mean = x.mean((2, 3))
    std = (x-mean[:, :, None, None]).square().mean((2, 3)).clamp_min(1e-10).sqrt()
    return torch.cat((mean, std), dim=1)


def objective(p, key, theta):
    z, t, b = p.z, p.t, p.b
    if key == 'sde_semantic_direct':
        noise = p.context.future_noise[0]
        return lambda x: (theta-p.target_probability(p.end(x, noise=noise))).clamp_min(0)[:, None]
    if key in ('coupled_sde', 'sde_moments', 'sliced_sde', 'sde_semantic_success'):
        count = int(theta) if key == 'coupled_sde' else (4 if key == 'sliced_sde' else 2)
        projections = (torch.stack([flat(value/norm(value).clamp_min(EPS))
                       for value in p.context.random[:int(theta)]], -1)
                       if key == 'sliced_sde' else None)

        def stochastic(x):
            uncond = torch.stack([p.end(x, 'u', noise=p.context.future_noise[i]) for i in range(count)])
            if key == 'sde_semantic_success':
                probability = torch.stack([p.target_probability(value) for value in uncond])
                return (theta-probability.mean(0)).clamp_min(0)[:, None]
            cond = torch.stack([p.end(x, 'c', noise=p.context.future_noise[i]) for i in range(count)])
            if key == 'coupled_sde':
                return pack(*list((cond-uncond).unbind(0)))
            if key == 'sde_moments':
                cm, um = cond.mean(0), uncond.mean(0)
                cs = (cond-cm).square().mean(0).clamp_min(1e-12).sqrt()
                us = (uncond-um).square().mean(0).clamp_min(1e-12).sqrt()
                return pack(cm-um, math.sqrt(theta)*(cs-us))
            with old.exact_matmul():
                cproj = torch.einsum('kbd,bdq->bkq', cond.flatten(2), projections)
                uproj = torch.einsum('kbd,bdq->bkq', uncond.flatten(2), projections)
            return flat(cproj.sort(1).values-uproj.sort(1).values)
        return stochastic
    if key in ('antithetic_root', 'worst_neighbor', 'robust_semantic_success'):
        shift = theta*b*p.context.xi

        def robust(x):
            if key == 'robust_semantic_success':
                scores = torch.stack([(.6-p.target_probability(p.end(x+s*shift))).clamp_min(0)
                                      for s in (-1, 1)], 1)
                return scores.max(1).values[:, None]
            residuals = [p.gap(x+s*shift) for s in ((-1, 1) if key == 'antithetic_root' else (0, -1, 1))]
            if key == 'antithetic_root':
                return pack(*residuals)
            stacked = torch.stack([flat(value) for value in residuals], 1)
            index = stacked.square().mean(-1).argmax(1)
            return stacked[torch.arange(len(x), device=x.device), index]
        return robust
    if key in ('terminal_inverse', 'two_stage_mpc'):
        target = p.end(z, 'c').detach()
        return lambda x: flat(p.end(x)-target)
    if key in ('semantic_margin', 'semantic_handoff', 'recursive_semantic_response'):
        if key == 'semantic_margin':
            return lambda x: (theta-p.probability_margin(p.end(x))).clamp_min(0)[:, None]
        target = theta if key == 'semantic_handoff' else .6
        return lambda x: (target-p.target_probability(p.end(x))).clamp_min(0)[:, None]
    if key == 'semantic_teacher':
        target = p.probabilities(p.end(z, 'c'))[1].detach()
        onehot = F.one_hot(p.labels, 100).to(z.dtype)
        target = ((1-theta)*target+theta*onehot).clamp_min(1e-12).sqrt()
        return lambda x: p.probabilities(p.end(x))[1].clamp_min(1e-12).sqrt()-target
    if key == 'semantic_distribution':
        def distribution(x):
            c = p.probabilities(p.end(x, 'c'))[1].clamp_min(1e-12)
            u = p.probabilities(p.end(x))[1].clamp_min(1e-12)
            return (c.log()/theta).softmax(-1).sqrt()-(u.log()/theta).softmax(-1).sqrt()
        return distribution
    if key == 'semantic_style_guard':
        original = channel_moments(p.end(z)).detach()
        scale = original.abs().clamp_min(.1)
        return lambda x: pack((.6-p.target_probability(p.end(x))).clamp_min(0)[:, None],
                              math.sqrt(theta)*(channel_moments(p.end(x))-original)/scale)
    if key == 'bayes_calibration':
        q = p.q
        coefficient = b/max(t, .001)
        def calibrated(x):
            gap = flat(p.field(x, t, 'c')-p.field(x, t, 'u'))
            logp = p.clean_proxy(x, t, theta).clamp_min(1e-8).log()
            derivatives = []
            for index in range(q.shape[-1]):
                changed = p.clean_proxy(x+p.eps*q[:, :, index].reshape_as(x), t, theta).clamp_min(1e-8).log()
                derivatives.append((changed-logp)/p.eps.flatten())
            with old.exact_matmul():
                local = (q.transpose(1, 2) @ gap.unsqueeze(-1)).squeeze(-1)
            return local-coefficient*torch.stack(derivatives, dim=1)
        return calibrated
    if key == 'readout_transport':
        short = Probe(p.rt, z, t, p.context, steps=4, horizon=.25)
        def transported(x):
            moved = short.end(x)
            beginning = p.clean_proxy(x, t)
            ending = p.clean_proxy(moved, short.end_time)
            terminal = (.6-p.target_probability(p.end(x))).clamp_min(0)
            return torch.stack((terminal, math.sqrt(theta)*(ending-beginning)), dim=1)
        return transported
    if key == 'renoise_contract':
        target = p.end(z, 'c').detach()
        s = t+theta*b
        future = Probe(p.rt, z, s, p.context)
        def renoise(x):
            mean = x+b*p.field(x, t, 'c')
            view = s*mean+(1-s)*p.context.xi
            return flat(future.end(view)-target)
        return renoise

    # All scales and weighting matrices in this block are frozen at this event.
    scale = p.scale()
    if key in ('root_integral', 'root_derivative'):
        if 'output_scale' not in p.context.history:
            p.context.history['output_scale'] = scale.detach()
        scale = p.context.history['output_scale']
    base_c = p.end(z, 'c').detach()
    if key == 'whitened_root':
        values = base_c.flatten(2)
        values = values-values.mean(2, keepdim=True)
        with old.exact_matmul():
            cov = values.double() @ values.double().transpose(1, 2)/values.shape[-1]
            ridge = theta*cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
            vals, vectors = torch.linalg.eigh(cov+ridge[:, None, None]*torch.eye(4, device=z.device, dtype=torch.float64))
            whitening = ((vectors*vals.clamp_min(1e-10).rsqrt()[:, None, :]) @ vectors.transpose(1, 2)).to(z.dtype)
    if key == 'patch_bottleneck':
        values = F.avg_pool2d(p.gap(z).square().mean(1, keepdim=True), 4)
        logits = theta*values/values.mean((2, 3), keepdim=True).clamp_min(EPS)
        weights = logits.flatten(1).softmax(1).reshape_as(values)*values[0].numel()
        patch_weights = F.interpolate(weights.sqrt(), size=z.shape[-2:], mode='nearest')
    if key == 'condition_continuum':
        table = p.rt.model.y_embedder.embedding_table
        ec, eu = table(p.labels), table(torch.full_like(p.labels, 100))
        embedding = eu+theta*(ec-eu)
    if key == 'refinement_consistency':
        fine = Probe(p.rt, z, t, p.context, steps=16)
    if key == 'rolling_invariance':
        epsilon = 1e-3
        shifted = Probe(p.rt, z, t+epsilon, p.context, horizon=.25)
    if key == 'performance_funnel':
        if 'funnel_scale' not in p.context.history:
            p.context.history['funnel_scale'] = scale.detach()
            p.context.history['funnel_t'] = t
        radius = p.context.history['funnel_scale']*(b/(1-p.context.history['funnel_t']))**theta

    def value(x):
        residual = p.gap(x)
        normalized = flat(residual)/scale
        if key in ('terminal_root', 'rolling_root', 'terminal_inverse', 'noise_tangent_root',
                   'root_cotangent', 'root_sliding', 'root_integral', 'root_derivative',
                   'adaptive_trust', 'output_disturbance', 'future_input_metric',
                   'richer_input_subspace', 'event_triggered_root', 'golden_handoff',
                   'full_cycle', 'anderson_root'):
            return normalized
        if key == 'conditional_anchor':
            return pack(normalized, math.sqrt(theta)*flat(p.end(x, 'c')-base_c)/scale)
        if key == 'path_energy':
            c, u = p.pair_paths(x)
            pieces = [normalized]
            for index in (p.steps//4, p.steps//2, 3*p.steps//4):
                pieces.append(math.sqrt(theta/3)*flat(c[index]-u[index])/scale)
            return pack(*pieces)
        if key == 'golden_tube':
            u = p.rollout(x, 'u')
            pieces = [normalized]
            for index in (p.steps//4, p.steps//2):
                cond = p.rollout(u[index], 'c', start=index)[-1]
                pieces.append(math.sqrt(theta/2)*flat(cond-u[-1])/scale)
            return pack(*pieces)
        if key == 'terminal_tangent':
            gap = p.field(x, t, 'c')-p.field(x, t, 'u')
            length = norm(gap).clamp_min(EPS)
            derivative = (p.end(x+p.eps*gap/length, 'c')-p.end(x, 'c'))/p.eps*length
            return pack(normalized, math.sqrt(theta)*b*flat(derivative)/scale)
        if key == 'rolling_invariance':
            moved = x+epsilon*p.field(x, t, 'u')
            derivative = (shifted.gap(moved)-residual)/epsilon
            return pack(normalized, math.sqrt(theta)*.25*flat(derivative)/scale)
        if key == 'condition_continuum':
            return pack(normalized, flat(p.end(x, embedding)-p.end(x))/scale)
        if key == 'semantic_bernoulli':
            c = p.target_probability(p.end(x, 'c')).clamp(1e-8, 1-1e-8)
            u = p.target_probability(p.end(x)).clamp(1e-8, 1-1e-8)
            sem = torch.stack((c.sqrt()-u.sqrt(), (1-c).sqrt()-(1-u).sqrt()), 1)
            return pack(sem, math.sqrt(theta)*normalized)
        if key == 'semantic_golden_set':
            failure = (.6-p.target_probability(p.end(x))).clamp_min(0)[:, None]
            return pack(normalized, math.sqrt(theta)*failure)
        if key in ('contrastive_future', 'multi_rival_future'):
            count = 1 if key == 'contrastive_future' else 3
            rivals = p.rivals(count)
            negative = torch.stack([energy(flat(p.end(x)-p.end(x, rivals[:, i]))/scale)
                                    for i in range(count)], 1).min(1).values
            positive = energy(normalized)
            margin, weight = (.25, theta) if count == 1 else (theta, 1.)
            hinge = (margin+positive-negative).clamp_min(0)
            return pack(normalized, math.sqrt(weight)*hinge[:, None])
        if key == 'whitened_root':
            with old.exact_matmul():
                return flat(whitening @ residual.flatten(2))
        if key == 'coarse_detail_root':
            low = old.lowpass(residual)
            return pack(flat(low)/scale, math.sqrt(theta)*flat(residual-low)/scale)
        if key == 'patch_bottleneck':
            return flat(residual*patch_weights)/scale
        if key == 'refinement_consistency':
            return pack(normalized, math.sqrt(theta)*flat(fine.gap(x)-residual)/scale)
        if key == 'bounded_deadzone':
            return normalized.sign()*(normalized.abs()-theta).clamp_min(0)
        if key == 'performance_funnel':
            raw = flat(residual)
            return raw.sign()*(raw.abs()-radius).clamp_min(0)/p.context.history['funnel_scale']
        if key == 'bayes_curl':
            directions = p.q
            gap = p.field(x, t, 'c')-p.field(x, t, 'u')
            columns = []
            for index in range(2):
                changed = x+p.eps*directions[:, :, index].reshape_as(x)
                derivative = (p.field(changed, t, 'c')-p.field(changed, t, 'u')-gap)/p.eps
                with old.exact_matmul():
                    columns.append((directions.transpose(1, 2) @ flat(derivative).unsqueeze(-1)).squeeze(-1))
            matrix = torch.stack(columns, -1)
            curl = matrix[:, 0, 1]-matrix[:, 1, 0]
            return pack(normalized, math.sqrt(theta)*b*curl[:, None]/scale)
        raise KeyError(key)
    return value


def minimal_two_constraints(matrix, bound):
    """Exact unconstrained min-norm solution of two half-planes in R2."""
    b, _, dimension = matrix.shape
    assert dimension == 2 and bound.shape == (b, 2)
    candidates = [torch.zeros((b, 2), device=matrix.device, dtype=matrix.dtype)]
    for index in range(2):
        vector = matrix[:, index]
        candidates.append(vector*(bound[:, index]/vector.square().sum(1).clamp_min(EPS))[:, None])
    determinant = matrix[:, 0, 0]*matrix[:, 1, 1]-matrix[:, 0, 1]*matrix[:, 1, 0]
    safe = torch.where(determinant.abs() > 1e-12, determinant, torch.ones_like(determinant))
    intersection = torch.stack((bound[:, 0]*matrix[:, 1, 1]-matrix[:, 0, 1]*bound[:, 1],
                                matrix[:, 0, 0]*bound[:, 1]-bound[:, 0]*matrix[:, 1, 0]), 1)/safe[:, None]
    candidates.append(intersection)
    points = torch.stack(candidates, 1)
    with old.exact_matmul():
        achieved = torch.einsum('bij,bkj->bki', matrix, points)
    feasible = (achieved >= bound[:, None, :]-1e-7).all(-1)
    feasible[:, -1] &= determinant.abs() > 1e-12
    costs = points.square().sum(-1).masked_fill(~feasible, torch.inf)
    best = costs.argmin(-1)
    any_feasible = feasible.any(-1)
    answer = points[torch.arange(b, device=matrix.device), best]
    return torch.where(any_feasible[:, None], answer, torch.zeros_like(answer)), any_feasible


def rls_update(matrix, covariance, action, observed, forgetting):
    """Uniform-forgetting RLS in a fixed input basis, with bounded covariance."""
    with old.exact_matmul():
        m, p = matrix.double(), covariance.double()
        a, y = action.double(), observed.double()
        pa = (p @ a.unsqueeze(-1)).squeeze(-1)
        denominator = forgetting+(a*pa).sum(-1)
        gain = pa/denominator.clamp_min(1e-12)[:, None]
        innovation = y-(m @ a.unsqueeze(-1)).squeeze(-1)
        m = m+innovation[:, :, None]*gain[:, None, :]
        p = (p-gain[:, :, None]*pa[:, None, :])/forgetting
        p = (p+p.transpose(1, 2))*.5
        values, vectors = torch.linalg.eigh(p)
        p = (vectors*values.clamp(1e-6, 1e6)[:, None, :]) @ vectors.transpose(1, 2)
    return m.to(matrix.dtype), p.to(covariance.dtype), innovation.to(matrix.dtype)


def record(before, after, original, actual, accepted, **extra):
    if not torch.isfinite(before).all() or not torch.isfinite(after).all() or not torch.isfinite(actual).all():
        raise FloatingPointError('Nonfinite control-output candidate')
    return dict(accepted=float(accepted.sum()), before=float(before.sum()),
                after=float(after.sum()), shift=float(norm(actual-original).sum()), **extra)


def semantic_qp(p, theta, radius):
    z, q = p.z, p.q
    def probabilities(x):
        return torch.stack((p.target_probability(p.end(x)), p.target_probability(p.end(x, 'c'))), 1)
    baseline, jacobian = finite_jacobian(probabilities, z, q, p.eps)
    bounds = torch.stack((.2*(.6-baseline[:, 0]).clamp_min(0),
                          torch.full_like(baseline[:, 0], -theta)), 1)
    coefficient, feasible = minimal_two_constraints(jacobian, bounds)
    candidate = z+cap(combine(q, coefficient, z), radius)
    actual = probabilities(candidate)
    accepted = (feasible & (actual[:, 0] > baseline[:, 0])
                & (actual[:, 1] >= baseline[:, 1]-theta-1e-6))
    result = choose(accepted, candidate, z)
    initial = (.6-baseline[:, 0]).clamp_min(0).square()
    final = torch.where(accepted, (.6-actual[:, 0]).clamp_min(0).square(), initial)
    with old.exact_matmul():
        actual_coeff = (q.transpose(1, 2) @ flat(candidate-z).unsqueeze(-1)).squeeze(-1)
        predicted = (jacobian @ actual_coeff.unsqueeze(-1)).squeeze(-1)
    return result, record(initial, final, z, result, accepted,
                          qp_infeasible=float((~feasible).sum()),
                          qp_clipped_progress_shortfall=float((predicted[:, 0] < bounds[:, 0]-1e-7).sum()))


def recursive_semantic(p, theta, radius):
    z, memory = p.z, p.context.history
    if 'fixed_q' not in memory:
        memory['fixed_q'] = p.q.detach()
    q = memory['fixed_q']
    fn = lambda x: p.target_probability(p.end(x))[:, None]
    initial = fn(z)
    if 'response' not in memory:
        _, matrix = finite_jacobian(fn, z, q, p.eps, initial)
        covariance = torch.eye(2, device=z.device, dtype=z.dtype).expand(len(z), -1, -1).clone()*100.
    else:
        matrix = memory['response'].clone()
        covariance = memory['covariance'].clone()
        index = memory['response_events'] % 2
        value = fn(z+p.eps*q[:, :, index].reshape_as(z))
        observed = (value-initial)/p.eps.flatten(1)
        matrix[:, :, index] = theta*matrix[:, :, index]+(1-theta)*observed
    residual = (initial-.6).clamp_max(0)
    coefficient = ridge_step(residual, matrix)
    candidate = z+cap(combine(q, coefficient, z), radius)
    measured = fn(candidate)
    with old.exact_matmul():
        actual_action = (q.transpose(1, 2) @ flat(candidate-z).unsqueeze(-1)).squeeze(-1)
    updated, covariance, innovation = rls_update(matrix, covariance, actual_action, measured-initial, theta)
    memory.update(response=updated.detach(), covariance=covariance.detach(),
                  response_events=memory.get('response_events', 0)+1)
    accepted = measured[:, 0] > initial[:, 0]
    result = choose(accepted, candidate, z)
    before = residual[:, 0].square()
    after = torch.where(accepted, (measured[:, 0]-.6).clamp_max(0).square(), before)
    return result, record(before, after, z, result, accepted,
                          response_prediction_error=float(innovation.square().sum()))


def select_finite(losses, lengths, reduction):
    """Exhaustive minimum norm among sufficient decreases; zero is column 0."""
    assert losses.shape == lengths.shape and 0 < reduction < 1
    assert torch.isfinite(losses).all() and torch.isfinite(lengths).all()
    assert (losses >= 0).all() and (lengths >= 0).all() and (lengths[:, 0] == 0).all()
    eligible = losses <= (1-reduction)*losses[:, :1]
    # argmin uses the first occurrence: index 0 wins a satisfied zero-loss tie.
    shortest = lengths.masked_fill(~eligible, torch.inf).argmin(1)
    best = losses.argmin(1)
    selected = torch.where(eligible.any(1), shortest, best)
    chosen = losses.gather(1, selected[:, None]).squeeze(1)
    assert (chosen <= losses[:, 0]).all()
    return selected, eligible.any(1)


def verified_semantic(p, theta, radius):
    z = p.z
    fn = lambda x: p.target_probability(p.end(x))[:, None]
    baseline, jacobian = finite_jacobian(fn, z, p.q, p.eps)
    coefficient = ridge_step((baseline-.6).clamp_max(0), jacobian)
    candidate = z+cap(combine(p.q, coefficient, z), radius)
    proposed = fn(candidate)
    fine = Probe(p.rt, z, p.t, p.context, steps=16)
    fine_before = fine.target_probability(fine.end(z))[:, None]
    fine_after = fine.target_probability(fine.end(candidate))[:, None]
    bound = (fine_before-baseline).abs()+(fine_after-proposed).abs()
    coarse_gain = proposed-baseline
    # The second check protects the strict conclusion from floating point ties.
    accepted = ((coarse_gain > theta*bound) & (fine_after > fine_before)).squeeze(1)
    result = choose(accepted, candidate, z)
    before = (.6-fine_before[:, 0]).clamp_min(0).square()
    after = torch.where(accepted, (.6-fine_after[:, 0]).clamp_min(0).square(), before)
    p.context.history.setdefault('priority_trace', []).append(dict(
        kind='verified', time=p.t, accepted=accepted.detach().cpu().numpy(),
        coarse_before=baseline[:, 0].detach().cpu().numpy(),
        coarse_after=proposed[:, 0].detach().cpu().numpy(),
        fine_before=fine_before[:, 0].detach().cpu().numpy(),
        fine_after=fine_after[:, 0].detach().cpu().numpy(),
        discrepancy_bound=bound[:, 0].detach().cpu().numpy()))
    return result, record(before, after, z, result, accepted,
        resolution_rejections=float((~accepted).sum()),
        verified_fine_gain=float(torch.where(accepted[:, None], fine_after-fine_before, 0.).sum()))


def finite_semantic(p, theta, radius):
    z = p.z
    candidates = [z]
    for factor in (.5, 1.):
        for index in range(2):
            delta = factor*radius*p.q[:, :, index].reshape_as(z)
            candidates.extend((z+delta, z-delta))
    losses = torch.stack([(.6-p.target_probability(p.end(x))).clamp_min(0).square()
                          for x in candidates], 1)
    lengths = torch.stack([norm(x-z).flatten() for x in candidates], 1)
    selected, attained = select_finite(losses, lengths, theta)
    batch = torch.arange(len(z), device=z.device)
    result = torch.stack(candidates, 1)[batch, selected]
    after = losses[batch, selected]
    p.context.history.setdefault('priority_trace', []).append(dict(
        kind='finite', time=p.t, losses=losses.detach().cpu().numpy(),
        lengths=lengths.detach().cpu().numpy(), selected=selected.detach().cpu().numpy(),
        attained=attained.detach().cpu().numpy()))
    return result, record(losses[:, 0], after, z, result, after < losses[:, 0],
        finite_target_attained=float(attained.sum()), finite_zero_selected=float((selected == 0).sum()))


def two_stage_mpc(p, theta, radius):
    z, q = p.z, p.q
    target = p.end(z, 'c').detach()
    scale = energy(p.end(z)-target).sqrt().clamp_min(1e-5)[:, None]
    def output(action):
        first = combine(q, action[:, :2], z)
        second = combine(q, action[:, 2:], z)
        terminal = p.planned_null(z+first, second)
        return pack(flat(terminal-target)/scale, math.sqrt(theta)*flat(second)/scale)
    zero = torch.zeros((len(z), 4), device=z.device, dtype=z.dtype)
    residual = output(zero)
    columns = []
    for index in range(4):
        probe = zero.clone()
        probe[:, index] = p.eps.flatten()
        columns.append((output(probe)-residual)/p.eps.flatten(1))
    coefficient = ridge_step(residual, torch.stack(columns, -1))
    for offset in (0, 2):
        # Q is orthonormal; keep each planned write within the same physical cap.
        length = coefficient[:, offset:offset+2].norm(dim=1, keepdim=True).clamp_min(EPS)
        coefficient[:, offset:offset+2] *= (radius.flatten(1)/length).clamp_max(1.)
    ending = output(coefficient)
    before, after = energy(residual), energy(ending)
    accepted = after < before
    first = combine(q, coefficient[:, :2], z)
    result = choose(accepted, z+first, z)
    final = torch.where(accepted, after, before)
    # The second write is a virtual plan. Only the first write changes sampling.
    executed_error = energy(flat(p.end(result)-target)/scale)
    return result, record(before, final, z, result, accepted,
                          mpc_executed_terminal_error=float(executed_error.sum()),
                          mpc_planned_second_shift=float(coefficient[:, 2:].norm(dim=1).sum()))


def calibrate(rt, z, t, config, context, amount, *, disabled=False):
    empty = dict(accepted=0., before=0., after=0., shift=0.)
    key, theta = config['key'], config['theta']
    if disabled or amount == 0:
        return z, empty
    memory = context.history
    if key == 'semantic_handoff' and 'latched' in memory and bool(memory['latched'].all()):
        return z, empty
    horizon = theta if key == 'rolling_root' else (.25 if key == 'rolling_invariance' else None)
    p = Probe(rt, z, t, context, horizon=horizon)
    radius = (4/64)*amount*norm(p.d)
    if key in ('terminal_root', 'terminal_inverse', 'noise_tangent_root', 'root_cotangent', 'full_cycle'):
        radius = radius*theta
    if key == 'semantic_guard_qp':
        return semantic_qp(p, theta, radius)
    if key == 'recursive_semantic_response':
        return recursive_semantic(p, theta, radius)
    if key == 'verified_semantic_step':
        return verified_semantic(p, theta, radius)
    if key == 'finite_semantic_control':
        return finite_semantic(p, theta, radius)
    if key == 'two_stage_mpc':
        return two_stage_mpc(p, theta, radius)
    dimensions = int(theta) if key == 'richer_input_subspace' else 2
    q = ortho_basis(p.d, context.random, dimensions,
                    tangent=p.noise if key == 'noise_tangent_root' else None)
    p.q = q
    active = torch.ones(len(z), device=z.device, dtype=torch.bool)
    if key == 'semantic_handoff' and 'latched' in memory:
        active &= ~memory['latched']
    if key in ('event_triggered_root', 'golden_handoff'):
        current = energy(p.gap(z)).sqrt()/max(1-t, .01)
        if 'trigger_scale' not in memory:
            memory['trigger_scale'] = current.detach().clamp_min(1e-6)
        normalized = current/memory['trigger_scale']
        if key == 'event_triggered_root':
            active &= normalized > theta
        else:
            latched = memory.get('latched', torch.zeros_like(active))
            latched = latched & (normalized <= 1.5*theta)
            memory['latched'] = latched
            active &= ~latched
        if not bool(active.any()):
            # Golden handoff still checks its release threshold above.
            return z, dict(empty, skipped_controls=float(len(z)))
    if key == 'adaptive_trust':
        radius = radius*memory.get('trust_radius', torch.ones_like(radius))
    fn = objective(p, key, theta)
    residual = fn(z)
    before = energy(residual)
    model_error = 0.
    cycle_defect = 0.

    def step(start, r, *, target=None, mode=None):
        _, jacobian = finite_jacobian(fn, start, q, p.eps, r)
        target = r if target is None else target
        metric = None
        if key == 'future_input_metric':
            terminal = lambda x: flat(p.end(x))/p.scale()
            _, ju = finite_jacobian(terminal, start, q, p.eps)
            with old.exact_matmul():
                metric = theta*(ju.double().transpose(1, 2) @ ju.double())
        if mode in ('gradient', 'sliding'):
            desired = target if mode == 'gradient' else torch.tanh(target/theta)
            with old.exact_matmul():
                coefficient = -(jacobian.transpose(1, 2) @ desired.unsqueeze(-1)).squeeze(-1)
            direction = combine(q, coefficient, start)
            delta = direction/norm(direction).clamp_min(EPS)*radius
        else:
            coefficient = ridge_step(target, jacobian, metric=metric)
            delta = cap(combine(q, coefficient, start), radius)
        candidate = start+delta
        if key == 'noise_tangent_root':
            estimated = p.noise+(candidate-z)/max(p.b, .01)
            candidate = p.t*p.m+p.b*estimated/norm(estimated).clamp_min(EPS)*norm(p.noise)
        ending = fn(candidate)
        with old.exact_matmul():
            actual_coefficient = (q.transpose(1, 2) @ flat(candidate-start).unsqueeze(-1)).squeeze(-1)
            predicted = r+(jacobian @ actual_coefficient.unsqueeze(-1)).squeeze(-1)
        return candidate, ending, predicted

    if key == 'full_cycle':
        translated = p.reverse_null(p.end(z, 'c'))-z
        defect = p.reverse_null(p.end(z))-z
        cycle_defect = float(norm(defect).sum())
        candidate = z+cap(translated, radius)
        ending = fn(candidate)
        predicted = ending
    else:
        target = residual
        if key in ('root_integral', 'root_derivative'):
            dt = t-memory.get('last_t', t)
            if key == 'root_integral':
                integral = .8*memory.get('integral', torch.zeros_like(residual))+dt*residual
                memory['integral'] = integral.detach()
                target = residual+theta*integral
            else:
                derivative = ((residual-memory['last_residual'])/max(dt, 1e-6)
                              if 'last_residual' in memory else torch.zeros_like(residual))
                derivative = .5*memory.get('derivative', torch.zeros_like(derivative))+.5*derivative
                memory['derivative'] = derivative.detach()
                target = residual+theta*derivative
            memory.update(last_t=t, last_residual=residual.detach())
        mode = 'gradient' if key == 'root_cotangent' else ('sliding' if key == 'root_sliding' else None)
        candidate, ending, predicted = step(z, residual, target=target, mode=mode)
        if key in ('output_disturbance', 'anderson_root'):
            first_ok = energy(ending) < before
            first = choose(first_ok, candidate, z)
            first_r = torch.where(first_ok[:, None], ending, residual)
            if key == 'output_disturbance':
                defect = ending-predicted
                model_error = float(defect.square().sum())
                second_target = first_r+theta*defect
            else:
                second_target = first_r
            second, second_r, second_pred = step(first, first_r, target=second_target)
            second_ok = energy(second_r) < energy(first_r)
            result2 = choose(second_ok, second, first)
            result2_r = torch.where(second_ok[:, None], second_r, first_r)
            candidate, ending, predicted = result2, result2_r, second_pred
            if key == 'anderson_root':
                f0, f1 = first-z, result2-first
                difference = f1-f0
                ridge = theta*(norm(f0).square()+norm(f1).square())*.5
                gamma = (inner(difference, f1)/(norm(difference).square()+ridge).clamp_min(EPS)).clamp(-1., 2.)
                accelerated = z+cap(result2-gamma*(result2-first)-z, 2*radius)
                accelerated_r = fn(accelerated)
                acceleration_ok = energy(accelerated_r) < energy(result2_r)
                candidate = choose(acceleration_ok, accelerated, result2)
                ending = torch.where(acceleration_ok[:, None], accelerated_r, result2_r)
    after = energy(ending)
    accepted = (after < before) & active
    result = choose(accepted, candidate, z)
    actual = torch.where(accepted, after, before)
    if not torch.isfinite(result).all() or not torch.isfinite(actual).all():
        raise FloatingPointError(f'{config["arm"]}: nonfinite calibration')
    if key == 'adaptive_trust':
        predicted_reduction = before-energy(predicted)
        ratio = (before-after)/predicted_reduction.clamp_min(1e-8)
        relative = (before-actual)/before.clamp_min(1e-8)
        factor = torch.where((ratio < .25) | ~accepted, .5,
                             torch.where((ratio > .75) & (relative < theta), 1.4, .9))
        memory['trust_radius'] = (memory.get('trust_radius', torch.ones_like(radius))*factor[:, None, None, None]).clamp(.2, 2.).detach()
        model_error = float((ending-predicted).square().sum())
    if key == 'semantic_handoff':
        probability = p.target_probability(p.end(result))
        memory['latched'] = memory.get('latched', torch.zeros_like(active)) | (probability >= theta)
    if key == 'golden_handoff':
        normalized = energy(p.gap(result)).sqrt()/max(1-t, .01)/memory['trigger_scale']
        confirmations = memory.get('confirmations', torch.zeros(len(z), device=z.device, dtype=torch.int64))
        confirmations = torch.where(normalized < theta, confirmations+1, torch.zeros_like(confirmations))
        memory['confirmations'] = confirmations
        memory['latched'] = memory.get('latched', torch.zeros_like(active)) | (confirmations >= 2)
    return result, record(before, actual, z, result, accepted, response_prediction_error=model_error,
                          cycle_defect_norm=cycle_defect,
                          latched_samples=float(memory.get('latched', torch.zeros_like(active)).sum()),
                          skipped_controls=float((~active).sum()))


def main_field(rt, z, t, amount, context, latched=None):
    if latched is not None and bool(latched.all()):
        before = rt.labels
        rt.labels = torch.full_like(before, 100)
        try:
            return rt.field(z, z.new_tensor(t), 'full')
        finally:
            rt.labels = before
    config = dict(key='native_cfg', source='cfg', theta=0.)
    if latched is None or not bool(latched.any()):
        return old.evaluate(rt, z, t, config, amount, context)[0]
    query = old.Queries(rt, z, t, 'native_cfg', context)
    uncond = query.null()[0]
    guided = query.s+amount*(query.s-uncond)
    return choose(latched, uncond, guided)


@torch.inference_mode()
def sample(rt, noise, labels, config, *, zero=False, disable_calibration=False):
    if config['key'] not in SPECIAL:
        return seven.sample(rt, noise, labels, config, zero=zero,
                            disable_calibration=disable_calibration)
    seed = int(array_sha(noise.detach().cpu().numpy())[:15], 16)
    is_sde = config['solver'] == 'sde_tail64'
    if zero and not is_sde:
        return old.sample(rt, noise, labels, dict(config, solver='heun64'), batch_seed=seed, zero=True)
    rt.labels = labels
    z = noise.clone()
    counts = rt.counts.copy()
    semantic = getattr(rt, 'pasted_semantic', None)
    semantic_start = (semantic.decoded_images, semantic.classified_images) if semantic else (0, 0)
    main_context = old.StepContext()
    history = {}
    generator = torch.Generator(device=z.device).manual_seed((seed+130363) % (2**63-1))
    brownian = torch.randn((64, *z.shape), generator=generator, device=z.device) if is_sde else None
    events = config.get('parameters', {}).get('events', catalog.EVENT_STEPS)
    extra = dict(calibration_events=0, accepted_calibrations=0., calibration_loss_before_sum=0.,
                 calibration_loss_after_sum=0., calibration_shift_norm_sum=0.,
                 calibration_full_calls=0, response_prediction_error=0., qp_infeasible=0.,
                 qp_clipped_progress_shortfall=0., mpc_executed_terminal_error=0.,
                 mpc_planned_second_shift=0., cycle_defect_norm=0., latched_samples=0.,
                 skipped_controls=0., resolution_rejections=0., verified_fine_gain=0.,
                 finite_target_attained=0., finite_zero_selected=0.)
    names = {'accepted': 'accepted_calibrations', 'before': 'calibration_loss_before_sum',
             'after': 'calibration_loss_after_sum', 'shift': 'calibration_shift_norm_sum'}
    for k in range(64):
        t, h = k/64, 1/64
        amount = 0. if zero else old.amount_at(config, t)
        if amount and k in events and not disable_calibration:
            event = event_context(noise, k, history)
            before = rt.counts['full']
            z, stats = calibrate(rt, z, t, config, event, amount)
            extra['calibration_events'] += 1
            extra['calibration_full_calls'] += rt.counts['full']-before
            for key, value in stats.items():
                extra[names.get(key, key)] += value
        latched = history.get('latched') if not disable_calibration else None
        first = main_field(rt, z, t, amount, main_context, latched)
        if is_sde and k >= 8:
            z = z+h*(2*first-z/t)+math.sqrt(2*(1-t)/t*h)*brownian[k]
        else:
            second = main_field(rt, z+h*first, t+h, amount, main_context, latched)
            z = z+(h/2)*(first+second)
        if not torch.isfinite(z).all() or z.abs().max() > 1e6:
            raise FloatingPointError(f'{config["arm"]}: invalid trajectory at {k}')
    full = rt.counts['full']-counts['full']
    prefix = rt.counts['prefix']-counts['prefix']
    semantic = getattr(rt, 'pasted_semantic', None)
    extra['auxiliary_decoder_images'] = semantic.decoded_images-semantic_start[0] if semantic else 0
    extra['auxiliary_classifier_images'] = semantic.classified_images-semantic_start[1] if semantic else 0
    traces = history.get('priority_trace', [])
    if traces:
        # Persist every tested finite loss and every coarse/fine gate, not only
        # the selected score, in each immutable generation batch.
        extra['priority_trace_kind'] = traces[0]['kind']
        for key in traces[0]:
            if key != 'kind':
                extra['priority_trace_'+key] = np.stack([trace[key] for trace in traces])
    return z, dict(full_calls=full, prefix_calls=prefix,
                   auxiliary_full_calls=max(0, full-(128 if is_sde else 224)),
                   diagnostic_queries=0, strang_active_steps=0,
                   diagnostics=np.zeros((len(z), 5), dtype=np.float64), **extra)


@torch.inference_mode()
def limiting_checks(rt, noise, labels):
    rt.labels = labels
    original = rt.field(noise, noise.new_tensor(.25), 'full')
    context = old.StepContext()
    value = main_field(rt, noise, .25, 0., context)
    assert torch.equal(value, original)
    for method in catalog.IDEAS:
        cfg = next(row for row in configurations() if row['idea_id'] == method['id'])
        unchanged, _ = calibrate(rt, noise, .25, cfg, None, cfg['strength'], disabled=True)
        assert torch.equal(unchanged, noise)
        unchanged, _ = calibrate(rt, noise, .25, cfg, None, 0.)
        assert torch.equal(unchanged, noise)
    assert rt.labels is labels
    return dict(zero_field_exact=True, disabled_state_writes_exact=len(catalog.IDEAS),
                zero_amount_state_writes_exact=len(catalog.IDEAS))
