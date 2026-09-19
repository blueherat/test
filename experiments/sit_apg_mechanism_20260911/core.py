"""Inference-only APG extensions with counted probes and inspectable decisions."""
from __future__ import annotations

import math
import numpy as np
import torch
from experiments.sit_apg_mechanism_20260911 import catalog, common as ops
from experiments.sit_control_50_20260910 import core as previous
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.lifting_scale_sweep_20260909 import array_sha

FAMILIES = {f'i{m["id"]}_{m["key"]}':m for m in catalog.IDEAS}
configurations = catalog.configurations
norm, inner, projection, cap = ops.norm, ops.inner, ops.projection, ops.cap
EPS = 1e-10


def install():
    pass


def choose(mask, yes, no):
    return torch.where(mask.reshape(-1, *([1]*(yes.ndim-1))), yes, no)


def take(values, index):
    """Per-sample choice from a list of equal-shaped tensors."""
    stacked = torch.stack(values, 1)
    return stacked[torch.arange(len(index), device=index.device), index]


def guided(rt, x, t, labels, amount, *, eta=0., memory=None, beta=0., radius=None,
           released=None):
    if released is not None and bool(released.all()):
        return ops.field(rt, x, t, labels, null=True), None
    if amount == 0 and released is None:
        return ops.field(rt, x, t, labels), None
    c, u = ops.pair(rt, x, t, labels)
    d = c-u
    b = 1-t
    # Published ordering: clean residual -> clean history -> fixed-radius clip
    # -> projection onto clean conditional prediction -> conversion to velocity.
    if radius is not None:
        clean_buffer = b*d+beta*(torch.zeros_like(d) if memory is None else memory)
        bounded = cap(clean_buffer, torch.full_like(norm(d), radius))
        perpendicular = bounded-projection(bounded, x+b*c)
        modified = (perpendicular+eta*(bounded-perpendicular))/max(b, 1e-8)
    else:
        clean_buffer = None
        parallel = projection(d, x+b*c)
        modified = d-(1-eta)*parallel
    result = c+amount*modified
    if released is not None:
        result = choose(released, u, result)
    return result, clean_buffer


def step(rt, x, t, h, labels, amount, *, eta=0., memory=None, beta=0., radius=None,
         released=None, substeps=1):
    state = x
    left_buffer = None
    for j in range(substeps):
        left, width = t+j*h/substeps, h/substeps
        first, buffer = guided(rt, state, left, labels, amount, eta=eta,
            memory=memory, beta=beta, radius=radius, released=released)
        second, _ = guided(rt, state+width*first, left+width, labels, amount, eta=eta,
            memory=memory, beta=beta, radius=radius, released=released)
        state = state+(width/2)*(first+second)
        if j == 0:
            left_buffer = buffer
    return state, left_buffer


def block(rt, x, k, labels, amount, *, eta=0., released=None, steps=8):
    state = x
    for j in range(steps):
        t = (k+j)/64
        state, _ = step(rt, state, t, 1/64, labels, amount if t < .75 else 0.,
                        eta=eta, released=released)
    return state


def terminal(rt, x, t, labels, steps=16):
    return ops.readout(rt, ops.future(rt, x, t, labels, steps), labels)


def embedded(rt, x, t, labels, amount):
    first, _ = guided(rt, x, t, labels, amount, eta=1.)
    euler = x+first/64
    second, _ = guided(rt, euler, t+1/64, labels, amount, eta=1.)
    heun = x+(first+second)/128
    return (norm(heun-euler)/(norm(first)/64).clamp_min(EPS)).flatten()


def skew_probe(rt, x, t, labels, amount, generator):
    dimension = x[0].numel()
    epsilon = .001*norm(x).clamp_min(64.)
    values = []
    disagreement = None
    def field(state):
        return guided(rt, state, t, labels, amount, eta=1.)[0]
    for index in range(2):
        p, q = ops.random_unit(x, generator), ops.random_unit(x, generator)
        jp = (field(x+epsilon*p)-field(x-epsilon*p))/(2*epsilon)
        jq = (field(x+epsilon*q)-field(x-epsilon*q))/(2*epsilon)
        anti = (inner(p, jq)-inner(q, jp)).flatten()
        values.append(dimension**2*anti.double().square())
        if index == 0:
            wide = (field(x+2*epsilon*q)-field(x-2*epsilon*q))/(4*epsilon)
            disagreement = (norm(jq-wide)/norm(wide).clamp_min(EPS)).flatten()
    # A=(J-J^T)/2. Frobenius/sqrt(d) is a typical response scale, not ||A||_op.
    severity = torch.stack(values).mean(0).sqrt()/(2*64*math.sqrt(dimension))
    return severity.to(x.dtype), disagreement


def plane(rt, x, t, labels, amount):
    fields = ops.bundle(rt, x, t, labels, amount)
    first, second = ops.unit(fields['orthogonal']), ops.unit(fields['parallel'])
    radius = (2/64)*amount*norm(fields['gap'])
    basis = torch.stack((first.flatten(1), second.flatten(1)), -1)
    raw = torch.stack((norm(fields['orthogonal']).flatten(), norm(fields['parallel']).flatten()), -1)
    raw = raw/norm(fields['gap']).flatten()[:, None].clamp_min(EPS)
    return basis, radius, raw


def combine(basis, coefficient, radius, like):
    with old.exact_matmul():
        return (basis @ coefficient.unsqueeze(-1)).squeeze(-1).reshape_as(like)*radius


def coefficient_candidates(x, raw):
    # A finite, explicit disk grid plus exact raw/projection/zero proposals.
    angles = torch.arange(64, device=x.device, dtype=x.dtype)*(2*math.pi/64)
    circle = torch.stack((angles.cos(), angles.sin()), -1)
    lattice = torch.cat([r*circle for r in (.25, .5, .75, 1.)], 0)
    lattice = lattice[None].expand(len(x), -1, -1)
    projected = torch.stack((raw[:, 0], torch.zeros_like(raw[:, 0])), -1)
    return torch.cat((x.new_zeros((len(x), 1, 2)), raw[:, None], projected[:, None], lattice), 1)


def moment_write(rt, x, t, labels, amount, config):
    basis, radius, raw = plane(rt, x, t, labels, amount)
    base = terminal(rt, x, t, labels)
    responses, moment_responses = [], []
    for index in range(2):
        direction = basis[:, :, index].reshape_as(x)
        plus = terminal(rt, x+.5*radius*direction, t, labels)
        minus = terminal(rt, x-.5*radius*direction, t, labels)
        # Derivatives with respect to dimensionless coefficients (actual radius included).
        responses.append(plus['q']-minus['q'])
        moment_responses.append(plus['moments']-minus['moments'])
    semantic = torch.stack(responses, -1)
    nuisance = torch.stack(moment_responses, -1)
    # A per-state common scale makes the penalty insensitive to uniformly
    # changing the units of the selected 8 moment outputs.
    nuisance = nuisance/nuisance.square().sum((1, 2)).sqrt()[:, None, None].clamp_min(1e-8)
    candidates = coefficient_candidates(x, raw)
    with old.exact_matmul():
        predicted_q = (candidates*semantic[:, None]).sum(-1)
        predicted_moments = torch.einsum('bmd,bnd->bnm', nuisance, candidates)
    objective = (candidates-raw[:, None]).square().sum(-1)+config['theta']*predicted_moments.square().sum(-1)
    key = config['key']
    if key == 'semantic_direction':
        objective = -predicted_q
        eligible = torch.ones_like(objective, dtype=torch.bool)
    elif key == 'moment_without_semantic':
        eligible = torch.ones_like(objective, dtype=torch.bool)
    else:
        bound = .5*(raw*semantic).sum(-1).clamp_min(0.)
        eligible = predicted_q >= bound[:, None]-1e-10
    scores = torch.where(eligible, objective, torch.inf)
    feasible = eligible.any(-1)
    index = scores.argmin(-1)
    index = torch.where(feasible, index, torch.zeros_like(index))
    coefficient = candidates[torch.arange(len(x), device=x.device), index]
    delta = combine(basis, coefficient, radius, x)
    after = terminal(rt, x+delta, t, labels)
    accepted = feasible & (after['q'] > base['q'])
    raw_after = terminal(rt, x+combine(basis, raw, radius, x), t, labels)
    moment_change = (after['moments']-base['moments']).square().mean(1).sqrt()
    raw_moment_change = (raw_after['moments']-base['moments']).square().mean(1).sqrt()
    semantic_floor = base['q']+.5*(raw_after['q']-base['q']).clamp_min(0.)
    if key == 'future_moment_apg':
        accepted &= (after['q'] >= semantic_floor) & (moment_change <= raw_moment_change+1e-8)
    elif key == 'moment_without_semantic':
        accepted = feasible & (coefficient.square().sum(-1) > 0) & (moment_change <= raw_moment_change+1e-8)
    result = choose(accepted, x+delta, x)
    trace = dict(q_before=base['q'], q_proposed=after['q'], accepted=accepted,
        feasible=feasible, coefficient=coefficient, raw_coefficient=raw,
        selected=index, predicted_q=predicted_q, objectives=objective, eligible=eligible,
        moment_change=moment_change, raw_moment_change=raw_moment_change,
        q_raw=raw_after['q'], semantic_floor=semantic_floor,
        shift_norm=norm(delta).flatten(), radius=radius.flatten())
    return result, trace


def eta_block(rt, x, k, labels, amount, config):
    states = [block(rt, x, k, labels, amount, eta=eta) for eta in (0., .5, 1.)]
    ends = [terminal(rt, state, (k+8)/64, labels) for state in states]
    q = torch.stack([r['q'] for r in ends], -1)
    moments = torch.stack([(r['moments']-ends[0]['moments']).square().mean(1) for r in ends], -1)
    scale = ends[0]['moments'].square().mean(1).clamp_min(1e-6)
    penalty = moments/scale[:, None]
    scores = q-config['theta']*penalty
    eligible = q >= q[:, :1]
    index = torch.where(eligible, scores, -torch.inf).argmax(-1)
    return take(states, index), dict(q_all=q, penalty_all=penalty, scores=scores,
        eligible=eligible, selected=index, selected_eta=index.to(x.dtype)*.5,
        q_before=q[:, 0], q_selected=q.gather(1, index[:, None]).flatten())


def release_block(rt, x, k, labels, amount, config, released):
    # Both alternatives cover the same physical block, followed by the same
    # null grid. This removes the passive suffix-grid defect from the comparison.
    null = block(rt, x, k, labels, amount,
                 released=torch.ones_like(released))
    keep = block(rt, x, k, labels, amount, released=released)
    q_null = terminal(rt, null, (k+8)/64, labels)['q']
    q_keep = terminal(rt, keep, (k+8)/64, labels)['q']
    decision = (q_null >= .5) & (q_keep-q_null <= config['theta'])
    new = released | decision
    return choose(new, null, keep), new, dict(q_null=q_null, q_keep=q_keep,
        marginal_value=q_keep-q_null, released_before=released, released_after=new,
        newly_released=(~released & new), adequate_null=(q_null >= .5))


def horizon_proposal(rt, x, t, labels, amount, horizon, direction, radius):
    def residual(state):
        h = min(horizon, 1-t)
        c = ops.heun(rt, state, t, h, labels, kind='cfg', substeps=8)
        u = ops.heun(rt, state, t, h, labels, kind='null', substeps=8)
        return c-u
    before = residual(x)
    plus = residual(x+.5*radius*direction)
    minus = residual(x-.5*radius*direction)
    jacobian = plus-minus
    coefficient = (-inner(before, jacobian)/inner(jacobian, jacobian).clamp_min(EPS)).clamp(-1., 1.)
    delta = coefficient*radius*direction
    return x+delta, coefficient.flatten(), norm(delta).flatten()


def horizon_write(rt, x, t, labels, amount, config):
    gap = ops.bundle(rt, x, t, labels, amount)['gap']
    radius, direction = (2/64)*amount*norm(gap), ops.unit(gap)
    horizons = (config['theta'],) if config['key'] == 'fixed_horizon_verified' else (.125, .25, .5)
    states, coefficients, lengths = [x], [x.new_zeros(len(x))], [x.new_zeros(len(x))]
    for horizon in horizons:
        proposal, coefficient, length = horizon_proposal(rt, x, t, labels, amount, horizon, direction, radius)
        states.append(proposal);coefficients.append(coefficient);lengths.append(length)
    q = torch.stack([terminal(rt, state, t, labels)['q'] for state in states], -1)
    threshold = 0. if config['key'] == 'fixed_horizon_verified' else config['theta']
    eligible = q > q[:, :1]+threshold
    eligible[:, 0] = True
    index = torch.where(eligible, q, -torch.inf).argmax(-1)
    return take(states, index), dict(q_all=q, selected=index, eligible=eligible,
        coefficients=torch.stack(coefficients, -1), lengths=torch.stack(lengths, -1),
        q_before=q[:, 0], q_selected=q.gather(1, index[:, None]).flatten())


def limiting_checks(rt, noise, labels):
    rt.labels = labels
    original = next(c for c in configurations() if c['family'] == 'strong')
    baseline, _ = previous.sample(rt, noise, labels, original)
    for method in catalog.IDEAS:
        config = next(c for c in configurations() if c['idea_id'] == method['id'])
        zero, stats = sample(rt, noise, labels, config, zero=True)
        assert torch.equal(baseline, zero) and stats.get('extension_events', 0) == 0
    return dict(zero_guidance_exact=True, labels_restored=rt.labels is labels)


@torch.inference_mode()
def sample(rt, noise, labels, config, *, zero=False, disable_calibration=False):
    if config['parameters'].get('inherited_exact'):
        return previous.sample(rt, noise, labels, config, zero=zero,
                               disable_calibration=disable_calibration)
    if zero:
        # The zero-control limit uses exactly the existing baseline arithmetic.
        baseline = next(c for c in configurations() if c['family'] == 'strong')
        return previous.sample(rt, noise, labels, baseline, zero=True)
    rt.labels = labels
    counts = rt.counts.copy()
    reader = getattr(rt, 'pasted_semantic', None)
    start = (reader.decoded_images, reader.classified_images) if reader else (0, 0)
    key, amount = config['key'], config['strength']
    state = noise.clone()
    traces, memory, prior_erk = [], None, None
    main_eta = 1. if config['parameters'].get('main') == 'cfg' else 0.
    if key == 'projection_apg':
        main_eta = config['theta']
    nsteps = int(config['theta']) if key == 'uniform_refinement' else 64
    released = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    refinements = torch.zeros_like(released)
    shrink = torch.ones(len(state), device=state.device, dtype=state.dtype)
    seed = int(array_sha(noise.cpu().numpy())[:15], 16)
    generator = torch.Generator(device=noise.device).manual_seed((seed+20260911) % (2**63-1))
    refined_steps, event_calls, selected_nonzero = 0, 0, 0
    k = 0
    while k < nsteps:
        t, h = k/nsteps, 1/nsteps
        active_amount = amount if t < config['cutoff'] else 0.
        event = None
        before_queries = rt.counts['full']
        if not disable_calibration and active_amount:
            if key in ('curl_refine', 'curl_gain_shrink', 'embedded_refinement') and k in catalog.EVENTS:
                if key == 'embedded_refinement':
                    severity = embedded(rt, state, t, labels, active_amount)
                    refinements = severity > config['theta']
                    event = dict(severity=severity, refined=refinements)
                else:
                    severity, fd_error = skew_probe(rt, state, t, labels, active_amount, generator)
                    fallback_error = embedded(rt, state, t, labels, active_amount)
                    refinements = torch.where(fd_error <= .1, severity > config['theta'], fallback_error > .003)
                    shrink = torch.minimum(torch.ones_like(severity), config['theta']/severity.clamp_min(EPS))
                    shrink = torch.where(fd_error <= .1, shrink, torch.ones_like(shrink))
                    event = dict(severity=severity, fd_error=fd_error, fallback_error=fallback_error,
                                 refined=refinements, shrink=shrink)
            elif key in ('future_moment_apg', 'semantic_direction', 'moment_without_semantic') and k in catalog.WRITE_EVENTS:
                state, event = moment_write(rt, state, t, labels, active_amount, config)
            elif key == 'future_eta' and k in catalog.WRITE_EVENTS:
                state, event = eta_block(rt, state, k, labels, active_amount, config)
            elif key == 'marginal_release' and k in catalog.EVENTS:
                state, released, event = release_block(rt, state, k, labels, active_amount, config, released)
            elif key == 'probability_null_release' and k in catalog.EVENTS:
                probability = terminal(rt, state, t, labels)['q']
                earlier = released
                released = released | (probability >= config['theta'])
                event = dict(q_null=probability, released_before=earlier, released_after=released)
            elif key in ('terminal_horizon', 'fixed_horizon_verified') and k in catalog.WRITE_EVENTS:
                state, event = horizon_write(rt, state, t, labels, active_amount, config)
        event_calls += rt.counts['full']-before_queries
        if event is not None:
            traces.append(dict(step=k, **{name:ops.numpy(value) for name, value in event.items()}))
            selected_nonzero += int(event.get('accepted', event.get('selected', state.new_zeros(len(state)))).count_nonzero())
            if key in ('future_eta', 'marginal_release'):
                k += 8
                if not torch.isfinite(state).all() or state.abs().max() > 1e6:
                    raise FloatingPointError(f'{config["arm"]}: invalid selected block at {k}')
                continue
        if key == 'fixed_null_release' and t >= config['theta'] and not disable_calibration:
            released = torch.ones_like(released)
        release_mask = released if bool(released.any()) else None
        if key == 'clean_apg':
            state, next_memory = step(rt, state, t, h, labels, active_amount,
                memory=memory, beta=config['theta'], radius=config['parameters']['radius'])
            if active_amount:
                memory = next_memory
        elif key == 'erk_guid' and not disable_calibration:
            first, _ = guided(rt, state, t, labels, active_amount, eta=1.)
            correction = torch.zeros_like(state)
            if prior_erk is not None and k != 48:
                previous_euler, previous_field = prior_erk
                drift_difference = first-previous_field
                rho = norm(drift_difference)/norm(state-previous_euler).clamp_min(1e-8)
                correction = (config['theta']*h*rho).square()*projection(first, drift_difference)
            euler = state+h*first
            second, _ = guided(rt, euler, t+h, labels, active_amount, eta=1.)
            # In forward noise-to-data time, the paper's -h*g_sigma is +h*g_t.
            state = state+(h/2)*(first+second)+h*correction
            prior_erk = (euler, second)
        elif key in ('curl_refine', 'embedded_refinement') and active_amount and not disable_calibration and bool(refinements.any()):
            fine, _ = step(rt, state, t, h, labels, active_amount, eta=1., substeps=2)
            coarse = state
            if not bool(refinements.all()):
                coarse, _ = step(rt, state, t, h, labels, active_amount, eta=1.)
            state = choose(refinements, fine, coarse)
            refined_steps += int(refinements.sum())
        elif key == 'curl_gain_shrink' and active_amount and not disable_calibration:
            # Keep the conditional field and attenuate only the extra CFG term.
            first = ops.bundle(rt, state, t, labels, active_amount)
            velocity = first['conditional']+active_amount*shrink[:, None, None, None]*first['gap']
            second = ops.bundle(rt, state+h*velocity, t+h, labels, active_amount)
            velocity2 = second['conditional']+active_amount*shrink[:, None, None, None]*second['gap']
            state = state+(h/2)*(velocity+velocity2)
        else:
            state, _ = step(rt, state, t, h, labels, active_amount, eta=main_eta, released=release_mask)
        if not torch.isfinite(state).all() or state.abs().max() > 1e6:
            raise FloatingPointError(f'{config["arm"]}: invalid trajectory at {k}')
        k += 1
    rt.labels = labels
    full, prefix = rt.counts['full']-counts['full'], rt.counts['prefix']-counts['prefix']
    reader = getattr(rt, 'pasted_semantic', None)
    stats = dict(full_calls=full, prefix_calls=prefix, auxiliary_full_calls=max(0, full-224),
        diagnostic_queries=0, strang_active_steps=0, diagnostics=np.zeros((len(state), 5)),
        extension_events=len(traces), extension_event_full_calls=event_calls,
        refined_sample_steps=refined_steps, selected_nonzero=selected_nonzero,
        final_released_samples=int(released.sum()),
        auxiliary_decoder_images=reader.decoded_images-start[0] if reader else 0,
        auxiliary_classifier_images=reader.classified_images-start[1] if reader else 0)
    if traces:
        stats['extension_trace_kind'] = key
        for name in traces[0]:
            stats['extension_trace_'+name] = np.stack([value[name] for value in traces])
    return state, stats
