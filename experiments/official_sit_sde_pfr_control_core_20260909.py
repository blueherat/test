"""Explicit Brownian coupling and a parity-checked copy of the author's SDE.

The copy changes only the source of Gaussian increments and optionally adds
the fixed-state PFR velocity response before the existing score/CFG conversion.
The author's final deterministic step and floating-point operation order stay
explicit so that the zero-response control can be checked bit for bit.
"""
from __future__ import annotations
import hashlib
from collections.abc import Callable

import torch

from experiments.audit_official_sit_pfr_interface import prefix


def grid(steps: int) -> torch.Tensor:
    return torch.linspace(1., .04, steps, dtype=torch.float64)


def counts(steps: int, pfr: bool) -> dict:
    times = grid(steps).tolist()
    multiplicities = [2 if t <= .7 else 1 for t in times]
    full = sum(multiplicities)
    partial = sum(n for t, n in zip(times, multiplicities) if t > .5) if pfr else 0
    return dict(steps=steps, full_per_image=full, prefix_per_image=partial,
                block_evaluations_per_image=28 * full + 8 * partial,
                gaussian_calls=steps - 1)


def bridge_seed(batch: int) -> int:
    raw = hashlib.sha256(f'official_sit_sde_pfr_control_20260909/bridge/batch/{batch}'.encode()).digest()
    return int.from_bytes(raw[:8], 'big') % (2 ** 63)


def bridge_draw_count(source_grid: torch.Tensor, target_grid: torch.Tensor) -> int:
    a = (1 - source_grid).tolist()
    return sum(not any(abs(u-v) < 1e-14 for v in a) for u in (1-target_grid)[1:].tolist())


def couple_normals(base: torch.Tensor, source_grid: torch.Tensor,
                   target_grid: torch.Tensor, draw: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Conditionally fill target times, preserving the entire source Brownian path.

    base[i] are standard normals for source interval i.  Target-only times in
    each source interval are drawn successively from the conditional Brownian
    bridge.  Reusing the newest left endpoint is essential when several target
    times lie inside one source interval.  The source normals are never changed.
    """
    assert base.dtype == torch.float64 and len(base) == len(source_grid)-1
    assert source_grid.device.type == target_grid.device.type == 'cpu'
    a, b = (1-source_grid).tolist(), (1-target_grid).tolist()
    assert a[0] == b[0] == 0. and a[-1] == b[-1]
    assert all(x < y for x, y in zip(a[:-1], a[1:]))
    assert all(x < y for x, y in zip(b[:-1], b[1:]))
    shape = (len(base),) + (1,) * (base.ndim-1)
    dt = (source_grid[:-1]-source_grid[1:]).sqrt().to(base.device).reshape(shape)
    cumulative = torch.cat([torch.zeros_like(base[:1]), (base*dt).cumsum(0)])
    source_interval = 0
    left_t, left_w = a[0], cumulative[0]
    previous_w = cumulative[0]
    normals = []
    for j, u in enumerate(b[1:]):
        while source_interval+1 < len(a)-1 and u > a[source_interval+1] + 1e-14:
            source_interval += 1
            left_t, left_w = a[source_interval], cumulative[source_interval]
        right_t, right_w = a[source_interval+1], cumulative[source_interval+1]
        if abs(u-right_t) < 1e-14:
            next_w = right_w
        elif abs(u-a[source_interval]) < 1e-14:
            next_w = cumulative[source_interval]
        else:
            assert left_t < u < right_t, (left_t, u, right_t)
            eta = draw()
            assert eta.shape == base.shape[1:] and eta.dtype == base.dtype and eta.device == base.device
            weight = (u-left_t)/(right_t-left_t)
            variance = (u-left_t)*(right_t-u)/(right_t-left_t)
            next_w = left_w + weight*(right_w-left_w) + variance**.5*eta
        normals.append((next_w-previous_w)/(b[j+1]-b[j])**.5)
        left_t, left_w, previous_w = u, next_w, next_w
    return torch.stack(normals)


def make_noises(template: torch.Tensor, base_seed: int, extra_seed: int):
    assert template.dtype == torch.float64
    source = torch.Generator(device=template.device).manual_seed(base_seed)
    base = torch.stack([torch.randn(template.shape, dtype=template.dtype, device=template.device,
                                    generator=source) for _ in range(249)])
    source_end_state = source.get_state()
    extra = torch.Generator(device=template.device).manual_seed(extra_seed)
    calls = 0
    def draw():
        nonlocal calls
        calls += 1
        return torch.randn(template.shape, dtype=template.dtype, device=template.device, generator=extra)
    refined = couple_normals(base, grid(250), grid(281), draw)
    assert calls == bridge_draw_count(grid(250), grid(281)) == 279
    return base, refined, source_end_state


class CountedModel:
    def __init__(self, model):
        self.model = model
        self.batch_calls = self.sample_calls = self.prefix_batch_calls = self.prefix_sample_calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, x, *args, **kwargs):
        self.batch_calls += 1
        self.sample_calls += len(x)
        return self.model(x, *args, **kwargs)

    def future_base(self, x, t, y):
        self.prefix_batch_calls += 1
        self.prefix_sample_calls += len(x)
        return prefix(self.model, x, t, y)


@torch.inference_mode()
def sample_sde(model, latents, y, normals, *, steps=250, pfr=False, response_scale=1.):
    """Author SDE arithmetic, explicit normals, CFG1.35/.7 and IG1.4/all.

    `response_scale=0` with pfr=True still executes all queries and is a
    computation-path parity control, not a new quality arm.
    """
    from samplers import get_score_from_velocity, compute_diffusion
    assert normals.shape == (steps-1,) + latents.shape
    assert normals.dtype == torch.float64 and normals.device == latents.device
    assert steps >= 2
    cfg_scale, sg_scale = 1.35, 1.4
    y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype
    t_steps = torch.cat([grid(steps), torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    def velocity(model_input, time_input, y_cur, t_cur):
        full, base = model(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), y=y_cur)[:2]
        current = base.to(torch.float64) + sg_scale*(full.to(torch.float64)-base.to(torch.float64))
        if pfr and float(t_cur) > .5:
            future = max(.5, float(t_cur)-1/32)
            tf = torch.full((len(model_input),), future, device=device, dtype=_dtype)
            future_base = model.future_base(model_input.to(dtype=_dtype), tf, y_cur)
            current = current + (sg_scale*response_scale)*(base.to(torch.float64)-future_base.to(torch.float64))
        return current

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next-t_cur
        x_cur = x_next
        if t_cur <= .7:
            model_input = torch.cat([x_cur]*2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input, y_cur = x_cur, y
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64)*t_cur
        diffusion = compute_diffusion(t_cur)
        deps = normals[i]*torch.sqrt(torch.abs(dt))
        v_cur = velocity(model_input, time_input, y_cur, t_cur)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type='linear')
        d_cur = v_cur - .5*diffusion*s_cur
        if t_cur <= .7:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale*(d_cur_cond-d_cur_uncond)
        x_next = x_cur + d_cur*dt + torch.sqrt(diffusion)*deps

    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next-t_cur
    x_cur = x_next
    model_input = torch.cat([x_cur]*2, dim=0)
    y_cur = torch.cat([y, y_null], dim=0)
    time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64)*t_cur
    v_cur = velocity(model_input, time_input, y_cur, t_cur)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type='linear')
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - .5*diffusion*s_cur
    d_cur_cond, d_cur_uncond = d_cur.chunk(2)
    d_cur = d_cur_uncond + cfg_scale*(d_cur_cond-d_cur_uncond)
    mean_x = x_cur + dt*d_cur
    return mean_x
