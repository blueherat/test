#!/usr/bin/env python3
"""
Prediction-target error-geometry toy suite v10 (reviewed + optimized version).

This file consolidates the original v10 mechanism study with the later
visual-paradox additions motivated by the archived v4 scatter grids.

Key semantics
-------------
* x_t = (1-t) x + t eps, v = eps - x.
* All primary guidance is RECURSIVE: it is recomputed from the CURRENT state
  and applied at every active solver step.
* x-v and x-eps are treated symmetrically where possible.
* Three action parameterizations are kept separate:
    raw:       x + gamma * gap
    abs-norm:  x + eta * gap / RMS(gap)       [legacy v4 xv_rms_eta*]
    rel-norm:  x + rho * RMS(x) * gap / RMS(gap)
* The known-data geometry is decomposed as
      R^D = T_curve + N_ridge-within-surface + N_ambient.
* Visual evidence is first-class: an optional PDF atlas shows target paths,
  legacy normalized extrapolations, raw ridge cross-sections, and paired
  endpoint displacements.

The script imports run_prediction_target_extrapolation_toy_v4.py from the same
repository.  It does not overwrite v4 outputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-10


# -----------------------------------------------------------------------------
# Import v4
# -----------------------------------------------------------------------------

def load_v4():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "run_prediction_target_extrapolation_toy_v4.py",
        here.parent / "experiments" / "run_prediction_target_extrapolation_toy_v4.py",
        Path("/mnt/data/run_prediction_target_extrapolation_toy_v4.py"),
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("prediction_target_toy_v4_base", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not locate run_prediction_target_extrapolation_toy_v4.py. "
        "Place this v10 file beside the repository v4 script."
    )


v4 = load_v4()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*values: int) -> int:
    x = 2166136261
    for value in values:
        x ^= int(value) & 0xFFFFFFFF
        x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def tag_float(x: float) -> str:
    s = f"{float(x):.5g}"
    return s.replace("-", "m").replace(".", "p")


def row_rms(x: torch.Tensor) -> torch.Tensor:
    return x.double().square().mean(dim=1).sqrt().float()


def row_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.double(), b.double(), dim=1, eps=EPS).float()


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


# -----------------------------------------------------------------------------
# Spiral centerline geometry
# -----------------------------------------------------------------------------

def spiral_center(s: torch.Tensor) -> torch.Tensor:
    theta = 4.0 * math.pi * s
    radius = 0.15 + 0.85 * s
    return 1.6 * torch.stack(
        [radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1
    )


def spiral_tangent_2d(s: torch.Tensor) -> torch.Tensor:
    theta = 4.0 * math.pi * s
    radius = 0.15 + 0.85 * s
    dr = 0.85
    dtheta = 4.0 * math.pi
    dx = 1.6 * (dr * torch.cos(theta) - radius * dtheta * torch.sin(theta))
    dy = 1.6 * (dr * torch.sin(theta) + radius * dtheta * torch.cos(theta))
    out = torch.stack([dx, dy], dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class SpiralLocator:
    def __init__(self, points: int, device: torch.device):
        self.device = device
        self.s = torch.linspace(0.0, 1.0, int(points), device=device)
        self.u = spiral_center(self.s)
        self.tangent = spiral_tangent_2d(self.s)

    @torch.inference_mode()
    def nearest(
        self, u: torch.Tensor, chunk: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, distances = [], []
        for start in range(0, len(u), chunk):
            part = u[start : start + chunk]
            d2 = (part[:, None, :] - self.u[None, :, :]).square().sum(dim=2)
            val, idx = d2.min(dim=1)
            ids.append(idx)
            distances.append(val.sqrt())
        idx = torch.cat(ids)
        dist = torch.cat(distances)
        return self.s[idx], self.u[idx], self.tangent[idx], dist


# -----------------------------------------------------------------------------
# Embedding wrapper with curvature-energy matching
# -----------------------------------------------------------------------------

class ControlledEmbedding:
    def __init__(
        self,
        D: int,
        *,
        curvature: float,
        frequency_scale: float,
        seed: int,
        device: torch.device,
        scale_mode: str,
        energy_match: bool,
        calibration_samples: int,
        data_jitter: float,
    ):
        self.base = v4.CurvedEmbedding(
            D,
            curvature=curvature,
            frequency_scale=frequency_scale,
            seed=seed,
            device=device,
            scale_mode=scale_mode,
        )
        self.D = int(D)
        self.curvature = float(curvature)
        self.frequency_scale = float(frequency_scale)
        self.seed = int(seed)
        self.device = device
        self.scale_mode = str(scale_mode)
        self.energy_match = bool(energy_match)
        self.extra_scale = 1.0

        if energy_match and curvature != 0.0:
            ref = v4.CurvedEmbedding(
                D,
                curvature=0.0,
                frequency_scale=frequency_scale,
                seed=seed,
                device=device,
                scale_mode=scale_mode,
            )
            g = torch.Generator(device=device.type)
            g.manual_seed(stable_seed(seed, 99173))
            u = v4.sample_spiral_2d(
                calibration_samples, device=device, jitter=data_jitter, generator=g
            )
            with torch.inference_mode():
                ref_rms = row_rms(ref.embed(u)).mean()
                cur_rms = row_rms(self.base.embed(u)).mean()
            self.extra_scale = float((ref_rms / cur_rms.clamp_min(1e-12)).cpu())

    def embed(self, u: torch.Tensor) -> torch.Tensor:
        return self.extra_scale * self.base.embed(u)

    def decode_intrinsic(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.decode_intrinsic(x / self.extra_scale)

    def manifold_consistency_rms(self, x: torch.Tensor) -> torch.Tensor:
        u = self.decode_intrinsic(x)
        recon = self.embed(u)
        return row_rms(x - recon)

    def jacobian(self, u: torch.Tensor) -> torch.Tensor:
        return self.extra_scale * self.base.jacobian(u)

    def tangent_basis(self, u: torch.Tensor) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.jacobian(u).double(), mode="reduced")
        return q.float()


@torch.inference_mode()
def observability_diagnostics(
    emb: ControlledEmbedding, *, samples: int, jitter: float, seed: int
) -> dict[str, float]:
    g = torch.Generator(device=emb.device.type)
    g.manual_seed(seed)
    u = v4.sample_spiral_2d(samples, device=emb.device, jitter=jitter, generator=g)
    x = emb.embed(u)
    j = emb.jacobian(u).double()
    gram = torch.einsum("bdi,bdj->bij", j, j)
    eig = torch.linalg.eigvalsh(gram)
    return {
        "clean_rms": float(row_rms(x).mean().cpu()),
        "jtj_lambda_min_mean": float(eig[:, 0].mean().cpu()),
        "jtj_lambda_max_mean": float(eig[:, -1].mean().cpu()),
        "jtj_trace_mean": float(eig.sum(dim=1).mean().cpu()),
        "jtj_condition_median": float(
            (eig[:, -1] / eig[:, 0].clamp_min(1e-12)).median().cpu()
        ),
        "embedding_extra_scale": emb.extra_scale,
    }


# -----------------------------------------------------------------------------
# True 3-way geometry
# -----------------------------------------------------------------------------

@dataclass
class GeometryFrame:
    nearest_s: torch.Tensor
    nearest_u: torch.Tensor
    tangent_2d: torch.Tensor
    normal_2d: torch.Tensor
    curve_unit: torch.Tensor
    ridge_unit: torch.Tensor
    ridge_scale: torch.Tensor
    ridge_distance_2d: torch.Tensor
    signed_ridge_coordinate: torch.Tensor


@dataclass
class GapComponents:
    curve_tangent: torch.Tensor
    ridge_normal: torch.Tensor
    ambient_normal: torch.Tensor
    nearest_s: torch.Tensor
    ridge_distance_2d: torch.Tensor


@torch.inference_mode()
def build_geometry_frame(
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    anchor_x: torch.Tensor,
) -> GeometryFrame:
    """Build one local orthonormal frame and reuse it for all vector splits.

    The old v10 implementation recomputed nearest-point search, Jacobian and QR
    for every vector (real gap, selected component, action).  This function
    constructs the same local tangent plane once.  curve_unit and ridge_unit
    form an orthonormal basis of that plane; ambient is their orthogonal
    complement.
    """
    u_hat = emb.decode_intrinsic(anchor_x)
    s, u0, t2, ridge_dist = locator.nearest(u_hat)
    n2 = torch.stack([-t2[:, 1], t2[:, 0]], dim=1)
    j = emb.jacobian(u0).float()

    curve_vec = torch.einsum("bdi,bi->bd", j, t2)
    curve_norm = curve_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    curve_unit = curve_vec / curve_norm

    # Map the intrinsic ridge-normal direction into ambient space and remove
    # any metric-induced component along the embedded curve tangent.
    ridge_vec = torch.einsum("bdi,bi->bd", j, n2)
    ridge_vec = ridge_vec - curve_unit * (curve_unit * ridge_vec).sum(dim=1, keepdim=True)
    ridge_scale = ridge_vec.norm(dim=1).clamp_min(1e-12)
    ridge_unit = ridge_vec / ridge_scale[:, None]

    signed_ridge = ((u_hat - u0) * n2).sum(dim=1)
    return GeometryFrame(
        nearest_s=s,
        nearest_u=u0,
        tangent_2d=t2,
        normal_2d=n2,
        curve_unit=curve_unit,
        ridge_unit=ridge_unit,
        ridge_scale=ridge_scale,
        ridge_distance_2d=ridge_dist,
        signed_ridge_coordinate=signed_ridge,
    )


def split_with_geometry_frame(
    frame: GeometryFrame, vec: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    curve = frame.curve_unit * (frame.curve_unit * vec).sum(dim=1, keepdim=True)
    ridge = frame.ridge_unit * (frame.ridge_unit * vec).sum(dim=1, keepdim=True)
    ambient = vec - curve - ridge
    return curve, ridge, ambient


@torch.inference_mode()
def decompose_gap_three_way(
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    anchor_x: torch.Tensor,
    gap: torch.Tensor,
) -> GapComponents:
    frame = build_geometry_frame(emb, locator, anchor_x)
    curve, ridge, ambient = split_with_geometry_frame(frame, gap)
    return GapComponents(curve, ridge, ambient, frame.nearest_s, frame.ridge_distance_2d)


def _contraction_stats(
    frame: GeometryFrame, vec: torch.Tensor
) -> tuple[float, float, float]:
    """Return slope, inward fraction and intrinsic-equivalent ridge action RMS.

    If n is the signed intrinsic ridge coordinate and a is the action expressed
    in intrinsic ridge-coordinate units, a ~= -k n has k>0 for a contractive
    field.  The conversion by ridge_scale removes local embedding stretch.
    """
    signed_ambient = (vec * frame.ridge_unit).sum(dim=1)
    signed_intrinsic = signed_ambient / frame.ridge_scale
    n = frame.signed_ridge_coordinate
    mask = n.abs() > 1e-5
    if int(mask.sum()) == 0:
        return float("nan"), float("nan"), float(row_rms(signed_intrinsic[:, None]).mean().cpu())
    nm = n[mask].double()
    am = signed_intrinsic[mask].double()
    slope = -float((nm * am).sum().cpu() / nm.square().sum().clamp_min(EPS).cpu())
    inward = float(((nm * am) < 0).double().mean().cpu())
    action_rms = float(am.square().mean().sqrt().cpu())
    return slope, inward, action_rms


# -----------------------------------------------------------------------------
# Endpoint metrics
# -----------------------------------------------------------------------------

@dataclass
class EndpointFeatures:
    intrinsic: np.ndarray
    arc_s: np.ndarray
    signed_ridge: np.ndarray
    ridge_distance: np.ndarray
    ambient_rms: np.ndarray
    radius: np.ndarray


@dataclass
class ReferenceEndpointGeometry:
    arc_s: np.ndarray
    signed_ridge: np.ndarray
    ridge_distance: np.ndarray
    radius: np.ndarray
    hist_counts: np.ndarray


@torch.inference_mode()
def endpoint_features(
    ambient: np.ndarray,
    *,
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    device: torch.device,
) -> EndpointFeatures:
    x = torch.from_numpy(ambient).to(device)
    u = emb.decode_intrinsic(x)
    s, u0, t2, d = locator.nearest(u)
    n2 = torch.stack([-t2[:, 1], t2[:, 0]], dim=1)
    signed = ((u - u0) * n2).sum(dim=1)
    ambient_rms = emb.manifold_consistency_rms(x)
    radius = u.norm(dim=1)
    return EndpointFeatures(
        intrinsic=u.cpu().numpy(),
        arc_s=s.cpu().numpy(),
        signed_ridge=signed.cpu().numpy(),
        ridge_distance=d.cpu().numpy(),
        ambient_rms=ambient_rms.cpu().numpy(),
        radius=radius.cpu().numpy(),
    )


@torch.inference_mode()
def build_reference_endpoint_geometry(
    reference_u: np.ndarray,
    *,
    locator: SpiralLocator,
    bins: int,
    device: torch.device,
) -> ReferenceEndpointGeometry:
    ref = torch.from_numpy(reference_u).to(device)
    rs, ru0, rt2, rd = locator.nearest(ref)
    rn2 = torch.stack([-rt2[:, 1], rt2[:, 0]], dim=1)
    ref_signed = ((ref - ru0) * rn2).sum(dim=1)
    ref_radius = ref.norm(dim=1)
    rs_np = rs.cpu().numpy()
    return ReferenceEndpointGeometry(
        arc_s=rs_np,
        signed_ridge=ref_signed.cpu().numpy(),
        ridge_distance=rd.cpu().numpy(),
        radius=ref_radius.cpu().numpy(),
        hist_counts=np.histogram(rs_np, bins=bins, range=(0, 1), density=False)[0],
    )


def _wasserstein_1d_sorted(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    # np.quantile handles unequal sample counts while preserving a deterministic
    # empirical W1 approximation.
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def conditional_ridge_w1(
    gen_s: np.ndarray,
    gen_signed: np.ndarray,
    ref_s: np.ndarray,
    ref_signed: np.ndarray,
    *,
    bins: int,
    min_count: int,
) -> tuple[float, float, int]:
    """Match ridge thickness conditional on arc position, separating precision from coverage."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    vals, weights = [], []
    for bi in range(bins):
        if bi == bins - 1:
            gm = (gen_s >= edges[bi]) & (gen_s <= edges[bi + 1])
            rm = (ref_s >= edges[bi]) & (ref_s <= edges[bi + 1])
        else:
            gm = (gen_s >= edges[bi]) & (gen_s < edges[bi + 1])
            rm = (ref_s >= edges[bi]) & (ref_s < edges[bi + 1])
        if int(gm.sum()) < min_count or int(rm.sum()) < min_count:
            continue
        vals.append(_wasserstein_1d_sorted(gen_signed[gm], ref_signed[rm]))
        weights.append(int(rm.sum()))
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.average(arr, weights=w)), float(np.max(arr)), len(vals)


def ridge_metrics_from_features(
    feat: EndpointFeatures,
    ref: ReferenceEndpointGeometry,
    *,
    bins: int,
    conditional_bins: int,
    conditional_min_count: int,
) -> dict[str, float]:
    hg = np.histogram(feat.arc_s, bins=bins, range=(0, 1), density=False)[0]
    hr = ref.hist_counts
    pg = hg / max(hg.sum(), 1)
    pr = hr / max(hr.sum(), 1)
    tv = 0.5 * np.abs(pg - pr).sum()
    m = 0.5 * (pg + pr)
    kl_g = np.sum(np.where(pg > 0, pg * np.log((pg + 1e-12) / (m + 1e-12)), 0))
    kl_r = np.sum(np.where(pr > 0, pr * np.log((pr + 1e-12) / (m + 1e-12)), 0))
    js = 0.5 * (kl_g + kl_r)

    ref_signed_std = float(np.std(ref.signed_ridge))
    gen_signed_std = float(np.std(feat.signed_ridge))
    ref_radius_mean = float(np.mean(ref.radius))
    cond_mean, cond_worst, cond_valid = conditional_ridge_w1(
        feat.arc_s,
        feat.signed_ridge,
        ref.arc_s,
        ref.signed_ridge,
        bins=conditional_bins,
        min_count=conditional_min_count,
    )
    return {
        "ridge_distance_mean": float(np.mean(feat.ridge_distance)),
        "ridge_distance_median": float(np.median(feat.ridge_distance)),
        "ridge_distance_q90": float(np.quantile(feat.ridge_distance, 0.90)),
        "reference_ridge_distance_mean": float(np.mean(ref.ridge_distance)),
        "ridge_distance_excess": float(np.mean(feat.ridge_distance) - np.mean(ref.ridge_distance)),
        "signed_ridge_mean": float(np.mean(feat.signed_ridge)),
        "signed_ridge_std": gen_signed_std,
        "signed_ridge_q10": float(np.quantile(feat.signed_ridge, 0.10)),
        "signed_ridge_q50": float(np.quantile(feat.signed_ridge, 0.50)),
        "signed_ridge_q90": float(np.quantile(feat.signed_ridge, 0.90)),
        "reference_signed_ridge_mean": float(np.mean(ref.signed_ridge)),
        "reference_signed_ridge_std": ref_signed_std,
        "ridge_width_ratio": (
            gen_signed_std / ref_signed_std if ref_signed_std > 1e-5 else float("nan")
        ),
        "ridge_signed_bias_in_ref_std": float(
            abs(np.mean(feat.signed_ridge) - np.mean(ref.signed_ridge))
            / max(ref_signed_std, 1e-8)
        ),
        "conditional_ridge_w1": cond_mean,
        "conditional_ridge_w1_worst_bin": cond_worst,
        "conditional_ridge_valid_bins": float(cond_valid),
        "arc_coverage_w1": _wasserstein_1d_sorted(feat.arc_s, ref.arc_s),
        "arc_hist_tv": float(tv),
        "arc_hist_js": float(js),
        "arc_empty_bin_fraction": float(np.mean(hg == 0)),
        # Keep the legacy name for backwards compatibility; this is a
        # decode->re-embed consistency residual, not a strict nearest-surface distance.
        "ambient_surface_rms": float(np.mean(feat.ambient_rms)),
        "ambient_surface_rms_q50": float(np.quantile(feat.ambient_rms, 0.50)),
        "ambient_surface_rms_q90": float(np.quantile(feat.ambient_rms, 0.90)),
        "ambient_surface_rms_q99": float(np.quantile(feat.ambient_rms, 0.99)),
        "intrinsic_radius_mean": float(np.mean(feat.radius)),
        "intrinsic_radius_std": float(np.std(feat.radius)),
        "intrinsic_radius_q10": float(np.quantile(feat.radius, 0.10)),
        "intrinsic_radius_q90": float(np.quantile(feat.radius, 0.90)),
        "reference_intrinsic_radius_mean": ref_radius_mean,
        "intrinsic_radius_mean_ratio": (
            float(np.mean(feat.radius)) / ref_radius_mean if ref_radius_mean > 1e-8 else float("nan")
        ),
    }


# -----------------------------------------------------------------------------
# Decoder-free high-D curve geometry
# -----------------------------------------------------------------------------

@dataclass
class EmbeddedCurveFeatures:
    nearest_s: np.ndarray
    curve_distance_rms: np.ndarray
    signed_ridge_rms_units: np.ndarray
    ambient_rms: np.ndarray
    tangent_abs_rms_units: np.ndarray


class EmbeddedCurveLocator:
    """Nearest-point search on the known embedded 1-D centerline.

    A coarse global search prevents wrong spiral turns; two cheap local
    refinements provide near-continuous resolution without the O(B*G*D) cost of
    a very dense high-D grid.  This never uses decode_intrinsic().
    """

    def __init__(
        self,
        emb: ControlledEmbedding,
        *,
        coarse_points: int,
        refine_points: int,
        refine_rounds: int,
    ) -> None:
        self.emb = emb
        self.device = emb.device
        self.coarse_points = max(int(coarse_points), 32)
        self.refine_points = max(int(refine_points), 3)
        if self.refine_points % 2 == 0:
            self.refine_points += 1
        self.refine_rounds = max(int(refine_rounds), 0)
        self.s = torch.linspace(0.0, 1.0, self.coarse_points, device=self.device)
        self.u = spiral_center(self.s)
        self.x = emb.embed(self.u)
        self.x2 = self.x.double().square().sum(dim=1)

    @torch.inference_mode()
    def nearest(self, z: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        best_s_parts = []
        for start in range(0, len(z), chunk):
            zz = z[start : start + chunk]
            z2 = zz.double().square().sum(dim=1, keepdim=True)
            # Exact distances to the coarse grid using a GEMM instead of
            # materializing [B,G,D].
            d2 = z2 + self.x2[None, :] - 2.0 * (zz.double() @ self.x.double().T)
            idx = d2.argmin(dim=1)
            s_best = self.s[idx]
            half = 1.0 / max(self.coarse_points - 1, 1)
            for _ in range(self.refine_rounds):
                offsets = torch.linspace(-half, half, self.refine_points, device=self.device)
                cand_s = (s_best[:, None] + offsets[None, :]).clamp(0.0, 1.0)
                cand_u = spiral_center(cand_s.reshape(-1))
                cand_x = self.emb.embed(cand_u).reshape(len(zz), self.refine_points, self.emb.D)
                local_d2 = (zz[:, None, :] - cand_x).double().square().sum(dim=2)
                j = local_d2.argmin(dim=1)
                s_best = cand_s[torch.arange(len(zz), device=self.device), j]
                half = 2.0 * half / max(self.refine_points - 1, 2)
            best_s_parts.append(s_best)
        s_best = torch.cat(best_s_parts)
        nearest_x = self.emb.embed(spiral_center(s_best))
        return s_best, nearest_x


@torch.inference_mode()
def embedded_curve_features(
    ambient: np.ndarray,
    *,
    emb: ControlledEmbedding,
    curve_locator: EmbeddedCurveLocator,
    device: torch.device,
) -> EmbeddedCurveFeatures:
    z = torch.from_numpy(ambient).to(device)
    s, x0 = curve_locator.nearest(z)
    u0 = spiral_center(s)
    t2 = spiral_tangent_2d(s)
    n2 = torch.stack([-t2[:, 1], t2[:, 0]], dim=1)
    j = emb.jacobian(u0).float()
    curve_vec = torch.einsum("bdi,bi->bd", j, t2)
    curve_unit = curve_vec / curve_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ridge_vec = torch.einsum("bdi,bi->bd", j, n2)
    ridge_vec = ridge_vec - curve_unit * (curve_unit * ridge_vec).sum(dim=1, keepdim=True)
    ridge_unit = ridge_vec / ridge_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)

    r = z - x0
    curve_coeff = (r * curve_unit).sum(dim=1)
    ridge_coeff = (r * ridge_unit).sum(dim=1)
    curve_part = curve_unit * curve_coeff[:, None]
    ridge_part = ridge_unit * ridge_coeff[:, None]
    ambient_part = r - curve_part - ridge_part
    sqrt_d = math.sqrt(float(emb.D))
    return EmbeddedCurveFeatures(
        nearest_s=s.cpu().numpy(),
        curve_distance_rms=row_rms(r).cpu().numpy(),
        signed_ridge_rms_units=(ridge_coeff / sqrt_d).cpu().numpy(),
        ambient_rms=row_rms(ambient_part).cpu().numpy(),
        tangent_abs_rms_units=(curve_coeff.abs() / sqrt_d).cpu().numpy(),
    )


def embedded_curve_metrics(
    feat: EmbeddedCurveFeatures,
    ref: EmbeddedCurveFeatures,
) -> dict[str, float]:
    ref_ridge_std = float(np.std(ref.signed_ridge_rms_units))
    gen_ridge_std = float(np.std(feat.signed_ridge_rms_units))
    return {
        "curveD_distance_rms_mean": float(np.mean(feat.curve_distance_rms)),
        "curveD_distance_rms_q90": float(np.quantile(feat.curve_distance_rms, 0.90)),
        "curveD_reference_distance_rms_mean": float(np.mean(ref.curve_distance_rms)),
        "curveD_signed_ridge_std": gen_ridge_std,
        "curveD_reference_signed_ridge_std": ref_ridge_std,
        "curveD_ridge_width_ratio": (
            gen_ridge_std / ref_ridge_std if ref_ridge_std > 1e-8 else float("nan")
        ),
        "curveD_ambient_rms_mean": float(np.mean(feat.ambient_rms)),
        "curveD_ambient_rms_q90": float(np.quantile(feat.ambient_rms, 0.90)),
        "curveD_reference_ambient_rms_mean": float(np.mean(ref.ambient_rms)),
        "curveD_tangent_abs_rms_mean": float(np.mean(feat.tangent_abs_rms_units)),
        "curveD_arc_w1": _wasserstein_1d_sorted(feat.nearest_s, ref.nearest_s),
    }


@torch.inference_mode()
def local_surface_gauss_newton_rms(
    ambient: np.ndarray,
    *,
    emb: ControlledEmbedding,
    device: torch.device,
    iterations: int,
    damping: float,
) -> np.ndarray:
    """Local nearest-surface audit initialized by the analytic readout.

    This is deliberately labeled an audit rather than a guaranteed global
    nearest point.  The objective is only 2-D and each Gauss-Newton step is
    accepted by a simple per-sample backtracking test, so the residual cannot
    increase from the decode->re-embed starting point.
    """
    z = torch.from_numpy(ambient).to(device)
    u = emb.decode_intrinsic(z).clone()
    current = row_rms(z - emb.embed(u)).double()
    eye = torch.eye(2, device=device, dtype=torch.float64)[None]
    for _ in range(max(int(iterations), 0)):
        x = emb.embed(u)
        r = (z - x).double()
        j = emb.jacobian(u).double()
        gram = torch.einsum("bdi,bdj->bij", j, j) + float(damping) * eye
        rhs = torch.einsum("bdi,bd->bi", j, r)
        step = torch.linalg.solve(gram, rhs[:, :, None]).squeeze(2).float()
        # Vectorized backtracking; keep the best non-increasing candidate.
        best_u = u
        best = current
        scale = torch.ones(len(u), device=device)
        for _ls in range(5):
            cand_u = u + scale[:, None] * step
            cand = row_rms(z - emb.embed(cand_u)).double()
            improve = cand < best
            best_u = torch.where(improve[:, None], cand_u, best_u)
            best = torch.where(improve, cand, best)
            scale = scale * 0.5
        u = best_u
        current = best
    return current.float().cpu().numpy()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@dataclass
class HeadStop:
    best_score: float = float("inf")
    best_step: int = 0
    stale: int = 0
    stopped: bool = False
    best_state: dict | None = None


def _serialize_head_stops(status: dict[str, HeadStop]) -> dict[str, dict]:
    return {
        target: {
            "best_score": value.best_score,
            "best_step": value.best_step,
            "stale": value.stale,
            "stopped": value.stopped,
            "best_state": value.best_state,
        }
        for target, value in status.items()
    }


def _restore_head_stops(payload: dict[str, dict]) -> dict[str, HeadStop]:
    return {
        target: HeadStop(
            best_score=float(value["best_score"]),
            best_step=int(value["best_step"]),
            stale=int(value["stale"]),
            stopped=bool(value["stopped"]),
            best_state=value.get("best_state"),
        )
        for target, value in payload.items()
    }


def _validate_training_resume_config(
    stored: dict,
    current: dict,
    *,
    checkpoint_step: int,
) -> None:
    ignored = {"planned_steps"}
    keys = sorted((set(stored) | set(current)) - ignored)
    mismatches = [
        f"{key}: checkpoint={stored.get(key)!r}, current={current.get(key)!r}"
        for key in keys
        if stored.get(key) != current.get(key)
    ]
    planned_steps = int(current["planned_steps"])
    if planned_steps < checkpoint_step:
        mismatches.append(
            f"planned_steps={planned_steps} is below checkpoint step {checkpoint_step}"
        )
    # Extending a constant-LR fixed run is exactly equivalent to having selected
    # the longer horizon initially.  A cosine schedule depends on its horizon,
    # so changing that horizon after training has started is not equivalent.
    if (
        stored.get("planned_steps") != current.get("planned_steps")
        and current.get("actual_scheduler") != "constant"
    ):
        mismatches.append(
            "planned_steps may only change when the actual scheduler is constant"
        )
    if mismatches:
        raise ValueError("incompatible v10 training resume:\n  " + "\n  ".join(mismatches))


def lr_scale(step: int, max_steps: int, warmup: int, mode: str) -> float:
    if mode == "constant":
        return 1.0
    if warmup > 0 and step <= warmup:
        return max(step / warmup, 1e-3)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def validate_heads(
    models: dict[str, torch.nn.Module],
    emb: ControlledEmbedding,
    *,
    times: Sequence[float],
    samples_per_time: int,
    batch_size: int,
    jitter: float,
    clip: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict]]:
    totals = {k: 0.0 for k in models}
    counts = {k: 0 for k in models}
    rows = []
    for tid, time in enumerate(times):
        g = torch.Generator(device=device.type)
        g.manual_seed(stable_seed(seed, tid, 7331))
        per_head = {k: 0.0 for k in models}
        seen = 0
        for start in range(0, samples_per_time, batch_size):
            n = min(batch_size, samples_per_time - start)
            u = v4.sample_spiral_2d(n, device=device, jitter=jitter, generator=g)
            x = emb.embed(u)
            eps = torch.randn(x.shape, device=device, generator=g)
            t = torch.full((n,), float(time), device=device)
            xt = (1 - t[:, None]) * x + t[:, None] * eps
            target_v = eps - x
            for target, model in models.items():
                out = model(xt, t)
                pred_v = v4.velocity_from_output(out, xt, t, target, clip)
                loss = (pred_v - target_v).square().mean(dim=1)
                per_head[target] += float(loss.double().sum().cpu())
            seen += n
        row = {"time": float(time)}
        for target in models:
            value = per_head[target] / seen
            row[f"v_loss_{target}"] = value
            totals[target] += value
            counts[target] += 1
        rows.append(row)
    return {k: totals[k] / counts[k] for k in models}, rows


def train_triplet_v10(
    *,
    emb: ControlledEmbedding,
    hidden: int,
    depth: int,
    time_dim: int,
    training_mode: str,
    fixed_steps: int,
    max_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    loss_space: str,
    t_min: float,
    t_max: float,
    clip: float,
    jitter: float,
    val_times: Sequence[float],
    val_samples_per_time: int,
    val_batch_size: int,
    val_every: int,
    patience_evals: int,
    min_rel_improve: float,
    scheduler: str,
    warmup_steps: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    resume_training: bool,
    checkpoint_every: int,
) -> dict[str, torch.nn.Module]:
    if training_mode not in {"fixed", "converged"}:
        raise ValueError(training_mode)
    steps = fixed_steps if training_mode == "fixed" else max_steps
    actual_scheduler = scheduler
    if scheduler == "auto":
        actual_scheduler = "constant" if training_mode == "fixed" else "cosine"
    training_config = {
        "D": emb.D,
        "curvature": emb.curvature,
        "frequency_scale": emb.frequency_scale,
        "embedding_seed": emb.seed,
        "scale_mode": emb.scale_mode,
        "embedding_extra_scale": emb.extra_scale,
        "hidden": hidden,
        "depth": depth,
        "time_dim": time_dim,
        "training_mode": training_mode,
        "planned_steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "loss_space": loss_space,
        "t_min": t_min,
        "t_max": t_max,
        "conversion_clip": clip,
        "data_jitter": jitter,
        "val_times": [float(value) for value in val_times],
        "val_samples_per_time": val_samples_per_time,
        "val_batch_size": val_batch_size,
        "val_every": val_every,
        "patience_evals": patience_evals,
        "min_rel_improve": min_rel_improve,
        "actual_scheduler": actual_scheduler,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }
    models = v4.build_same_init_models(
        emb.D, hidden, depth, time_dim, device, stable_seed(seed, hidden, 91)
    )
    opts = {
        k: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        for k, m in models.items()
    }
    status = {k: HeadStop() for k in models}
    train_rows, val_rows = [], []
    g = torch.Generator(device=device.type)
    g.manual_seed(stable_seed(seed, hidden, 92))
    checkpoint_path = output_dir / "training_checkpoint_v10.pt"
    start_step = 0
    resumed = False
    if resume_training:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("protocol") != "prediction_target_toy_v10_training_v1":
            raise ValueError(f"unsupported v10 training checkpoint: {checkpoint_path}")
        start_step = int(checkpoint["step"])
        _validate_training_resume_config(
            checkpoint["training_config"],
            training_config,
            checkpoint_step=start_step,
        )
        for target, model in models.items():
            model.load_state_dict(checkpoint["models"][target])
            opts[target].load_state_dict(checkpoint["optimizers"][target])
        status = _restore_head_stops(checkpoint["head_status"])
        train_rows = list(checkpoint.get("train_rows", []))
        val_rows = list(checkpoint.get("val_rows", []))
        g.set_state(checkpoint["data_generator_state"].cpu())
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)
        resumed = True

    def save_training_checkpoint(step: int) -> None:
        atomic_torch_save(
            {
                "protocol": "prediction_target_toy_v10_training_v1",
                "step": int(step),
                "training_config": training_config,
                "models": {target: model.state_dict() for target, model in models.items()},
                "optimizers": {target: opt.state_dict() for target, opt in opts.items()},
                "head_status": _serialize_head_stops(status),
                "train_rows": train_rows,
                "val_rows": val_rows,
                "data_generator_state": g.get_state().cpu(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
                ),
            },
            checkpoint_path,
        )
        save_csv(output_dir / "train_history_v10.csv", train_rows)
        save_csv(output_dir / "validation_history_v10.csv", val_rows)

    completed_step = start_step
    already_stopped = training_mode == "converged" and all(
        value.stopped for value in status.values()
    )
    step_iterator = range(start_step + 1, steps + 1) if not already_stopped else ()
    for step in step_iterator:
        scale = lr_scale(step, steps, warmup_steps, actual_scheduler)
        for opt in opts.values():
            for group in opt.param_groups:
                group["lr"] = lr * scale
        u = v4.sample_spiral_2d(batch_size, device=device, jitter=jitter, generator=g)
        x = emb.embed(u)
        eps = torch.randn(x.shape, device=device, generator=g)
        t = torch.empty(batch_size, device=device).uniform_(t_min, t_max, generator=g)
        xt = (1 - t[:, None]) * x + t[:, None] * eps

        losses = {}
        for target, model in models.items():
            if training_mode == "converged" and status[target].stopped:
                losses[target] = float("nan")
                continue
            opts[target].zero_grad(set_to_none=True)
            out = model(xt, t)
            loss = v4.loss_for_output(
                out,
                x_t=xt,
                t=t,
                x=x,
                eps=eps,
                target=target,
                loss_space=loss_space,
                conversion_clip=clip,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opts[target].step()
            losses[target] = float(loss.detach().cpu())

        if step == 1 or step % max(val_every // 4, 1) == 0 or step == steps:
            train_rows.append({"step": step, "lr": lr * scale, **{f"loss_{k}": v for k, v in losses.items()}})

        should_stop = False
        if step % val_every == 0 or step == steps:
            scores, by_time = validate_heads(
                models,
                emb,
                times=val_times,
                samples_per_time=val_samples_per_time,
                batch_size=val_batch_size,
                jitter=jitter,
                clip=clip,
                seed=stable_seed(seed, 93811),
                device=device,
            )
            val_rows.append({"step": step, **{f"score_{k}": v for k, v in scores.items()}})
            for r in by_time:
                val_rows.append({"step": step, **r})
            if training_mode == "converged":
                for target, score in scores.items():
                    st = status[target]
                    rel = (st.best_score - score) / max(abs(st.best_score), 1e-12)
                    if (not math.isfinite(st.best_score)) or rel > min_rel_improve:
                        st.best_score = score
                        st.best_step = step
                        st.stale = 0
                        st.best_state = copy.deepcopy(models[target].state_dict())
                    else:
                        st.stale += 1
                        if st.stale >= patience_evals:
                            st.stopped = True
                should_stop = all(s.stopped for s in status.values())

        completed_step = step
        should_checkpoint = (
            (checkpoint_every > 0 and step % checkpoint_every == 0)
            or step == steps
            or should_stop
        )
        if should_checkpoint:
            save_training_checkpoint(step)
        if should_stop:
            break

    if training_mode == "converged":
        for target, st in status.items():
            if st.best_state is not None:
                models[target].load_state_dict(st.best_state)
    atomic_torch_save(
        {target: model.state_dict() for target, model in models.items()},
        output_dir / "models_v10.pt",
    )
    save_csv(output_dir / "train_history_v10.csv", train_rows)
    save_csv(output_dir / "validation_history_v10.csv", val_rows)
    json_dump(
        output_dir / "training_stop_status.json",
        {
            "training_mode": training_mode,
            "actual_scheduler": actual_scheduler,
            "completed_step": completed_step,
            "resumed": resumed,
            "checkpoint": str(checkpoint_path),
            "heads": {
                k: {
                    "best_score": v.best_score,
                    "best_step": v.best_step,
                    "stale": v.stale,
                    "stopped": v.stopped,
                }
                for k, v in status.items()
            },
        },
    )
    return models


# -----------------------------------------------------------------------------
# Optional exact Bayes teacher probe
# -----------------------------------------------------------------------------

@torch.inference_mode()
def exact_bayes_clean_jitter0(
    state: torch.Tensor,
    t: torch.Tensor,
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    chunk_grid: int = 2048,
) -> torch.Tensor:
    grid_x = emb.embed(locator.u)
    results = []
    for b in range(len(state)):
        tb = float(t[b].item())
        sigma2 = max(tb * tb, 1e-12)
        mean_scale = 1.0 - tb
        parts = []
        for start in range(0, len(grid_x), chunk_grid):
            gx = grid_x[start : start + chunk_grid]
            d2 = (state[b : b + 1] - mean_scale * gx).double().square().sum(dim=1)
            parts.append(d2)
        d2 = torch.cat(parts)
        logw = -0.5 * d2 / sigma2
        w = torch.softmax(logw - logw.max(), dim=0).float()
        results.append(torch.sum(w[:, None] * grid_x, dim=0))
    return torch.stack(results)


@torch.inference_mode()
def run_exact_bayes_probe(
    models: dict[str, torch.nn.Module],
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    *,
    times: Sequence[float],
    samples: int,
    clip: float,
    seed: int,
    device: torch.device,
) -> list[dict]:
    rows = []
    for tid, time in enumerate(times):
        g = torch.Generator(device=device.type)
        g.manual_seed(stable_seed(seed, tid, 55191))
        u = v4.sample_spiral_2d(samples, device=device, jitter=0.0, generator=g)
        x = emb.embed(u)
        eps = torch.randn(x.shape, device=device, generator=g)
        t = torch.full((samples,), float(time), device=device)
        state = (1 - t[:, None]) * x + t[:, None] * eps
        exact = exact_bayes_clean_jitter0(state, t, emb, locator)
        clean = {}
        for target, model in models.items():
            out = model(state, t)
            clean[target] = v4.clean_from_output(out, state, t, target, clip)
        gxv = clean["x"] - clean["v"]
        gxe = clean["x"] - clean["eps"]
        residual = exact - clean["x"]

        frame = build_geometry_frame(emb, locator, clean["x"])
        rx_c, rx_r, rx_a = split_with_geometry_frame(frame, residual)
        xv_c, xv_r, xv_a = split_with_geometry_frame(frame, gxv)
        xe_c, xe_r, xe_a = split_with_geometry_frame(frame, gxe)

        def gamma_star(res, gap):
            return (
                (res.double() * gap.double()).sum(1)
                / gap.double().square().sum(1).clamp_min(EPS)
            )

        rows.append({
            "time": float(time),
            "samples": samples,
            "x_bayes_mse": float((clean["x"] - exact).square().mean().cpu()),
            "v_bayes_mse": float((clean["v"] - exact).square().mean().cpu()),
            "eps_bayes_mse": float((clean["eps"] - exact).square().mean().cpu()),
            "cos_xv_bayes_residual": float(row_cos(gxv, residual).mean().cpu()),
            "cos_xeps_bayes_residual": float(row_cos(gxe, residual).mean().cpu()),
            "gamma_star_xv_bayes": float(gamma_star(residual, gxv).mean().cpu()),
            "gamma_star_xeps_bayes": float(gamma_star(residual, gxe).mean().cpu()),
            "cos_xv_curve_bayes_residual": float(row_cos(xv_c, rx_c).mean().cpu()),
            "cos_xv_ridge_bayes_residual": float(row_cos(xv_r, rx_r).mean().cpu()),
            "cos_xv_ambient_bayes_residual": float(row_cos(xv_a, rx_a).mean().cpu()),
            "cos_xeps_curve_bayes_residual": float(row_cos(xe_c, rx_c).mean().cpu()),
            "cos_xeps_ridge_bayes_residual": float(row_cos(xe_r, rx_r).mean().cpu()),
            "cos_xeps_ambient_bayes_residual": float(row_cos(xe_a, rx_a).mean().cpu()),
            "gamma_star_xv_curve_bayes": float(gamma_star(rx_c, xv_c).mean().cpu()),
            "gamma_star_xv_ridge_bayes": float(gamma_star(rx_r, xv_r).mean().cpu()),
            "gamma_star_xv_ambient_bayes": float(gamma_star(rx_a, xv_a).mean().cpu()),
            "gamma_star_xeps_curve_bayes": float(gamma_star(rx_c, xe_c).mean().cpu()),
            "gamma_star_xeps_ridge_bayes": float(gamma_star(rx_r, xe_r).mean().cpu()),
            "gamma_star_xeps_ambient_bayes": float(gamma_star(rx_a, xe_a).mean().cpu()),
        })
    return rows


# -----------------------------------------------------------------------------
# Guidance
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    strength: float = 0.0
    t_low: float = 0.0
    t_high: float = 1.0

    def active(self, time: float) -> bool:
        return self.t_low <= time <= self.t_high


def normalized_relative_action(
    gap: torch.Tensor, x_clean: torch.Tensor, rho: float
) -> torch.Tensor:
    return (
        float(rho)
        * row_rms(x_clean)[:, None]
        * gap
        / row_rms(gap)[:, None].clamp_min(1e-8)
    )


def normalized_absolute_action(gap: torch.Tensor, eta: float) -> torch.Tensor:
    """Legacy v4 xv_rms_eta semantics: RMS(delta)=|eta| sample-by-sample."""
    return float(eta) * gap / row_rms(gap)[:, None].clamp_min(1e-8)


def gaussian_covariance_surrogate(gap: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    b = len(gap)
    mean = gap.mean(dim=0, keepdim=True)
    centered = gap - mean
    if b <= 1:
        return torch.randn(gap.shape, device=gap.device, generator=generator) * row_rms(gap)[:, None]
    z = torch.randn((b, b), device=gap.device, generator=generator)
    return mean + (z @ centered) / math.sqrt(b - 1)


def _needs_v(kind: str) -> bool:
    return kind == "v" or kind.startswith("xv")


def _needs_eps(kind: str) -> bool:
    return kind == "eps" or kind.startswith("xeps")


def _apply_guidance_from_predictions_v10(
    *,
    x_clean: torch.Tensor,
    other: torch.Tensor | None,
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    condition: Condition,
    time: float,
    step: int,
    control_seed: int,
    collect_diag: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    kind = condition.kind
    if kind == "x" or not condition.active(time):
        return x_clean, {"active": 0.0}
    if other is None:
        raise ValueError(f"{kind} requires a second prediction head")
    if kind in {"v", "eps"}:
        return other, {"active": 1.0}

    real_gap = x_clean - other
    gap = real_gap
    gen = torch.Generator(device=x_clean.device.type)
    gen.manual_seed(stable_seed(control_seed, step, len(x_clean), 473))

    if "_shuffle" in kind:
        if len(gap) < 2:
            raise ValueError(f"{kind} requires batch size >=2")
        gap = gap[torch.randperm(len(gap), generator=gen, device=gap.device)]
    elif "_gausscov" in kind:
        if len(gap) < 2:
            raise ValueError(f"{kind} requires batch size >=2")
        gap = gaussian_covariance_surrogate(gap, generator=gen)
    elif "_random" in kind:
        rnd = torch.randn(gap.shape, device=gap.device, generator=gen)
        gap = rnd * row_rms(real_gap)[:, None] / row_rms(rnd)[:, None].clamp_min(1e-8)

    frame: GeometryFrame | None = None
    component_name = "full"
    if "_curve" in kind or "_ridge" in kind or "_ambient" in kind:
        frame = build_geometry_frame(emb, locator, x_clean)
        curve, ridge, ambient = split_with_geometry_frame(frame, gap)
        if "_curve" in kind:
            gap = curve
            component_name = "curve"
        elif "_ridge" in kind:
            gap = ridge
            component_name = "ridge"
        else:
            gap = ambient
            component_name = "ambient"

    if kind.endswith("_rel"):
        delta = normalized_relative_action(gap, x_clean, condition.strength)
    elif kind.endswith("_absnorm"):
        delta = normalized_absolute_action(gap, condition.strength)
    else:
        delta = float(condition.strength) * gap
    guided = x_clean + delta

    if not collect_diag:
        return guided, {"active": 1.0}

    if frame is None:
        frame = build_geometry_frame(emb, locator, x_clean)
    gap_curve, gap_ridge, gap_ambient = split_with_geometry_frame(frame, real_gap)
    action_curve, action_ridge, action_ambient = split_with_geometry_frame(frame, delta)
    gap_contract, gap_inward, gap_ridge_intrinsic_rms = _contraction_stats(frame, real_gap)
    action_contract, action_inward, action_ridge_intrinsic_rms = _contraction_stats(frame, delta)
    return guided, {
        "active": 1.0,
        "time": time,
        "gap_rms": float(row_rms(real_gap).mean().cpu()),
        "action_rms": float(row_rms(delta).mean().cpu()),
        "action_relative_to_x": float(
            (row_rms(delta) / row_rms(x_clean).clamp_min(1e-8)).mean().cpu()
        ),
        "gap_curve_rms": float(row_rms(gap_curve).mean().cpu()),
        "gap_ridge_rms": float(row_rms(gap_ridge).mean().cpu()),
        "gap_ambient_rms": float(row_rms(gap_ambient).mean().cpu()),
        "action_curve_rms": float(row_rms(action_curve).mean().cpu()),
        "action_ridge_rms": float(row_rms(action_ridge).mean().cpu()),
        "action_ambient_rms": float(row_rms(action_ambient).mean().cpu()),
        "gap_ridge_contraction_slope": gap_contract,
        "gap_ridge_inward_fraction": gap_inward,
        "gap_ridge_intrinsic_action_rms": gap_ridge_intrinsic_rms,
        "action_ridge_contraction_slope": action_contract,
        "action_ridge_inward_fraction": action_inward,
        "action_ridge_intrinsic_action_rms": action_ridge_intrinsic_rms,
        "anchor_ridge_signed_mean": float(frame.signed_ridge_coordinate.mean().cpu()),
        "anchor_ridge_abs_mean": float(frame.signed_ridge_coordinate.abs().mean().cpu()),
        "component": component_name,
    }


@torch.inference_mode()
def guided_clean_v10(
    *,
    models: dict[str, torch.nn.Module],
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    state: torch.Tensor,
    t: torch.Tensor,
    condition: Condition,
    clip: float,
    step: int,
    control_seed: int,
    collect_diag: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    time = float(t[0].item())
    x_out = models["x"](state, t)
    x_clean = v4.clean_from_output(x_out, state, t, "x", clip)
    kind = condition.kind
    if kind == "x" or not condition.active(time):
        other = None
    elif _needs_v(kind):
        out = models["v"](state, t)
        other = v4.clean_from_output(out, state, t, "v", clip)
    elif _needs_eps(kind):
        out = models["eps"](state, t)
        other = v4.clean_from_output(out, state, t, "eps", clip)
    else:
        raise ValueError(kind)
    return _apply_guidance_from_predictions_v10(
        x_clean=x_clean,
        other=other,
        emb=emb,
        locator=locator,
        condition=condition,
        time=time,
        step=step,
        control_seed=control_seed,
        collect_diag=collect_diag,
    )


@torch.inference_mode()
def batched_guided_clean_v10(
    *,
    models: dict[str, torch.nn.Module],
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    states: torch.Tensor,
    t: torch.Tensor,
    conditions: Sequence[Condition],
    clip: float,
    step: int,
    control_seed: int,
    collect_diag: bool = False,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Evaluate independent trajectories in one batch per prediction head."""
    if states.ndim != 3:
        raise ValueError("states must have shape [conditions,batch,D]")
    if len(states) != len(conditions):
        raise ValueError("states and conditions must have equal length")
    if t.ndim != 1 or t.shape[0] != states.shape[1]:
        raise ValueError("t must have shape [batch]")

    condition_count, batch, dimension = states.shape
    time = float(t[0].item())
    flat_states = states.reshape(condition_count * batch, dimension)
    flat_t = t.repeat(condition_count)
    x_output = models["x"](flat_states, flat_t)
    x_clean = v4.clean_from_output(x_output, flat_states, flat_t, "x", clip).reshape_as(states)

    other_by_condition: dict[int, torch.Tensor] = {}
    for target, needed in (
        ("v", _needs_v),
        ("eps", _needs_eps),
    ):
        indices = [
            index
            for index, condition in enumerate(conditions)
            if condition.active(time) and needed(condition.kind)
        ]
        if not indices:
            continue
        target_states = states[indices].reshape(len(indices) * batch, dimension)
        target_t = t.repeat(len(indices))
        output = models[target](target_states, target_t)
        clean = v4.clean_from_output(
            output, target_states, target_t, target, clip
        ).reshape(len(indices), batch, dimension)
        for position, condition_index in enumerate(indices):
            other_by_condition[condition_index] = clean[position]

    guided_values = []
    diagnostics = []
    for index, condition in enumerate(conditions):
        guided, diag = _apply_guidance_from_predictions_v10(
            x_clean=x_clean[index],
            other=other_by_condition.get(index),
            emb=emb,
            locator=locator,
            condition=condition,
            time=time,
            step=step,
            control_seed=control_seed,
            collect_diag=collect_diag,
        )
        guided_values.append(guided)
        diagnostics.append(diag)
    return torch.stack(guided_values), diagnostics


@torch.inference_mode()
def sample_conditions_batched_v10(
    *,
    models: dict[str, torch.nn.Module],
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    conditions: Sequence[Condition],
    sample_count: int,
    batch_size: int,
    steps: int,
    t_max: float,
    t_min: float,
    clip: float,
    seed: int,
    device: torch.device,
    diag_stride: int,
) -> tuple[list[np.ndarray], list[dict]]:
    if not conditions:
        return [], []
    collected: list[list[np.ndarray]] = [[] for _ in conditions]
    diagnostics_by_condition: list[list[dict]] = [[] for _ in conditions]
    grid = torch.linspace(t_max, t_min, steps + 1, device=device)
    for start in range(0, sample_count, batch_size):
        n = min(batch_size, sample_count - start)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed + start)
        initial = float(t_max) * torch.randn(
            (n, emb.D), device=device, generator=generator
        )
        states = initial[None].expand(len(conditions), -1, -1).clone()
        for step in range(steps):
            t_now, t_next = grid[step], grid[step + 1]
            t = t_now.expand(n)
            collect = start == 0 and step % max(diag_stride, 1) == 0
            guided, diag_values = batched_guided_clean_v10(
                models=models,
                emb=emb,
                locator=locator,
                states=states,
                t=t,
                conditions=conditions,
                clip=clip,
                step=step,
                control_seed=stable_seed(seed, start, 7021),
                collect_diag=collect,
            )
            states = states + (t_next - t_now) * (
                (states - guided) / t_now.clamp_min(clip)
            )
            if collect:
                for index, (condition, diag) in enumerate(zip(conditions, diag_values)):
                    if diag.get("active", 0) > 0 and "time" in diag:
                        diagnostics_by_condition[index].append(
                            {"condition": condition.name, "step": step, **diag}
                        )

        t = grid[-1].expand(n)
        final, _ = batched_guided_clean_v10(
            models=models,
            emb=emb,
            locator=locator,
            states=states,
            t=t,
            conditions=conditions,
            clip=clip,
            step=steps,
            control_seed=stable_seed(seed, start, 7022),
            collect_diag=False,
        )
        for index, value in enumerate(final):
            collected[index].append(value.cpu().numpy())
    diagnostics = [
        row for condition_rows in diagnostics_by_condition for row in condition_rows
    ]
    return [np.concatenate(parts, axis=0) for parts in collected], diagnostics


@torch.inference_mode()
def sample_condition_recursive_v10(
    *,
    models: dict[str, torch.nn.Module],
    emb: ControlledEmbedding,
    locator: SpiralLocator,
    condition: Condition,
    sample_count: int,
    batch_size: int,
    steps: int,
    t_max: float,
    t_min: float,
    clip: float,
    seed: int,
    device: torch.device,
    diag_stride: int,
) -> tuple[np.ndarray, list[dict]]:
    outputs, diagnostics = [], []
    grid = torch.linspace(t_max, t_min, steps + 1, device=device)
    for start in range(0, sample_count, batch_size):
        n = min(batch_size, sample_count - start)
        g = torch.Generator(device=device.type)
        g.manual_seed(seed + start)
        state = float(t_max) * torch.randn((n, emb.D), device=device, generator=g)
        for i in range(steps):
            t_now, t_next = grid[i], grid[i + 1]
            t = torch.full((n,), float(t_now), device=device)
            guided, diag = guided_clean_v10(
                models=models,
                emb=emb,
                locator=locator,
                state=state,
                t=t,
                condition=condition,
                clip=clip,
                step=i,
                control_seed=stable_seed(seed, start, 7021),
                collect_diag=(start == 0 and i % max(diag_stride, 1) == 0),
            )
            vel = (state - guided) / t[:, None].clamp_min(clip)
            state = state + (t_next - t_now) * vel
            if (
                start == 0
                and i % max(diag_stride, 1) == 0
                and diag.get("active", 0) > 0
                and "time" in diag
            ):
                diagnostics.append({"condition": condition.name, "step": i, **diag})
        t = torch.full((n,), float(grid[-1]), device=device)
        final, _ = guided_clean_v10(
            models=models,
            emb=emb,
            locator=locator,
            state=state,
            t=t,
            condition=condition,
            clip=clip,
            step=steps,
            control_seed=stable_seed(seed, start, 7022),
            collect_diag=False,
        )
        outputs.append(final.cpu().numpy())
    return np.concatenate(outputs), diagnostics


# -----------------------------------------------------------------------------
# Conditions
# -----------------------------------------------------------------------------

def build_conditions(args) -> list[Condition]:
    out = [Condition("x", "x"), Condition("v", "v"), Condition("eps", "eps")]
    if args.condition_suite == "baseline":
        return out

    if args.condition_suite in {"fine", "mechanism"}:
        for g in args.gamma_xeps:
            out.append(Condition(f"xeps_g{tag_float(g)}", "xeps", g))
        for g in args.gamma_xv:
            out.append(Condition(f"xv_g{tag_float(g)}", "xv", g))

    # Explicit path: alpha=0 other, alpha=1 x, alpha>1 beyond x.
    if args.condition_suite in {"mechanism", "paradox"}:
        for alpha in args.path_alphas:
            if abs(alpha) < 1e-12 or abs(alpha - 1.0) < 1e-12:
                continue
            g = float(alpha) - 1.0
            out.extend([
                Condition(f"xeps_path_a{tag_float(alpha)}", "xeps", g),
                Condition(f"xv_path_a{tag_float(alpha)}", "xv", g),
            ])

    # Legacy v4 absolute-normalized actions.  These are critical regression anchors.
    for eta in args.absolute_actions:
        out.extend([
            Condition(f"xeps_absnorm_eta{tag_float(eta)}", "xeps_absnorm", eta),
            Condition(f"xv_absnorm_eta{tag_float(eta)}", "xv_absnorm", eta),
        ])

    # Relative-to-x action matching.
    for rho in args.relative_actions:
        out.extend([
            Condition(f"xeps_rel{tag_float(rho)}", "xeps_rel", rho),
            Condition(f"xv_rel{tag_float(rho)}", "xv_rel", rho),
        ])

    if args.condition_suite == "fine":
        return out

    # Destroy pairing / structure for BOTH gap families.
    for g in args.control_gammas:
        for prefix in ("xeps", "xv"):
            out.extend([
                Condition(f"{prefix}_shuffle_g{tag_float(g)}", f"{prefix}_shuffle", g),
                Condition(f"{prefix}_gausscov_g{tag_float(g)}", f"{prefix}_gausscov", g),
                Condition(f"{prefix}_random_g{tag_float(g)}", f"{prefix}_random", g),
            ])

    # Geometry components: raw gamma.
    for g in args.geometry_gammas:
        for prefix in ("xeps", "xv"):
            for comp in ("curve", "ridge", "ambient"):
                out.append(Condition(f"{prefix}_{comp}_g{tag_float(g)}", f"{prefix}_{comp}", g))

    # Geometry components: equal relative action.
    for rho in args.geometry_relative_actions:
        for prefix in ("xeps", "xv"):
            for comp in ("curve", "ridge", "ambient"):
                out.append(Condition(
                    f"{prefix}_{comp}_rel{tag_float(rho)}",
                    f"{prefix}_{comp}_rel",
                    rho,
                ))

    # Geometry components: legacy absolute-normalized action.
    for eta in args.geometry_absolute_actions:
        for prefix in ("xeps", "xv"):
            for comp in ("curve", "ridge", "ambient"):
                out.append(Condition(
                    f"{prefix}_{comp}_absnorm_eta{tag_float(eta)}",
                    f"{prefix}_{comp}_absnorm",
                    eta,
                ))

    windows = [
        ("early", 0.80, args.sample_t_max),
        ("mid_hi", 0.60, 0.80),
        ("mid_lo", 0.40, 0.60),
        ("late", args.sample_t_min, 0.40),
    ]
    for g in args.stage_gammas:
        for label, lo, hi in windows:
            for prefix in ("xeps", "xv"):
                out.append(Condition(
                    f"{prefix}_{label}_g{tag_float(g)}", prefix, g, t_low=lo, t_high=hi
                ))
    return out


def _path_condition_name(pair: str, alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return "eps" if pair == "xeps" else "v"
    if abs(alpha - 1.0) < 1e-12:
        return "x"
    return f"{pair}_path_a{tag_float(alpha)}"


def _atlas_entries_for_pair(
    pair: str,
    *,
    cache: dict[str, np.ndarray],
    path_alphas: Sequence[float],
    raw_gammas: Sequence[float],
) -> list[tuple[float, str, np.ndarray]]:
    """Resolve one visual path without resampling duplicate raw conditions."""
    entries: dict[float, tuple[str, np.ndarray]] = {}

    def add(alpha: float, name: str) -> None:
        if name in cache:
            entries[round(float(alpha), 12)] = (name, cache[name])

    add(0.0, "eps" if pair == "xeps" else "v")
    add(1.0, "x")
    for alpha in path_alphas:
        add(float(alpha), _path_condition_name(pair, float(alpha)))
    # Prefer already-computed raw-gamma trajectories when both spellings exist.
    for gamma in raw_gammas:
        add(1.0 + float(gamma), f"{pair}_g{tag_float(gamma)}")
    return [
        (alpha, name, value)
        for alpha, (name, value) in sorted(entries.items(), key=lambda item: item[0])
    ]


# -----------------------------------------------------------------------------
# Distribution metrics
# -----------------------------------------------------------------------------

def fixed_theta_nd(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, dim))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12
    return theta


def swd_fixed(
    a: np.ndarray,
    b: np.ndarray,
    *,
    theta: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> float:
    aa = a[idx_a]
    bb = b[idx_b]
    pa = np.sort(aa @ theta.T, axis=0)
    pb = np.sort(bb @ theta.T, axis=0)
    return float(np.mean(np.sqrt(np.mean((pa - pb) ** 2, axis=0))))


@torch.inference_mode()
def prepare_swd_reference_device(
    reference: np.ndarray,
    *,
    theta: np.ndarray,
    idx_ref: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache fixed reference projections for repeated high-D SWD comparisons.

    Uses float64 to preserve the numerical definition of the NumPy metric while
    moving the expensive D-dimensional GEMM/sort to the selected torch device.
    This changes execution only, not the metric or matched-subset protocol.
    """
    theta_t = torch.as_tensor(theta, device=device, dtype=torch.float64)
    ref_t = torch.as_tensor(reference[idx_ref], device=device, dtype=torch.float64)
    ref_sorted = torch.sort(ref_t @ theta_t.T, dim=0).values
    return theta_t, ref_sorted


@torch.inference_mode()
def swd_fixed_against_cached_reference_device(
    candidate: np.ndarray,
    *,
    theta_t: torch.Tensor,
    idx_candidate: np.ndarray,
    ref_sorted: torch.Tensor,
    device: torch.device,
) -> float:
    cand_t = torch.as_tensor(candidate[idx_candidate], device=device, dtype=torch.float64)
    cand_sorted = torch.sort(cand_t @ theta_t.T, dim=0).values
    value = torch.sqrt(torch.mean((cand_sorted - ref_sorted).square(), dim=0)).mean()
    return float(value.cpu())


def rbf_bandwidth_fixed(
    a: np.ndarray,
    b: np.ndarray,
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    bandwidth_subset: np.ndarray,
) -> float:
    aa = a[idx_a]
    bb = b[idx_b]
    joined = np.concatenate([aa, bb], axis=0)
    sub = joined[bandwidth_subset % len(joined)]
    d2 = ((sub[:, None] - sub[None, :]) ** 2).sum(axis=2)
    pos = d2[d2 > 0]
    return max(float(np.median(pos)) if len(pos) else 1.0, 1e-8)


def mmd_rbf_fixed(
    a: np.ndarray,
    b: np.ndarray,
    *,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    sigma2: float,
) -> float:
    aa = a[idx_a]
    bb = b[idx_b]
    n = len(aa)
    if n <= 1:
        return float("nan")

    def k(x, y):
        z = ((x[:, None] - y[None, :]) ** 2).sum(axis=2)
        return np.exp(-z / (2.0 * max(float(sigma2), 1e-8)))

    kaa, kbb, kab = k(aa, aa), k(bb, bb), k(aa, bb)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    return float(
        kaa.sum() / (n * (n - 1))
        + kbb.sum() / (n * (n - 1))
        - 2.0 * kab.mean()
    )


def bootstrap_swd_delta_fixed(
    candidate: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    *,
    theta: np.ndarray,
    reps: int,
    seed: int,
    max_points: int,
) -> tuple[float, float, float]:
    if reps <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    paired_count = min(len(candidate), len(baseline))
    n = min(paired_count, len(reference), max_points)
    cp = candidate[:paired_count] @ theta.T
    bp = baseline[:paired_count] @ theta.T
    rp = reference @ theta.T
    deltas = []
    for _ in range(reps):
        ids = rng.integers(0, paired_count, size=n)
        rid = rng.integers(0, len(reference), size=n)
        ca = np.sort(cp[ids], axis=0)
        ba = np.sort(bp[ids], axis=0)
        rr = np.sort(rp[rid], axis=0)
        c = float(np.mean(np.sqrt(np.mean((ca - rr) ** 2, axis=0))))
        b = float(np.mean(np.sqrt(np.mean((ba - rr) ** 2, axis=0))))
        deltas.append(c - b)
    arr = np.asarray(deltas, dtype=np.float64)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


@dataclass
class DistributionMetricContext:
    idx_sample_2d: np.ndarray
    idx_ref_2d: np.ndarray
    theta_2d: np.ndarray
    idx_sample_full: np.ndarray
    idx_ref_full: np.ndarray
    theta_full: np.ndarray
    idx_sample_mmd: np.ndarray
    idx_ref_mmd: np.ndarray
    bandwidth_subset: np.ndarray
    mmd_sigma2: float | None = None


def build_distribution_metric_context(
    *,
    sample_count: int,
    reference_count: int,
    D: int,
    swd_projections: int,
    swd_max_points: int,
    full_swd_projections: int,
    full_swd_max_points: int,
    mmd_max_points: int,
    seed: int,
) -> DistributionMetricContext:
    rng = np.random.default_rng(seed)
    n2 = min(sample_count, reference_count, swd_max_points)
    nf = min(sample_count, reference_count, full_swd_max_points)
    nm = min(sample_count, reference_count, max(mmd_max_points, 1))
    idx_s2 = rng.choice(sample_count, n2, replace=False)
    idx_r2 = rng.choice(reference_count, n2, replace=False)
    idx_sf = rng.choice(sample_count, nf, replace=False)
    idx_rf = rng.choice(reference_count, nf, replace=False)
    idx_sm = rng.choice(sample_count, nm, replace=False)
    idx_rm = rng.choice(reference_count, nm, replace=False)
    bw = rng.choice(2 * nm, min(1024, 2 * nm), replace=False)
    return DistributionMetricContext(
        idx_sample_2d=idx_s2,
        idx_ref_2d=idx_r2,
        theta_2d=fixed_theta_nd(swd_projections, 2, stable_seed(seed, 11)),
        idx_sample_full=idx_sf,
        idx_ref_full=idx_rf,
        theta_full=fixed_theta_nd(full_swd_projections, D, stable_seed(seed, 12)),
        idx_sample_mmd=idx_sm,
        idx_ref_mmd=idx_rm,
        bandwidth_subset=bw,
    )


# -----------------------------------------------------------------------------
# Visual atlas
# -----------------------------------------------------------------------------

def _intrinsic_signed_features(
    intrinsic: np.ndarray, *, locator: SpiralLocator, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    u = torch.from_numpy(intrinsic).to(device)
    s, u0, t2, _ = locator.nearest(u)
    n2 = torch.stack([-t2[:, 1], t2[:, 0]], dim=1)
    signed = ((u - u0) * n2).sum(dim=1)
    return s.cpu().numpy(), signed.cpu().numpy()


def save_visual_paradox_atlas(
    path: Path,
    *,
    reference_u: np.ndarray,
    cache: dict[str, np.ndarray],
    rows: list[dict],
    alphas: Sequence[float],
    gamma_xeps: Sequence[float],
    gamma_xv: Sequence[float],
    absolute_actions: Sequence[float],
    locator: SpiralLocator,
    device: torch.device,
    plot_points: int,
) -> None:
    row_map = {str(r["condition"]): r for r in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path_entries = {
        "xeps": _atlas_entries_for_pair(
            "xeps", cache=cache, path_alphas=alphas, raw_gammas=gamma_xeps
        ),
        "xv": _atlas_entries_for_pair(
            "xv", cache=cache, path_alphas=alphas, raw_gammas=gamma_xv
        ),
    }

    def title_for(name: str, extra: str = "") -> str:
        r = row_map.get(name)
        if r is None:
            return name + ("\n" + extra if extra else "")
        bits = [name]
        if extra:
            bits.append(extra)
        wr = r.get("ridge_width_ratio", float("nan"))
        rr = r.get("intrinsic_radius_mean_ratio", float("nan"))
        aq = r.get("ambient_surface_rms_q90", float("nan"))
        if np.isfinite(wr): bits.append(f"ridge width/ref={wr:.2f}")
        if np.isfinite(rr): bits.append(f"radius/ref={rr:.2f}")
        bits.append(f"ambient q90={aq:.3g}")
        return "\n".join(bits)

    def scatter(ax, arr: np.ndarray, title: str):
        a = arr[:plot_points]
        ax.scatter(a[:, 0], a[:, 1], s=3, alpha=0.42)
        ax.plot(locator.u[:, 0].cpu(), locator.u[:, 1].cpu(), linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_title(title, fontsize=8)

    with PdfPages(path) as pdf:
        # Page 1 baseline.
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        scatter(axes[0], reference_u, "reference")
        for ax, name in zip(axes[1:], ("eps", "v", "x")):
            if name in cache: scatter(ax, cache[name], title_for(name))
            else: ax.axis("off")
        fig.suptitle("Page 1 — baseline endpoint populations")
        fig.tight_layout(); pdf.savefig(fig, dpi=180); plt.close(fig)

        # Page 2/3 raw path.
        for page_index, pair in enumerate(("xeps", "xv"), start=2):
            entries = path_entries[pair]
            cols = 4; nr = max(1, math.ceil(len(entries) / cols))
            fig, axes = plt.subplots(nr, cols, figsize=(4.1*cols, 4.1*nr), squeeze=False)
            for ax, (alpha, name, arr) in zip(axes.flat, entries):
                scatter(ax, arr, title_for(name, f"alpha={alpha:g}"))
            for ax in axes.flat[len(entries):]: ax.axis("off")
            label = "epsilon -> x -> beyond x" if pair == "xeps" else "v -> x -> beyond x"
            fig.suptitle(f"Page {page_index} — {label}")
            fig.tight_layout(); pdf.savefig(fig, dpi=180); plt.close(fig)

        # Page 4 legacy absolute-normalized actions; direct regression of old xv_rms_eta.
        entries = []
        for pair in ("xeps", "xv"):
            for eta in absolute_actions:
                name = f"{pair}_absnorm_eta{tag_float(eta)}"
                if name in cache:
                    entries.append((pair, eta, name, cache[name]))
        cols = 4; nr = max(1, math.ceil(len(entries) / cols))
        fig, axes = plt.subplots(nr, cols, figsize=(4.1*cols, 4.1*nr), squeeze=False)
        for ax, (pair, eta, name, arr) in zip(axes.flat, entries):
            scatter(ax, arr, title_for(name, f"{pair}, eta={eta:g}"))
        for ax in axes.flat[len(entries):]: ax.axis("off")
        fig.suptitle("Page 4 — legacy absolute RMS-normalized guidance (v4 regression)")
        fig.tight_layout(); pdf.savefig(fig, dpi=180); plt.close(fig)

        # Page 5 raw distributions.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ref_s, ref_signed = _intrinsic_signed_features(reference_u, locator=locator, device=device)
        for row_i, pair in enumerate(("xeps", "xv")):
            ax_s, ax_n = axes[row_i]
            ax_s.hist(ref_s, bins=80, density=True, histtype="step", label="reference")
            ax_n.hist(ref_signed, bins=80, density=True, histtype="step", label="reference")
            for alpha, name, value in path_entries[pair]:
                s, signed = _intrinsic_signed_features(value, locator=locator, device=device)
                ax_s.hist(s, bins=80, density=True, histtype="step", label=f"a={alpha:g}")
                ax_n.hist(signed, bins=80, density=True, histtype="step", label=f"a={alpha:g}")
            ax_s.set_title(f"{pair}: coverage along spiral")
            ax_n.set_title(f"{pair}: signed ridge-normal distribution")
            ax_s.legend(fontsize=7); ax_n.legend(fontsize=7)
        fig.suptitle("Page 5 — raw distribution decomposition")
        fig.tight_layout(); pdf.savefig(fig, dpi=180); plt.close(fig)

        # Page 6 paired endpoint displacement relative to x.
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
        x0 = cache.get("x")
        if x0 is not None:
            for ri, pair in enumerate(("xeps", "xv")):
                candidates = [entry for entry in path_entries[pair] if entry[0] > 1.0]
                if len(candidates) > 3:
                    selected = np.linspace(0, len(candidates) - 1, 3).round().astype(int)
                    candidates = [candidates[int(index)] for index in selected]
                for ci, (alpha, name, y) in enumerate(candidates):
                    ax = axes[ri, ci]
                    n = min(len(x0), len(y), max(200, plot_points // 8))
                    ids = np.linspace(0, min(len(x0), len(y)) - 1, n).astype(int)
                    a, b = x0[ids], y[ids]
                    d = b - a
                    ax.scatter(a[:, 0], a[:, 1], s=2, alpha=0.18)
                    ax.quiver(a[:, 0], a[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1, width=0.002, alpha=0.35)
                    ax.plot(locator.u[:, 0].cpu(), locator.u[:, 1].cpu(), linewidth=0.8)
                    ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect("equal", adjustable="box")
                    ax.set_title(f"{pair}, alpha={alpha:g}: x endpoint -> guided")
                for ci in range(len(candidates), 3):
                    axes[ri, ci].axis("off")
        fig.suptitle("Page 6 — paired endpoint displacement field")
        fig.tight_layout(); pdf.savefig(fig, dpi=180); plt.close(fig)


# -----------------------------------------------------------------------------
# Checkpoint reuse
# -----------------------------------------------------------------------------

def load_models_from_v4_setting(
    setting: Path,
    *,
    D: int,
    hidden: int,
    depth: int,
    time_dim: int,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    path = setting / "models.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Point --reuse-v4-setting at the local v4 setting directory."
        )
    payload = torch.load(path, map_location=device, weights_only=False)
    models = {}
    for target in ("x", "v", "eps"):
        model = v4.DenoiseMLP(D, hidden, depth, time_dim).to(device)
        state = payload[target] if isinstance(payload, dict) and target in payload else payload
        model.load_state_dict(state)
        model.eval()
        models[target] = model
    return models


# -----------------------------------------------------------------------------
# Setting runner
# -----------------------------------------------------------------------------

def run_setting(args, *, D: int, curvature: float, hidden: int, seed: int, device):
    out = args.output_root / f"seed{seed}" / f"D{D}" / f"curv{tag_float(curvature)}" / f"H{hidden}"
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_v4_setting:
        embedding_seed = v4.stable_seed(seed, D, int(curvature * 10000), 41)
    else:
        embedding_seed = stable_seed(seed, D, 4811)
    emb = ControlledEmbedding(
        D,
        curvature=curvature,
        frequency_scale=args.frequency_scale,
        seed=embedding_seed,
        device=device,
        scale_mode=args.scale_mode,
        energy_match=args.energy_match_curvature,
        calibration_samples=args.energy_calibration_samples,
        data_jitter=args.data_jitter,
    )
    locator = SpiralLocator(args.ridge_grid_points, device)
    geometry_meta = observability_diagnostics(
        emb,
        samples=args.observability_samples,
        jitter=args.data_jitter,
        seed=stable_seed(seed, D, 199),
    )
    json_dump(out / "embedding_observability.json", geometry_meta)

    if args.reuse_v4_setting:
        models = load_models_from_v4_setting(
            Path(args.reuse_v4_setting), D=D, hidden=hidden, depth=args.depth,
            time_dim=args.time_dim, device=device
        )
        training_source = "reused_v4_checkpoint"
    else:
        models = train_triplet_v10(
            emb=emb,
            hidden=hidden,
            depth=args.depth,
            time_dim=args.time_dim,
            training_mode=args.training_mode,
            fixed_steps=args.fixed_steps,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            loss_space=args.loss_space,
            t_min=args.train_t_min,
            t_max=args.train_t_max,
            clip=args.conversion_clip,
            jitter=args.data_jitter,
            val_times=args.val_times,
            val_samples_per_time=args.val_samples_per_time,
            val_batch_size=args.val_batch_size,
            val_every=args.val_every,
            patience_evals=args.patience_evals,
            min_rel_improve=args.min_rel_improve,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            seed=seed,
            device=device,
            output_dir=out,
            resume_training=args.resume_training,
            checkpoint_every=args.training_checkpoint_every,
        )
        training_source = f"v10_{args.training_mode}"
    for model in models.values():
        model.eval()

    if args.exact_bayes_probe:
        if args.data_jitter != 0.0:
            raise ValueError("--exact-bayes-probe requires --data-jitter 0")
        save_csv(
            out / "exact_bayes_teacher_v10.csv",
            run_exact_bayes_probe(
                models, emb, locator, times=args.val_times, samples=args.exact_bayes_samples,
                clip=args.conversion_clip, seed=seed, device=device
            ),
        )

    # ------------------------------------------------------------------
    # Reference bank and all reusable metric geometry.
    # ------------------------------------------------------------------
    rg = torch.Generator(device=device.type)
    rg.manual_seed(stable_seed(seed, D, 823))
    ref_u_t = v4.sample_spiral_2d(
        max(args.sample_count, args.reference_count), device=device,
        jitter=args.data_jitter, generator=rg
    )
    reference_u = ref_u_t.cpu().numpy()
    with torch.inference_mode():
        reference_ambient = emb.embed(ref_u_t).cpu().numpy()

    ref_endpoint = build_reference_endpoint_geometry(
        reference_u,
        locator=locator,
        bins=args.coverage_bins,
        device=device,
    )
    metric_ctx = build_distribution_metric_context(
        sample_count=args.sample_count,
        reference_count=len(reference_u),
        D=D,
        swd_projections=args.swd_projections,
        swd_max_points=args.swd_max_points,
        full_swd_projections=args.full_swd_projections,
        full_swd_max_points=args.full_swd_max_points,
        mmd_max_points=args.mmd_max_points,
        seed=stable_seed(seed, D, hidden, 331),
    )
    reference_ambient_rms = float(np.sqrt(np.mean(reference_ambient.astype(np.float64) ** 2)))

    # Full-D SWD is a primary decoder-free diagnostic.  Cache the fixed
    # reference projection once and evaluate candidate projections on the
    # selected torch device; this is algebraically identical to swd_fixed().
    full_theta_t, full_ref_sorted = prepare_swd_reference_device(
        reference_ambient,
        theta=metric_ctx.theta_full,
        idx_ref=metric_ctx.idx_ref_full,
        device=device,
    )

    geom_n = min(args.geometry_metric_points, args.sample_count, len(reference_u))
    curve_locator = None
    ref_curve_feat = None
    geom_sample_ids = geom_ref_ids = None
    if geom_n > 0:
        grng = np.random.default_rng(stable_seed(seed, D, hidden, 8841))
        geom_sample_ids = grng.choice(args.sample_count, geom_n, replace=False)
        geom_ref_ids = grng.choice(len(reference_u), geom_n, replace=False)
        curve_locator = EmbeddedCurveLocator(
            emb,
            coarse_points=args.curveD_coarse_points,
            refine_points=args.curveD_refine_points,
            refine_rounds=args.curveD_refine_rounds,
        )
        ref_curve_feat = embedded_curve_features(
            reference_ambient[geom_ref_ids],
            emb=emb,
            curve_locator=curve_locator,
            device=device,
        )

    conditions = build_conditions(args)
    rows, all_diag = [], []
    visual_cache: dict[str, np.ndarray] = {}
    sample_seed = stable_seed(seed, D, hidden, int(curvature * 10000), 119)
    x_intrinsic: np.ndarray | None = None

    def iter_sampled_conditions():
        if not args.batch_conditions:
            for condition in conditions:
                ambient, diagnostics = sample_condition_recursive_v10(
                    models=models,
                    emb=emb,
                    locator=locator,
                    condition=condition,
                    sample_count=args.sample_count,
                    batch_size=args.sample_batch_size,
                    steps=args.sample_steps,
                    t_max=args.sample_t_max,
                    t_min=args.sample_t_min,
                    clip=args.conversion_clip,
                    seed=sample_seed,
                    device=device,
                    diag_stride=args.diag_stride,
                )
                all_diag.extend(diagnostics)
                yield condition, ambient
            return

        group_size = min(args.condition_batch_size, len(conditions))
        for start in range(0, len(conditions), group_size):
            group = conditions[start : start + group_size]
            print(
                f"[v10 D={D} curv={curvature:g} H={hidden}] sampling conditions "
                f"{start + 1}-{start + len(group)}/{len(conditions)}",
                flush=True,
            )
            ambient_values, diagnostics = sample_conditions_batched_v10(
                models=models,
                emb=emb,
                locator=locator,
                conditions=group,
                sample_count=args.sample_count,
                batch_size=args.sample_batch_size,
                steps=args.sample_steps,
                t_max=args.sample_t_max,
                t_min=args.sample_t_min,
                clip=args.conversion_clip,
                seed=sample_seed,
                device=device,
                diag_stride=args.diag_stride,
            )
            all_diag.extend(diagnostics)
            yield from zip(group, ambient_values)

    for ci, (condition, ambient) in enumerate(iter_sampled_conditions()):
        print(
            f"[v10 D={D} curv={curvature:g} H={hidden}] "
            f"evaluating {ci+1}/{len(conditions)} {condition.name}",
            flush=True,
        )

        # Decode / nearest spiral / consistency are computed ONCE per condition.
        feat = endpoint_features(ambient, emb=emb, locator=locator, device=device)
        intrinsic = feat.intrinsic
        if args.save_samples:
            np.savez_compressed(
                out / f"samples_{condition.name}.npz", ambient=ambient, intrinsic=intrinsic
            )
        rmet = ridge_metrics_from_features(
            feat,
            ref_endpoint,
            bins=args.coverage_bins,
            conditional_bins=args.conditional_ridge_bins,
            conditional_min_count=args.conditional_ridge_min_count,
        )

        swd2 = swd_fixed(
            intrinsic,
            reference_u,
            theta=metric_ctx.theta_2d,
            idx_a=metric_ctx.idx_sample_2d,
            idx_b=metric_ctx.idx_ref_2d,
        )
        swd_full = swd_fixed_against_cached_reference_device(
            ambient,
            theta_t=full_theta_t,
            idx_candidate=metric_ctx.idx_sample_full,
            ref_sorted=full_ref_sorted,
            device=device,
        )

        if args.mmd_max_points > 1:
            if condition.name == "x":
                metric_ctx.mmd_sigma2 = rbf_bandwidth_fixed(
                    intrinsic,
                    reference_u,
                    idx_a=metric_ctx.idx_sample_mmd,
                    idx_b=metric_ctx.idx_ref_mmd,
                    bandwidth_subset=metric_ctx.bandwidth_subset,
                )
            if metric_ctx.mmd_sigma2 is None:
                raise RuntimeError("x baseline must be evaluated before fixed-bandwidth MMD")
            mmd2 = mmd_rbf_fixed(
                intrinsic,
                reference_u,
                idx_a=metric_ctx.idx_sample_mmd,
                idx_b=metric_ctx.idx_ref_mmd,
                sigma2=metric_ctx.mmd_sigma2,
            )
        else:
            mmd2 = float("nan")

        if condition.name == "x":
            x_intrinsic = intrinsic.copy()
            boot_mean = boot_lo = boot_hi = 0.0
        elif args.bootstrap_reps > 0:
            if x_intrinsic is None:
                raise RuntimeError("x baseline must be evaluated before paired bootstrap")
            boot_mean, boot_lo, boot_hi = bootstrap_swd_delta_fixed(
                intrinsic,
                x_intrinsic,
                reference_u,
                theta=metric_ctx.theta_2d[: min(args.bootstrap_projections, len(metric_ctx.theta_2d))],
                reps=args.bootstrap_reps,
                seed=stable_seed(seed, D, hidden, ci, 993),
                max_points=min(args.bootstrap_max_points, args.sample_count),
            )
        else:
            boot_mean = boot_lo = boot_hi = float("nan")

        extra: dict[str, float] = {}
        if geom_n > 0 and curve_locator is not None and ref_curve_feat is not None:
            curve_feat = embedded_curve_features(
                ambient[geom_sample_ids],
                emb=emb,
                curve_locator=curve_locator,
                device=device,
            )
            extra.update(embedded_curve_metrics(curve_feat, ref_curve_feat))

            if args.surface_audit_samples > 0:
                audit_n = min(args.surface_audit_samples, geom_n)
                audit_ids = geom_sample_ids[:audit_n]
                nearest = local_surface_gauss_newton_rms(
                    ambient[audit_ids],
                    emb=emb,
                    device=device,
                    iterations=args.surface_audit_iterations,
                    damping=args.surface_audit_damping,
                )
                consistency = feat.ambient_rms[audit_ids]
                extra.update({
                    "surface_audit_local_nearest_rms_mean": float(np.mean(nearest)),
                    "surface_audit_local_nearest_rms_q90": float(np.quantile(nearest, 0.90)),
                    "surface_audit_consistency_rms_mean": float(np.mean(consistency)),
                    "surface_audit_consistency_minus_nearest_mean": float(
                        np.mean(consistency - nearest)
                    ),
                })

        row = {
            "seed": seed,
            "D": D,
            "curvature": curvature,
            "hidden": hidden,
            "training_source": training_source,
            "condition": condition.name,
            "kind": condition.kind,
            "strength": condition.strength,
            "t_low": condition.t_low,
            "t_high": condition.t_high,
            "swd_2d": swd2,
            "swd_fullD": swd_full,
            "swd_fullD_over_reference_rms": (
                swd_full / max(reference_ambient_rms, 1e-12)
            ),
            "mmd_2d": mmd2,
            "mmd_sigma2_fixed_from_x": (
                metric_ctx.mmd_sigma2 if metric_ctx.mmd_sigma2 is not None else float("nan")
            ),
            "swd_delta_vs_x_boot_mean": boot_mean,
            "swd_delta_vs_x_ci_low": boot_lo,
            "swd_delta_vs_x_ci_high": boot_hi,
            **rmet,
            **extra,
        }
        rows.append(row)
        save_csv(out / "generation_metrics_v10.partial.csv", rows)
        save_csv(out / "trajectory_mechanism_v10.partial.csv", all_diag)

        if args.visual_atlas and (
            condition.name in {"x", "v", "eps"}
            or (
                condition.kind in {"xeps", "xv"}
                and condition.name.startswith(("xeps_g", "xv_g"))
            )
            or "_path_a" in condition.name
            or "_absnorm_eta" in condition.name
        ):
            visual_cache[condition.name] = intrinsic.copy()

    save_csv(out / "generation_metrics_v10.csv", rows)
    save_csv(out / "trajectory_mechanism_v10.csv", all_diag)

    if args.visual_atlas and visual_cache:
        save_visual_paradox_atlas(
            out / "visual_paradox_atlas_v10.pdf",
            reference_u=reference_u,
            cache=visual_cache,
            rows=rows,
            alphas=args.path_alphas,
            gamma_xeps=args.gamma_xeps,
            gamma_xv=args.gamma_xv,
            absolute_actions=args.absolute_actions,
            locator=locator,
            device=device,
            plot_points=args.plot_points,
        )

    # Summary diagnostic plot. Conditional ridge match is less confounded by
    # changing arc coverage than global ridge width alone.
    xvals = [r["arc_coverage_w1"] for r in rows]
    yvals = [r["conditional_ridge_w1"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xvals, yvals, s=18)
    for r, x0, y0 in zip(rows, xvals, yvals):
        n = str(r["condition"])
        if n == "x" or "xeps_g" in n or "xv_g" in n or "absnorm" in n:
            ax.annotate(n, (x0, y0), fontsize=5)
    ax.set_xlabel("arc coverage W1 (lower is better)")
    ax.set_ylabel("conditional ridge W1 (lower is better)")
    ax.set_title("Prediction-target guidance: conditional precision vs coverage")
    fig.tight_layout()
    fig.savefig(out / "precision_coverage_v10.png", dpi=180)
    plt.close(fig)

    json_dump(out / "setting_manifest_v10.json", {
        "version": "v10_final",
        "every_timestep_recursive_guidance": True,
        "D": D, "curvature": curvature, "hidden": hidden, "seed": seed,
        "training_source": training_source,
        "sampling_execution": {
            "batch_conditions": args.batch_conditions,
            "condition_batch_size": args.condition_batch_size,
            "semantic_reference": "sample_condition_recursive_v10",
        },
        "embedding": geometry_meta,
        "metric_protocol": {
            "fixed_matched_subsets": True,
            "mmd_bandwidth_fixed_from_x": True,
            "fullD_swd_decoder_free": True,
            "fullD_swd_cached_reference_on_device": True,
            "curveD_uses_decode": False,
            "surface_audit_is_local_gauss_newton": True,
        },
        "conditions": [asdict(c) for c in conditions],
    })
    return rows


def aggregate_worker_outputs(output_root: Path) -> tuple[Path, Path]:
    worker_files = sorted(output_root.glob("worker_seed*/generation_metrics_v10_all.csv"))
    if not worker_files:
        raise FileNotFoundError(
            f"no worker_seed*/generation_metrics_v10_all.csv files under {output_root}"
        )

    group_keys = (
        "D",
        "curvature",
        "hidden",
        "training_source",
        "condition",
        "kind",
        "strength",
        "t_low",
        "t_high",
    )
    all_rows: list[dict[str, str]] = []
    expected_signatures: set[tuple[str, ...]] | None = None
    seen_seed_signatures: set[tuple[int, tuple[str, ...]]] = set()
    for worker_file in worker_files:
        rows = load_csv(worker_file)
        if not rows:
            raise ValueError(f"worker result is empty: {worker_file}")
        row_signatures = [tuple(row.get(key, "") for key in group_keys) for row in rows]
        signatures = set(row_signatures)
        if len(signatures) != len(row_signatures):
            raise ValueError(f"duplicate condition rows in worker result: {worker_file}")
        worker_seeds = {int(float(row["seed"])) for row in rows}
        if len(worker_seeds) != 1:
            raise ValueError(
                f"worker result must contain exactly one seed: {worker_file}, "
                f"found={sorted(worker_seeds)}"
            )
        worker_seed = next(iter(worker_seeds))
        if any(
            (worker_seed, signature) in seen_seed_signatures
            for signature in signatures
        ):
            raise ValueError(
                f"duplicate seed/condition rows across workers: {worker_file}, "
                f"seed={worker_seed}"
            )
        seen_seed_signatures.update((worker_seed, signature) for signature in signatures)
        if expected_signatures is None:
            expected_signatures = signatures
        elif signatures != expected_signatures:
            missing = sorted(expected_signatures - signatures)
            extra = sorted(signatures - expected_signatures)
            raise ValueError(
                f"worker condition mismatch in {worker_file}: "
                f"missing={missing[:3]!r}, extra={extra[:3]!r}"
            )
        all_rows.extend(rows)

    combined_path = output_root / "generation_metrics_v10_all_seeds.csv"
    save_csv(combined_path, all_rows)

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in all_rows:
        key = tuple(row.get(name, "") for name in group_keys)
        grouped.setdefault(key, []).append(row)

    excluded = set(group_keys) | {"seed"}
    candidate_fields: list[str] = []
    seen_fields: set[str] = set()
    for row in all_rows:
        for field in row:
            if field not in excluded and field not in seen_fields:
                seen_fields.add(field)
                candidate_fields.append(field)

    summary_rows: list[dict] = []
    for key, rows in sorted(grouped.items()):
        seeds = sorted({int(float(row["seed"])) for row in rows})
        if len(rows) != len(seeds):
            raise ValueError(
                f"duplicate seed rows for aggregate condition {key}: "
                f"rows={len(rows)}, seeds={len(seeds)}"
            )
        summary: dict[str, object] = dict(zip(group_keys, key))
        summary["seed_count"] = len(seeds)
        summary["seeds"] = ",".join(str(seed) for seed in seeds)
        for field in candidate_fields:
            values = []
            for row in rows:
                try:
                    value = float(row.get(field, "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            summary[f"{field}_mean"] = float(array.mean())
            summary[f"{field}_std"] = (
                float(array.std(ddof=1)) if len(array) > 1 else float("nan")
            )
            summary[f"{field}_sem"] = (
                float(array.std(ddof=1) / math.sqrt(len(array)))
                if len(array) > 1
                else float("nan")
            )
            summary[f"{field}_n"] = len(array)
        summary_rows.append(summary)

    summary_path = output_root / "generation_metrics_v10_seed_summary.csv"
    save_csv(summary_path, summary_rows)
    json_dump(
        output_root / "aggregate_manifest_v10.json",
        {
            "version": "v10_final",
            "worker_files": [str(path) for path in worker_files],
            "combined_rows": len(all_rows),
            "summary_rows": len(summary_rows),
            "group_keys": list(group_keys),
            "combined_csv": str(combined_path),
            "summary_csv": str(summary_path),
        },
    )
    return combined_path, summary_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prediction-target error-geometry toy v10 (modified consolidated)",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--dims", type=parse_int_list, default=parse_int_list("512"))
    p.add_argument("--curvatures", type=parse_float_list, default=parse_float_list("0,0.5"))
    p.add_argument("--hidden-dims", type=parse_int_list, default=parse_int_list("1024"))
    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260817,20260818,20260819"))

    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--frequency-scale", type=float, default=6.0)
    p.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    p.add_argument("--energy-match-curvature", action="store_true")
    p.add_argument("--energy-calibration-samples", type=int, default=32768)
    p.add_argument("--observability-samples", type=int, default=8192)
    p.add_argument("--data-jitter", type=float, default=0.015)

    p.add_argument("--training-mode", choices=("fixed", "converged"), default="converged")
    p.add_argument("--loss-space", choices=("v", "direct"), default="v")
    p.add_argument("--fixed-steps", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=150000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--scheduler", choices=("auto", "constant", "cosine"), default="auto")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--resume-training", action="store_true")
    p.add_argument("--training-checkpoint-every", type=int, default=5000)
    p.add_argument("--train-t-min", type=float, default=0.02)
    p.add_argument("--train-t-max", type=float, default=0.98)
    p.add_argument("--conversion-clip", type=float, default=0.02)

    p.add_argument("--val-times", type=parse_float_list, default=parse_float_list("0.1,0.3,0.5,0.7,0.9"))
    p.add_argument("--val-samples-per-time", type=int, default=2048)
    p.add_argument("--val-batch-size", type=int, default=1024)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--patience-evals", type=int, default=20)
    p.add_argument("--min-rel-improve", type=float, default=0.002)
    p.add_argument("--exact-bayes-probe", action="store_true")
    p.add_argument("--exact-bayes-samples", type=int, default=256)

    p.add_argument("--reuse-v4-setting", type=str, default="")

    p.add_argument(
        "--condition-suite",
        choices=("baseline", "fine", "mechanism", "paradox"),
        default="mechanism",
    )
    p.add_argument("--sample-count", type=int, default=10000)
    p.add_argument("--reference-count", type=int, default=20000)
    p.add_argument("--sample-batch-size", type=int, default=512)
    p.add_argument(
        "--batch-conditions", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--condition-batch-size", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=200)
    p.add_argument("--sample-t-min", type=float, default=0.02)
    p.add_argument("--sample-t-max", type=float, default=0.98)
    p.add_argument("--diag-stride", type=int, default=5)
    p.add_argument("--save-samples", action="store_true")
    p.add_argument("--visual-atlas", action="store_true")
    p.add_argument("--plot-points", type=int, default=4000)

    p.add_argument(
        "--gamma-xeps", type=parse_float_list,
        default=parse_float_list("-0.10,-0.075,-0.05,-0.03,-0.02,-0.01,-0.005,0,0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40"),
    )
    p.add_argument(
        "--gamma-xv", type=parse_float_list,
        default=parse_float_list("-0.5,-0.3,-0.2,-0.1,-0.05,-0.03,-0.01,0,0.01,0.03,0.05,0.1,0.2,0.3,0.5,0.75,1.0"),
    )
    p.add_argument(
        "--path-alphas", type=parse_float_list,
        default=parse_float_list("0,0.25,0.5,0.75,1,1.05,1.1,1.25,1.5,2"),
    )
    p.add_argument(
        "--absolute-actions", type=parse_float_list,
        default=parse_float_list("-0.03,-0.01,0.01,0.03"),
        help="Legacy absolute RMS-normalized action; eta=0.01 reproduces v4 xv_rms_eta0p01 semantics.",
    )
    p.add_argument(
        "--relative-actions", type=parse_float_list,
        default=parse_float_list("-0.08,-0.04,-0.02,-0.01,0.01,0.02,0.04,0.08"),
    )
    p.add_argument("--control-gammas", type=parse_float_list, default=parse_float_list("0.03,0.1,0.2"))
    p.add_argument("--geometry-gammas", type=parse_float_list, default=parse_float_list("-0.1,-0.03,0.03,0.1,0.2"))
    p.add_argument("--geometry-relative-actions", type=parse_float_list, default=parse_float_list("0.01,0.02,0.04,0.08"))
    p.add_argument(
        "--geometry-absolute-actions", type=parse_float_list,
        default=parse_float_list("0.01,0.03"),
        help="Component-wise legacy absolute RMS-normalized actions.",
    )
    p.add_argument("--stage-gammas", type=parse_float_list, default=parse_float_list("0.03,0.1,0.2"))

    p.add_argument("--ridge-grid-points", type=int, default=4096)
    p.add_argument("--coverage-bins", type=int, default=100)
    p.add_argument("--conditional-ridge-bins", type=int, default=20)
    p.add_argument("--conditional-ridge-min-count", type=int, default=20)

    # Distribution metrics use fixed matched subsets across every condition.
    p.add_argument("--swd-projections", type=int, default=256)
    p.add_argument("--swd-max-points", type=int, default=10000)
    p.add_argument("--full-swd-projections", type=int, default=64)
    p.add_argument("--full-swd-max-points", type=int, default=4096)
    p.add_argument("--mmd-max-points", type=int, default=4096)
    p.add_argument("--bootstrap-reps", type=int, default=0)
    p.add_argument("--bootstrap-max-points", type=int, default=1024)
    p.add_argument("--bootstrap-projections", type=int, default=64)

    # Decoder-free high-D embedded-curve geometry.  Expensive work is restricted
    # to a fixed matched subset rather than all endpoint samples.
    p.add_argument("--geometry-metric-points", type=int, default=512)
    p.add_argument("--curveD-coarse-points", type=int, default=512)
    p.add_argument("--curveD-refine-points", type=int, default=9)
    p.add_argument("--curveD-refine-rounds", type=int, default=2)

    # Optional local nearest-surface audit.  Zero keeps it disabled by default;
    # enable on focused runs rather than every large mechanism sweep.
    p.add_argument("--surface-audit-samples", type=int, default=0)
    p.add_argument("--surface-audit-iterations", type=int, default=6)
    p.add_argument("--surface-audit-damping", type=float, default=1e-5)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        combined, summary = aggregate_worker_outputs(args.output_root)
        print(f"Combined: {combined}", flush=True)
        print(f"Summary: {summary}", flush=True)
        return
    device = torch.device(args.device)
    if args.reuse_v4_setting and (
        len(args.dims) != 1 or len(args.curvatures) != 1
        or len(args.hidden_dims) != 1 or len(args.seeds) != 1
    ):
        raise ValueError("--reuse-v4-setting is one-setting-at-a-time")
    if args.sample_batch_size < 2 and args.control_gammas:
        raise ValueError("shuffle/Gaussian controls require sample batch size >=2")
    if args.condition_batch_size <= 0:
        raise ValueError("--condition-batch-size must be positive")
    if args.reuse_v4_setting and args.resume_training:
        raise ValueError("--resume-training only applies to v10 training, not v4 reuse")
    if args.training_checkpoint_every < 0:
        raise ValueError("--training-checkpoint-every must be non-negative")
    if args.fixed_steps <= 0 or args.max_steps <= 0 or args.val_every <= 0:
        raise ValueError("training step counts and --val-every must be positive")
    if args.swd_projections <= 0 or args.full_swd_projections <= 0:
        raise ValueError("SWD projection counts must be positive")
    if args.swd_max_points <= 0 or args.full_swd_max_points <= 0:
        raise ValueError("SWD point limits must be positive")
    if args.geometry_metric_points < 0 or args.surface_audit_samples < 0:
        raise ValueError("geometry/audit sample counts must be non-negative")
    if args.surface_audit_samples > 0 and args.geometry_metric_points <= 0:
        raise ValueError("surface audit requires --geometry-metric-points > 0")
    json_dump(args.output_root / "manifest_v10.json", {
        "version": "v10_final",
        "critical_semantics": "recursive current-state guidance at every active solver timestep",
        "legacy_absnorm_semantics": "delta = eta * gap / RMS(gap)",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })

    all_rows = []
    for seed in args.seeds:
        set_seed(seed)
        for D in args.dims:
            for curvature in args.curvatures:
                for hidden in args.hidden_dims:
                    rows = run_setting(args, D=D, curvature=curvature, hidden=hidden, seed=seed, device=device)
                    all_rows.extend(rows)
                    if device.type == "cuda": torch.cuda.empty_cache()
    save_csv(args.output_root / "generation_metrics_v10_all.csv", all_rows)
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
