"""Exact Gaussian witness: reference marginals do not identify guidance behavior.

This is a CPU mathematical example, not a SiT/JiT experiment or image FID.
Run with the existing myenv Python; outputs are written inside the repository.
"""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqvae_self_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/self_guidance_breakthrough_20260922"


def covariance(generator, time=1.0):
    flow = expm(time * generator)
    return flow @ flow.T


def gaussian_w2_squared(left, right):
    values, vectors = np.linalg.eigh(left)
    root = (vectors * np.sqrt(values)) @ vectors.T
    product_values = np.linalg.eigvalsh(root @ right @ root)
    return float(np.trace(left) + np.trace(right)
                 - 2 * np.sqrt(np.maximum(product_values, 0)).sum())


def ellipse(cov):
    angles = np.linspace(0, 2 * np.pi, 400)
    values, vectors = np.linalg.eigh(cov)
    return (vectors * np.sqrt(values)) @ np.stack([np.cos(angles), np.sin(angles)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = np.log(2.0)
    strong = np.diag([s, 0.0])
    b = s / 2
    rotation_generator = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotation = expm((np.pi / 6) * rotation_generator)
    # Fix the target before optimizing either model: a 30-degree rotated ellipse.
    target = rotation @ np.diag([4.0, 1.0]) @ rotation.T

    def guided(a, omega):
        weak = b * np.eye(2) + omega * rotation_generator
        return covariance((1 + a) * strong - a * weak)

    # This bounded optimum is illustrative. The impossibility of exact matching
    # with omega=0 is analytic: every time-varying scalar schedule stays diagonal.
    baseline = minimize_scalar(
        lambda a: gaussian_w2_squared(guided(a, 0), target),
        bounds=(-0.99, 5), method="bounded",
    )

    def residual(parameters):
        difference = guided(*parameters) - target
        return difference[np.triu_indices(2)]

    fitted = least_squares(
        residual, x0=[0.5, -1.0], bounds=([-0.99, -10], [5, 10]),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1000,
    )
    a, omega = fitted.x
    weak = b * np.eye(2) + omega * rotation_generator
    marginal_errors = [
        np.linalg.norm(covariance(weak, t) - np.exp(2 * b * t) * np.eye(2))
        for t in np.linspace(0, 1, 101)
    ]
    assert fitted.success
    assert max(marginal_errors) < 1e-11
    assert np.linalg.norm(guided(a, omega) - target) < 1e-11
    assert abs(target[0, 1]) > 1

    result = {
        "kind": "exact Gaussian population covariance witness; not image FID",
        "solver": "scipy.linalg.expm, continuous linear flow evaluated exactly up to roundoff",
        "target_covariance": target.tolist(),
        "strong_generator": strong.tolist(),
        "weak_isotropic_rate": float(b),
        "weak_covariance_formula": "exp(2*b*t) * I, independent of omega for all t",
        "baseline_search_a_bounds": [-0.99, 5],
        "baseline_a": float(baseline.x),
        "baseline_covariance": guided(baseline.x, 0).tolist(),
        "baseline_gaussian_W2_squared": float(baseline.fun),
        "learned_a": float(a),
        "learned_omega": float(omega),
        "learned_covariance": guided(a, omega).tolist(),
        "learned_gaussian_W2_squared_roundoff": gaussian_w2_squared(guided(a, omega), target),
        "weak_covariance_max_error_101_times": float(max(marginal_errors)),
        "analytic_obstruction": "With omega=0, any integrable scalar a(t) leaves guided covariance diagonal.",
        "limitations": [
            "Uses covariance least squares, not GAN training.",
            "No learned neural head, finite-step Heun, image data, or image quality evaluation.",
            "A general nonlinear marginal-preserving weak field remains an open implementation problem.",
        ],
    }
    (OUT / "gauge_reference_witness.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    entries = [
        ("Weak-only distributions", [
            (2 * np.eye(2), "Original weak", "#4477aa", "-"),
            (covariance(weak), "Weak + circulation", "#cc6677", "--"),
        ]),
        ("Guidance: tune scale only", [
            (target, "Target", "#333333", "--"),
            (guided(baseline.x, 0), "Best in searched range", "#4477aa", "-"),
        ]),
        ("Guidance: tune reference flow", [
            (target, "Target", "#333333", "-"),
            (guided(a, omega), "Same weak marginals", "#cc6677", "--"),
        ]),
    ]
    for ax, (title, curves) in zip(axes, entries):
        for cov, label, color, style in curves:
            points = ellipse(cov)
            ax.plot(*points, label=label, color=color, linestyle=style, linewidth=2.2)
        ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$", xlim=(-2.3, 2.3), ylim=(-2.3, 2.3))
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.legend(loc="lower center", fontsize=8)
    fig.suptitle("Identical reference marginals can produce different guided distributions", fontsize=12)
    fig.savefig(OUT / "gauge_reference_witness.png", dpi=180)
    fig.savefig(OUT / "gauge_reference_witness.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
