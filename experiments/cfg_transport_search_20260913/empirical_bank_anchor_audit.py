"""CPU-only coverage audit; no ODE rollout, network, or GPU.

Run:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
 experiments/cfg_transport_search_20260913/empirical_bank_anchor_audit.py
"""
from pathlib import Path
import json
import numpy as np


DATA = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae")
OUT = Path("/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/empirical_bank_anchor")


def squared_distances(x, y):
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2 * x @ y.T, 0.0
    )


def weights(x, bank, t, clean_sigma=0.0):
    variance = (1 - t) ** 2 + t * t * clean_sigma ** 2
    logits = -squared_distances(x, t * bank) / (2 * variance)
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def summarize(w):
    return {
        "median_ess": float(np.median(1 / np.sum(w * w, axis=1))),
        "median_max_weight": float(np.median(np.max(w, axis=1))),
        "fraction_max_weight_gt_99pct": float(np.mean(w.max(axis=1) > .99)),
    }


def main():
    labels = np.load(DATA / "train_labels.npy", mmap_mode="r")
    source = np.load(DATA / "train_source_indices.npy", mmap_mode="r")
    moments = np.load(DATA / "train_moments.npy", mmap_mode="r")
    reports = []
    for label, seed in [(0, 20260913), (1, 20260914)]:
        rng = np.random.default_rng(seed)
        ids = np.flatnonzero(labels == label)
        ids = rng.permutation(ids)
        bank_ids, bank2_ids, eval_ids = ids[:512], ids[512:1024], ids[1024:1040]
        assert len(eval_ids) == 16
        all_ids = np.concatenate([bank_ids, bank2_ids, eval_ids])
        assert len(np.unique(source[all_ids])) == len(all_ids)
        array = np.array(moments[all_ids], dtype=np.float64)
        mean = array[:, :4].reshape(len(array), -1) * .18215
        std = array[:, 4:].reshape(len(array), -1) * .18215
        bank, bank2, real = mean[:512], mean[512:1024], mean[1024:]
        eps = rng.standard_normal(real.shape)
        report = {
            "class": label, "seed": seed, "dimension": int(mean.shape[1]),
            "bank_size": 512, "heldout_size": 16,
            "source_id_overlap": 0,
            "posterior_std_median": float(np.median(std)),
            "median_heldout_nearest_rms": float(np.median(
                np.sqrt(squared_distances(real, bank).min(axis=1) / mean.shape[1])
            )),
            "bridge_responsibilities": [], "smoothed_endpoint_responsibilities": [],
        }
        for t in [.05, .1, .25, .5, .75, .9, .97]:
            z = t * real + (1 - t) * eps
            w = weights(z, bank, t)
            w2 = weights(z, bank2, t)
            self_z = t * bank[:16] + (1 - t) * eps
            self_w = weights(self_z, bank, t)
            report["bridge_responsibilities"].append({
                "t": t, "heldout": summarize(w), "self": summarize(self_w),
                "self_source_posterior_median": float(np.median(np.diag(self_w[:, :16]))),
                "median_bank_posterior_mean_difference_rms": float(np.median(
                    np.sqrt(np.mean((w @ bank - w2 @ bank2) ** 2, axis=1))
                )),
            })
        for sigma in [.05, .1, .25, .5, 1.0]:
            w = weights(real, bank, 1.0, sigma)
            report["smoothed_endpoint_responsibilities"].append({
                "clean_sigma_per_coordinate": sigma,
                "smoothing_noise_norm_rms": sigma * np.sqrt(mean.shape[1]),
                **summarize(w),
            })
        reports.append(report)
    result = {
        "description": "Two classes, disjoint real-image banks and heldout IDs; posterior-mean representation. Responsibilities only: these are not inverse trajectories or a generative-quality experiment.",
        "reports": reports,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {OUT / 'coverage.json'}")


if __name__ == "__main__":
    main()
