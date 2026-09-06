"""Descriptive FID subsampling of frozen historical features; no GPU or sampling.

Run with CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
MKL_NUM_THREADS=4 /home/zhoushunyu/miniconda3/envs/myenv/bin/python -u <this file>.
All five disjoint class-balanced folds are retained. No FID infinity estimate.
"""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import scipy
from scipy import linalg
import torch

OUT = Path(__file__).resolve().parent
BASE = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response")
SCREEN = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/evaluation")
REFS = {
    "historical_adm": Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    "current_nanogen": Path("/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/blobs/925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac"),
}
CONDITIONS = {"source": "source", "reconstruction": "real", "full": "scale_s1p000000", "ig": "scale_s1p780000"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def eig_sqrt_trace(matrix):
    matrix = (matrix + matrix.T) * 0.5
    eig = linalg.eigvalsh(matrix, check_finite=False, driver="evr")
    tol = 1e-10 * max(float(eig[-1]), 1.0)
    if eig[0] < -tol:
        raise ArithmeticError(f"material negative eigenvalue {eig[0]}, tolerance {tol}")
    return float(np.sqrt(np.maximum(eig, 0)).sum()), float(eig[0])


def fid(features, reference):
    """Equivalent Bures trace; use n x n Gram for singular 1K covariance."""
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(0)
    a = (x - mean) / np.sqrt(x.shape[0] - 1)
    if x.shape[0] <= x.shape[1]:
        matrix = (a @ reference["sigma"]) @ a.T
        tr, min_eig = eig_sqrt_trace(matrix)
        tr_gen = float(np.square(a).sum())
    else:
        covariance = a.T @ a
        tr, min_eig = eig_sqrt_trace(reference["sqrt"] @ covariance @ reference["sqrt"])
        tr_gen = float(np.trace(covariance))
    diff = mean - reference["mu"]
    mean_term = float(diff @ diff)
    return {"fid": float(mean_term + tr_gen + reference["trace"] - 2 * tr),
            "mean_term": mean_term, "generated_covariance_trace": tr_gen,
            "bures_cross_trace": tr, "minimum_gram_eigenvalue": min_eig}


def main():
    start = time.perf_counter()
    provenance = {"gpu_used": False, "sampling_performed": False,
        "purpose": "descriptive within historical frozen feature protocol; not a performance lower bound or a changed screen decision",
        "fold_rule": "fold j = global IDs [1000*j,1000*(j+1)); all j=0..4; all branches paired",
        "mean_dtype": "float64", "covariance_ddof": 1,
        "versions": {k: importlib.metadata.version(k) for k in ["numpy", "scipy", "torch", "torch-fidelity"]},
        "script_sha256": sha(__file__), "reference_statistics": {}, "cohorts": {}}
    references = {}
    for name, p in REFS.items():
        with np.load(p) as z:
            raw_mu, raw_sigma = z["mu"], z["sigma"]
        assert raw_mu.shape == (2048,) and raw_sigma.shape == (2048, 2048)
        assert np.isfinite(raw_mu).all() and np.isfinite(raw_sigma).all()
        assert np.allclose(raw_sigma, raw_sigma.T, atol=1e-12)
        mu, sigma = raw_mu.astype(np.float64), raw_sigma.astype(np.float64)
        ev, q = linalg.eigh(sigma, check_finite=False)
        assert ev.min() > 0
        references[name] = {"mu": mu, "sigma": sigma, "sqrt": (q * np.sqrt(ev)) @ q.T, "trace": float(np.trace(sigma))}
        provenance["reference_statistics"][name] = {"path": str(p), "resolved_path": str(p.resolve()), "size_bytes": p.stat().st_size,
            "mu_sha256": arr_sha(raw_mu), "mu_dtype": str(raw_mu.dtype), "sigma_sha256": arr_sha(raw_sigma),
            "sigma_dtype": str(raw_sigma.dtype), "minimum_eigenvalue": float(ev.min())}
    arrays, source_ids = {}, []
    for seed in [20260801, 20260802]:
        root = BASE / f"n5000_seed{seed}_scales7_v1"
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["samples"] == 5000
        assert manifest["world_size"] == 4 and manifest["fid_reference"] == str(REFS["historical_adm"])
        with np.load(root / "sample_protocol.npz") as z:
            ids, labels, src = z["sample_ids"], z["labels"], z["real_source_rows"]
        assert np.array_equal(ids, np.arange(5000))
        assert np.array_equal(labels, ids % 1000)
        assert np.all(np.bincount(labels, minlength=1000) == 5)
        assert np.unique(src).size == 5000
        source_ids.append(src)
        record = {"root": str(root), "manifest_sha256": sha(root / "manifest.json"),
            "sample_protocol_sha256": sha(root / "sample_protocol.npz"), "unique_source_rows": int(np.unique(src).size),
            "all_global_ids_complete": True, "class_counts_all_five": True, "features": {}}
        for display, key in CONDITIONS.items():
            a = np.empty((5000, 2048), np.float32)
            files = []
            for rank in range(4):
                p = root / "inception" / f"{key}_rank{rank:02d}.npy"
                part = np.load(p, allow_pickle=False)
                assert part.shape == (1250, 2048) and part.dtype == np.float32
                a[rank::4] = part
                files.append({"path": str(p), "sha256": sha(p)})
            assert np.isfinite(a).all() and np.unique(a, axis=0).shape[0] == 5000
            arrays[(seed, display)] = a
            record["features"][display] = {"shape": list(a.shape), "dtype": str(a.dtype), "finite": True,
                "unique_feature_rows": 5000, "ordered_array_sha256": arr_sha(a), "files": files}
        provenance["cohorts"][str(seed)] = record
    provenance["source_rows_shared_between_seeds"] = int(np.intersect1d(*source_ids).size)
    write_json("provenance.json", provenance)
    print("preflight passed: two references, eight aligned finite feature arrays", flush=True)
    rows = []
    legacy_validation = []
    for (seed, condition), a in arrays.items():
        root = BASE / f"n5000_seed{seed}_scales7_v1"
        for ref_name, reference in references.items():
            for fold in [-1, 0, 1, 2, 3, 4]:
                x = a if fold == -1 else a[1000 * fold:1000 * (fold + 1)]
                result = fid(x, reference)
                row = {"seed": seed, "condition": condition, "reference": ref_name,
                    "n": len(x), "fold": fold, **result}
                rows.append(row)
                if fold == -1:
                    print({k: row[k] for k in ["seed", "condition", "reference", "n", "fid"]}, flush=True)
                    if ref_name == "historical_adm":
                        if condition in ["source", "reconstruction"]:
                            expected = json.loads((root / "image_baseline_metrics.json").read_text())[f"fid_{condition}_to_official"]
                        else:
                            with (root / "scale_response_metrics.csv").open() as f:
                                expected = next(float(z["fid_to_official"]) for z in csv.DictReader(f)
                                    if z["space"] == "decoded_inception" and float(z["scale"]) == (1.0 if condition == "full" else 1.78))
                        err = result["fid"] - expected
                        assert abs(err) < 1e-4, (condition, err)
                        legacy_validation.append({"seed": seed, "condition": condition, "archived_fid": expected,
                            "recomputed_float64_fid": result["fid"], "difference": err})
        with (OUT / "subsample_fid.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    cp = SCREEN / "official_feature_cache" / "official100_seed202609095-4824d1b51990ba53-inception.features.pt"
    a = torch.load(cp, map_location="cpu", weights_only=True).numpy()
    assert a.shape == (1000, 2048) and np.isfinite(a).all()
    current = {name: fid(a, ref) for name, ref in references.items()}
    # Independent, original nanogen sqrtm formula check, CPU only.
    mean, covariance = a.astype(np.float64).mean(0), np.cov(a, rowvar=False)
    ref = references["current_nanogen"]
    root, _ = linalg.sqrtm(covariance @ ref["sigma"], disp=False)
    diff = mean - ref["mu"]
    sqrtm_fid = float(diff @ diff + np.trace(covariance) + ref["trace"] - 2 * np.trace(root).real)
    assert abs(sqrtm_fid - 37.56270372429994) < 1e-6
    assert abs(current["current_nanogen"]["fid"] - sqrtm_fid) < 1e-4
    old, new = references["historical_adm"], references["current_nanogen"]
    cross, _ = eig_sqrt_trace(old["sqrt"] @ new["sigma"] @ old["sqrt"])
    d = old["mu"] - new["mu"]
    reference_distance = float(d @ d + old["trace"] + new["trace"] - 2 * cross)
    write_json("validation.json", {"legacy_5k": legacy_validation, "current_official_feature_path": str(cp),
        "current_official_feature_file_sha256": sha(cp), "current_official_fid": current,
        "current_official_original_sqrtm_fid": sqrtm_fid, "reference_to_reference_fid": reference_distance,
        "elapsed_cpu_wall_seconds": time.perf_counter() - start, "gpu_used": False, "complete": True})
    print("complete", time.perf_counter() - start, flush=True)


if __name__ == "__main__":
    main()
