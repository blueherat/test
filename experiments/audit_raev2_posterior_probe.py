#!/usr/bin/env python3
"""Fit the one fixed posterior probe, freeze it, then audit independent C/q067.

This runner has no model-selection, rejection-sampling, or FID code. The sole
formal gate is the preselected one-sided empirical Bernstein upper bound on
C(f), with analytic feature/prior-induced bounds and training-only normalizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_proximal_calibration import atomic_json, atomic_torch_save, sha256_file, write_csv
from experiments.raev2_posterior_probe import (
    certificate_observations, fit_posterior_probe, probe_probabilities, summarize_certificate,
)

PROTOCOL = "raev2_fixed_cls_posterior_acceptance_risk_audit_v1"
DEFAULT_FEATURES = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/posterior_cls_features_v1")


def load_verified(directory, split, summary, request_sha):
    path = directory / f"{split}.pt"
    if sha256_file(path) != summary["splits"][split]["sha256"]:
        raise ValueError(f"feature archive changed: {split}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["split"] != split or payload["features"].dtype != torch.float64
            or payload["features"].shape != (1000, 1024)
            or payload["request_sha256"] != request_sha
            or not torch.equal(payload["ids"], torch.arange(1000))
            or not torch.equal(payload["labels"], torch.arange(1000))):
        raise ValueError(f"feature split does not match the fixed protocol: {split}")
    return payload


def classification_diagnostics(real, fake):
    labels = np.r_[np.ones(len(real["D"])), np.zeros(len(fake["D"]))]
    scores = np.r_[real["logits"], fake["logits"]]
    # Binary NLL uses real -log D and fake -log(1-D).
    nll = .5*(-real["log_D"].mean()+np.logaddexp(0, fake["logits"]).mean())
    return {"auc": float(roc_auc_score(labels, scores)), "balanced_binary_nll": float(nll),
            "real_D_mean": float(real["D"].mean()), "fake_D_mean": float(fake["D"].mean()),
            "auc_is_diagnostic_only": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    directory, out = args.features.resolve(), args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((directory / "summary.json").read_text())
    feature_request = json.loads((directory / "request.json").read_text())
    if not summary.get("complete") or summary["samples_per_split"] != 1000:
        raise ValueError("all four feature banks must be complete")
    feature_request_sha = sha256_file(directory / "request.json")
    started = time.perf_counter()
    request = {"protocol": PROTOCOL, "features": str(directory), "features_request_sha256": feature_request_sha,
               "features_summary_sha256": sha256_file(directory / "summary.json"),
               "feature_split_sha256": {key: value["sha256"] for key, value in summary["splits"].items()},
               "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                   Path(__file__), ROOT / "experiments/raev2_posterior_probe.py")},
               "feature": feature_request["feature"], "feature_normalization": feature_request["feature_normalization"],
               "probe": "single linear sigmoid on all 1024 fixed unit-L2 CLS coordinates; no class embedding",
               "training_objective": "sum balanced BCE + 0.5*||w||^2; unpenalized bias",
               "prior": "N(0,I), unit feature gives unit prior variance of w dot feature",
               "optimizer": "deterministic FP64 L-BFGS-B, numerical convergence only",
               "model_selection": "none; no layer, rank, normalization, prior, architecture, temperature, or time-window search",
               "normalizer": "m=mean_q066_train D fixed before any test score",
               "certificate": "C=mean_qtest[D/m-1]-mean_ptest[log(D/m)]; upper bounds joint forward-KL change of class-preserving accept-D target",
               "formal_gate": "Maurer-Pontil2009 Theorem11 empirical Bernstein one-sided alpha0.05; no selection among bounds",
               "test_statistical_unit": "1000 independent class-paired realC/generatedq067 observations; classes may differ in distribution",
               "acceptance_rule_if_future_deployed": "accept each official proposal with probability D; preserve requested class; no sampler implemented here",
               "fid_read_or_computed": False, "rejection_sampling_performed": False,
               "status": "fitting_train_only"}
    (out / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    (out / "probe_helper_source.py").write_bytes((ROOT / "experiments/raev2_posterior_probe.py").read_bytes())
    atomic_json(out / "request.json", request)
    real_train = load_verified(directory, "real_train", summary, feature_request_sha)
    fake_train = load_verified(directory, "fake_train", summary, feature_request_sha)
    fit = fit_posterior_probe(real_train["features"].numpy(), fake_train["features"].numpy())
    fit_payload = {**{key: value for key, value in fit.items() if key != "weight"},
                   "weight": torch.from_numpy(fit["weight"]), "request": request}
    atomic_torch_save(out / "probe.pt", fit_payload)
    freeze = {"probe_sha256": sha256_file(out / "probe.pt"), "frozen_before_loading_test_features": True,
              "frozen_utc": datetime.now(timezone.utc).isoformat(), "train_m": fit["train_m"],
              "weight_norm": float(np.linalg.norm(fit["weight"])), "bias": fit["bias"],
              "objective": fit["objective"], "gradient_inf": fit["gradient_inf"], "iterations": fit["iterations"]}
    atomic_json(out / "model_freeze.json", freeze)
    print(json.dumps({"status": "train_frozen", **freeze}), flush=True)
    real_test = load_verified(directory, "real_test", summary, feature_request_sha)
    fake_test = load_verified(directory, "fake_test", summary, feature_request_sha)
    if np.intersect1d(real_train["source_rows"].numpy(), real_test["source_rows"].numpy()).size:
        raise ValueError("real train/test rows overlap")
    observations = certificate_observations(real_test["features"].numpy(), fake_test["features"].numpy(), fit)
    metrics = summarize_certificate(observations)
    metrics["test_classifier"] = classification_diagnostics(observations["real"], observations["fake"])
    metrics["train_classifier"] = classification_diagnostics(
        probe_probabilities(real_train["features"].numpy(), fit), probe_probabilities(fake_train["features"].numpy(), fit))
    metrics["DV_plugin_diagnostic"] = float(observations["real"]["log_D"].mean()-np.log(observations["fake"]["D"].mean()))
    metrics["DV_plugin_is_not_formal_gate"] = "log empirical fake mean is biased; formal gate uses only the unbiased C observations"
    rows = [{"sample_id": i, "label": i, "real_source_row": int(real_test["source_rows"][i]),
             "C": float(observations["certificate"][i]),
             "real_logit": float(observations["real"]["logits"][i]), "real_log_D": float(observations["real"]["log_D"][i]),
             "real_D": float(observations["real"]["D"][i]), "fake_logit": float(observations["fake"]["logits"][i]),
             "fake_log_D": float(observations["fake"]["log_D"][i]), "fake_D": float(observations["fake"]["D"][i])}
            for i in range(1000)]
    write_csv(out / "heldout_per_class.csv", rows)
    result = {"protocol": PROTOCOL, "complete": True, "probe": freeze, **metrics,
              "fid_read_or_computed": False, "rejection_sampling_performed": False,
              "elapsed_seconds": time.perf_counter()-started}
    atomic_json(out / "summary.json", result)
    atomic_json(out / "progress.json", {"status": "complete", "complete": True, "formal_pass": result["formal_pass"]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
