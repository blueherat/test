"""Fixed convex posterior probe and independent exponential-tilt certificates.

Feature choice, unit-L2 normalization and unit Gaussian weight prior are fixed
before seeing validation outputs. AUC is diagnostic. Acceptance would use D
itself; this module neither selects generated images nor evaluates FID.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def validate_features(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("expected a finite nonempty feature matrix")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("probe requires the fixed unit-L2 features")
    return values


def map_objective_and_gradient(parameters, features, labels):
    weight, bias = parameters[:-1], parameters[-1]
    logits = features @ weight + bias
    residual = expit(logits)-labels
    objective = np.logaddexp(0, logits).sum()-np.dot(labels, logits)+0.5*np.dot(weight, weight)
    gradient = np.r_[features.T @ residual+weight, residual.sum()]
    return float(objective), gradient


def fit_posterior_probe(real_train, fake_train):
    real, fake = validate_features(real_train), validate_features(fake_train)
    if real.shape != fake.shape:
        raise ValueError("the fixed posterior training bank must be balanced with matching shapes")
    features = np.concatenate([real, fake])
    labels = np.r_[np.ones(len(real)), np.zeros(len(fake))]
    initial = np.zeros(features.shape[1]+1, dtype=np.float64)
    result = minimize(map_objective_and_gradient, initial, args=(features, labels), jac=True,
                      method="L-BFGS-B", options={"maxiter": 2000, "maxls": 50,
                                                 "ftol": 1e-15, "gtol": 1e-8, "maxcor": 100})
    objective, gradient = map_objective_and_gradient(result.x, features, labels)
    gradient_inf = float(np.abs(gradient).max())
    if not result.success or gradient_inf > 1e-5:
        raise RuntimeError(f"fixed MAP solver did not converge: {result.message}; gradient_inf={gradient_inf}")
    weight, bias = result.x[:-1].copy(), float(result.x[-1])
    log_d_fake = log_expit(fake @ weight+bias)
    train_m = float(np.exp(log_d_fake).mean())
    if not 0 < train_m < 1:
        raise FloatingPointError("invalid fixed training acceptance normalization")
    return {"weight": weight, "bias": bias, "train_m": train_m,
            "objective": objective, "gradient_inf": gradient_inf,
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "optimizer_message": str(result.message), "train_samples_per_domain": len(real),
            "training_objective": "sum BCE(real=1,fake=0)+0.5*||weight||^2; bias unpenalized",
            "prior": "weight~N(0,I); unit-L2 feature implies prior logit variance 1",
            "hyperparameter_selection": "none; prior and architecture fixed before validation"}


def probe_probabilities(features, fit):
    values = validate_features(features)
    logits = values @ fit["weight"]+fit["bias"]
    log_d = log_expit(logits)
    return {"logits": logits, "log_D": log_d, "D": np.exp(log_d)}


def certificate_observations(real_test, fake_test, fit):
    real, fake = probe_probabilities(real_test, fit), probe_probabilities(fake_test, fit)
    if len(real["D"]) != len(fake["D"]):
        raise ValueError("class-paired certificate requires equal domain sizes")
    m = fit["train_m"]
    if not 0 < m < 1:
        raise ValueError("normalizer must be frozen from the training fake bank")
    norm = float(np.linalg.norm(fit["weight"]))
    log_d_min = float(log_expit(fit["bias"]-norm))
    log_d_max = float(log_expit(fit["bias"]+norm))
    d_min, d_max = math.exp(log_d_min), math.exp(log_d_max)
    log_m = math.log(m)
    # Per-class rows preserve the dependence of real/fake contrast and their
    # fixed uniform class prior; f=log(D)-log(m) changes no accepted target.
    values = fake["D"]/m-1-real["log_D"]+log_m
    lower = d_min/m-1-log_d_max+log_m
    upper = d_max/m-1-log_d_min+log_m
    if not np.isfinite(values).all() or np.any(values < lower-1e-10) or np.any(values > upper+1e-10):
        raise FloatingPointError("certificate violates the analytic unit-feature bounds")
    return {"certificate": values, "real": real, "fake": fake,
            "bounds": {"weight_norm": norm, "logit_min": fit["bias"]-norm,
                       "logit_max": fit["bias"]+norm, "D_min": d_min, "D_max": d_max,
                       "log_D_min": log_d_min, "log_D_max": log_d_max,
                       "certificate_min": lower, "certificate_max": upper}}


def summarize_certificate(observations, *, alpha=0.05):
    values = observations["certificate"]
    if not 0 < alpha < 1 or len(values) < 2:
        raise ValueError("invalid confidence level or sample count")
    mean = float(values.mean())
    se = float(values.std(ddof=1)/math.sqrt(len(values)))
    width = observations["bounds"]["certificate_max"]-observations["bounds"]["certificate_min"]
    hoeffding_upper = mean+width*math.sqrt(math.log(1/alpha)/(2*len(values)))
    log_factor = math.log(2/alpha)
    eb_upper = (mean+math.sqrt(2*float(values.var(ddof=1))*log_factor/len(values))
                +7*width*log_factor/(3*(len(values)-1)))
    return {"samples": len(values), "C_mean": mean, "C_standard_error": se,
            "normal_approximation_ci95": [mean-1.96*se, mean+1.96*se],
            "normal_approximation_is_diagnostic": True,
            "hoeffding_one_sided_alpha": alpha, "hoeffding_upper": hoeffding_upper,
            "hoeffding_pass": hoeffding_upper < 0, "analytic_bounds": observations["bounds"],
            "empirical_bernstein_upper": eb_upper, "empirical_bernstein_alpha": alpha,
            "formal_gate": "preselected empirical Bernstein only; no minimum across bounds",
            "formal_pass": bool(width > 0 and eb_upper < 0),
            "empirical_bernstein_source": "Maurer & Pontil 2009 Theorem 11, independent non-identically distributed bounded variables; https://www.cs.mcgill.ca/~colt2009/papers/012.pdf",
            "confidence_assumption": "independent class-stratified real/generated samples and critic+normalizer fixed independently of validation",
            "conclusion_boundary": "C upper<0 certifies a forward-KL reduction for the accepted target under support/finite-KL assumptions; no FID or effect-size guarantee"}
