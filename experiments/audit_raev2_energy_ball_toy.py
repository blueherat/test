"""Tiny CPU checks of whole-cohort energy-ball projection and its limits."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from audit_raev2_global_proximal_w2_toy import discrete_w2


def project(points, weights, budget):
    energy = float(weights @ (points*points).sum(1))
    gain = min(1., np.sqrt(budget/energy)) if energy > 0 else 1.
    return gain*points, float(gain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = np.array([[-2., 1.], [0., 3.], [3., -1.], [4., 2.]])
    p = np.array([.1, .2, .3, .4])
    budget = float(p @ (target*target).sum(1))
    source = 3*target+np.array([1., -.5])
    projected, gain = project(source, p, budget)
    old = discrete_w2(source, target, p, p)
    new = discrete_w2(projected, target, p, p)
    assert new <= old

    # Law nonexpansivity against another arbitrary distribution, not only
    # against the target. Linear programs optimize both couplings separately.
    other = target[:, ::-1]+np.array([3., 1.])
    other_projected, _ = project(other, p, budget)
    pair_old = discrete_w2(source, other, p, p)
    pair_new = discrete_w2(projected, other_projected, p, p)
    assert pair_new <= pair_old

    # Same finite cohort, different compute chunking, ONE aggregate gain.
    cohort = np.repeat(source, np.array([100, 200, 300, 400]), axis=0)
    chunk_energies = [np.sum(cohort[start:start+8]**2) for start in range(0, 1000, 8)]
    chunk_gain = min(1., np.sqrt(budget/(sum(chunk_energies)/1000)))
    assert abs(chunk_gain-gain) < 1e-14

    # A target represented exactly by a finite cohort. Whole-cohort projection
    # is identity; independently clipping each microbatch destroys its law.
    binary_target, binary_p = np.array([[0.], [2.]]), np.array([.5, .5])
    whole, whole_gain = project(binary_target, binary_p, 2.)
    separate = np.array([[0.], [np.sqrt(2.)]])
    microbatch_w2 = discrete_w2(separate, binary_target, binary_p, binary_p)
    assert whole_gain == 1. and np.array_equal(whole, binary_target)
    assert microbatch_w2 > 0

    # Exact enumeration of iid N=2 cohorts from the CORRECT target law.
    # Every empirical measure improves, while the averaged tagged-particle
    # marginal becomes wrong. No Monte Carlo or high-dimensional estimate.
    output_points, output_weights, cohort_rows = [], [], []
    for values in itertools.product((0., 2.), repeat=2):
        points = np.array(values)[:, None]
        corrected, coefficient = project(points, binary_p, 2.)
        before = discrete_w2(points, binary_target, binary_p, binary_p)
        after = discrete_w2(corrected, binary_target, binary_p, binary_p)
        assert after <= before+1e-12
        output_points.extend(corrected.tolist())
        output_weights.extend([1/8, 1/8])
        cohort_rows.append({"cohort": list(values), "gain": coefficient,
                            "empirical_w2_squared_before": before,
                            "empirical_w2_squared_after": after})
    marginal_w2 = discrete_w2(np.array(output_points), binary_target,
                              np.array(output_weights), binary_p)
    assert marginal_w2 > 0

    # Target-energy underestimation: the error is controlled by the radius
    # deficit, and can be nonzero even when the raw source is already exact.
    wrong_budget_points, _ = project(binary_target, binary_p, 1.)
    wrong_budget_w2 = discrete_w2(wrong_budget_points, binary_target, binary_p, binary_p)
    radius_error = np.sqrt(2.)-1.
    assert abs(wrong_budget_w2-radius_error**2) < 1e-12

    report = {"status": "passed", "scope": "CPU theory audit; frozen samplers unchanged",
              "non_gaussian_actual_law": {"gain": gain, "target_mean": (p@target).tolist(),
                                           "w2_squared_before": old, "w2_squared_after": new},
              "law_nonexpansivity": {"w2_squared_before": pair_old, "w2_squared_after": pair_new},
              "whole_cohort_compute_chunks": {"particles": 1000, "microbatch": 8,
                                               "gain_difference": abs(chunk_gain-gain)},
              "wrong_independent_microbatch_projection": {"whole_cohort_w2_squared": 0.,
                                                            "microbatch_w2_squared": microbatch_w2},
              "finite_particle_marginal_boundary": {"particles": 2, "cohorts": cohort_rows,
                                                      "raw_population_w2_squared": 0.,
                                                      "tagged_output_w2_squared": marginal_w2},
              "target_budget_estimation_boundary": {"true_budget": 2., "used_budget": 1.,
                                                      "w2_squared": wrong_budget_w2,
                                                      "radius_error_squared": radius_error**2}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
