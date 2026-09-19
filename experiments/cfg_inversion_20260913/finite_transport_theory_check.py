"""CPU-only exact matrix counterexample; no model, sampler, GPU or image metrics."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm, logm, sqrtm
from scipy.optimize import minimize_scalar


def run():
    target = expm(np.diag([0.7, -0.4]))
    error_generator = np.array([[0.1, 0.4], [0.4, -0.1]])
    covariance = target @ target.T
    root = sqrtm(covariance)
    rows = []
    for epsilon in (0.25, 0.5, 1.0, 1.5):
        conditional = target @ expm(epsilon * error_generator)
        reference = target @ expm(2 * epsilon * error_generator)
        ac, au = logm(conditional), logm(reference)

        def risk(weight):
            matrix = expm(au + weight * (ac - au))
            generated = matrix @ matrix.T
            return float(np.trace(covariance + generated - 2 * sqrtm(root @ generated @ root)).real)

        grid = np.linspace(-10.0, 10.0, 2001)
        values = np.array([risk(weight) for weight in grid])
        local_minima = [i for i in range(1, len(grid) - 1)
                        if values[i] <= values[i-1] and values[i] <= values[i+1]]
        candidates = [(float(values[0]), float(grid[0])), (float(values[-1]), float(grid[-1]))]
        for i in local_minima:
            fit = minimize_scalar(risk, bounds=(grid[i-1], grid[i+1]),
                                  method='bounded', options={'xatol': 1e-12})
            candidates.append((float(fit.fun), float(fit.x)))
        best_risk, best_weight = min(candidates)
        relative = np.linalg.solve(reference, conditional)
        packets = {str(n): float(np.linalg.norm(reference @ np.linalg.matrix_power(relative, n) - target))
                   for n in range(5)}
        exact_packet = conditional @ np.linalg.solve(reference, conditional)
        assert np.linalg.norm(exact_packet - target) < 1e-12
        assert max(abs(np.linalg.det(matrix) - np.linalg.det(target))
                   for matrix in (reference, conditional, exact_packet)) < 1e-12
        rows.append({'epsilon': epsilon, 'packet_map_errors_n0_to_n4': packets,
                     'cfg_w1_W2_squared': risk(1), 'cfg_w2_W2_squared': risk(2),
                     'numerical_best_cfg_weight_in_minus10_to10': best_weight,
                     'numerical_best_cfg_W2_squared_in_minus10_to10': best_risk,
                     'Ac': ac.tolist(), 'Au': au.tolist(),
                     'det_target': float(np.linalg.det(target)),
                     'det_C': float(np.linalg.det(conditional)),
                     'det_U': float(np.linalg.det(reference))})
    return {'target_matrix': target.tolist(), 'error_generator': error_generator.tolist(),
            'input_distribution': 'N(0,I_2)',
            'target_distribution': 'N(0,T T^T)',
            'assumptions': 'C=T exp(epsilon V), U=T exp(2 epsilon V); real autonomous log-matrix flows.',
            'scope': 'Exact invertible linear flows; not claimed to be ideal canonical Gaussian FM fields.',
            'optimizer_scope': 'All detected minima on a 2001-point grid over [-10,10], then bounded local refinement; numerical, not a proof over all real weights.',
            'rows': rows}


if __name__ == '__main__':
    path = Path('docs/research/cfg_inversion_20260913/finite_transport_theory_check.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(path), 'epsilon1': result['rows'][2]}, indent=2))
