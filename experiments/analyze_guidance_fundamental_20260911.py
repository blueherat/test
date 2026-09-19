"""Read live screening results and audit guidance identities on CPU.

No imports of the sampling pipeline, CUDA, or model code. Existing experiment
inputs and outputs are read only. New files are written only to this analysis's
docs/data directory. Numerical examples are mechanism checks, not FID estimates.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = Path('/home/zhoushunyu/data/eqvae/experiments')
OUT = ROOT / 'docs/data/guidance_fundamental_20260911'


def dump_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def dump_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress():
    current = EXP / 'sit_control_output_50ideas_20260910'
    csv_path = current / 'control_screen_1k/results.csv'
    # Preserve the exact CSV bytes that produced this snapshot; running jobs can
    # commit further results immediately afterwards.
    raw = csv_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    (OUT / 'screening_results_snapshot.csv').write_bytes(raw)
    states = {
        name: json.loads((EXP / name / 'status.json').read_text())
        for name in [
            'sit_fsg_followup_20ideas_20260910',
            'sit_fsg_pasted_followup_20260910',
            'sit_control_output_50ideas_20260910',
            'sit_apg_mechanism_extension_20260911',
        ]
    }
    valid = [r for r in rows if r['complete'] == 'True' and np.isfinite(float(r['fid']))]
    grouped = defaultdict(list)
    for row in valid:
        grouped[row['family']].append(row)
    apg = min(grouped['cfg_apg'], key=lambda r: float(r['fid']))
    native = min(grouped['cfg_native'], key=lambda r: float(r['fid']))
    best = []
    for family, group in grouped.items():
        row = dict(min(group, key=lambda r: float(r['fid'])))
        row.update(
            completed_in_family=len(group),
            delta_fid_apg=float(row['fid']) - float(apg['fid']),
            delta_fid_native=float(row['fid']) - float(native['fid']),
            sampling_cost_ratio_apg=float(row['sum_batch_gpu_seconds']) / float(apg['sum_batch_gpu_seconds']),
        )
        best.append(row)
    dump_csv('family_best_snapshot.csv', best)
    snapshot = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'csv_path': str(csv_path),
        'csv_sha256': hashlib.sha256(raw).hexdigest(),
        'csv_rows': len(rows),
        'valid_rows': len(valid),
        'invalid_or_incomplete_rows': len(rows) - len(valid),
        'candidate_families_fully_complete': sum(k.startswith('i') and k != 'ig_local' and len(v) == 12 for k, v in grouped.items()),
        'states': states,
        'csv_rows_equal_current_status_completed': len(rows) == states['sit_control_output_50ideas_20260910']['completed'],
        'best_candidate': min((r for r in best if r['idea_id']), key=lambda r: float(r['fid'])),
        'interpretation': 'Paired 1K screening minima; unequal tuning grids and computational costs; no independent confirmation.',
    }
    dump_json('progress_snapshot.json', snapshot)
    return snapshot


def diagnostic_analysis():
    source = ROOT / 'docs/data/sit_fsg_pasted_followup_20260910/diagnostics.csv'
    rows = list(csv.DictReader(source.open()))
    keyed = {(int(r['seed_index']), float(r['t']), r['variant']): r for r in rows}
    assert len(keyed) == 960
    seeds = sorted({key[0] for key in keyed})
    summaries, teachers, pairs_out = [], [], []
    for t in [.125, .375, .625]:
        for variant in ['unchanged', 'root1', 'root4', 'tube1', 'path1']:
            group = [keyed[(i, t, variant)] for i in seeds]
            summary = dict(t=t, variant=variant, n=len(group))
            for key in ['terminal_rms', 'path_mean_squared_rms', 'conditional_target_probability', 'null_target_probability', 'conditional_terminal_change_rms', 'null_terminal_change_rms']:
                summary[key] = float(np.mean([float(r[key]) for r in group]))
            for key in ['conditional_target_success', 'null_target_success']:
                summary[key] = float(np.mean([r[key] == 'True' for r in group]))
            summaries.append(summary)
        pairs = [(keyed[(i, t, 'unchanged')], keyed[(i, t, 'root4')]) for i in seeds]
        for before, after in pairs:
            pairs_out.append(dict(
                seed_index=before['seed_index'], label=before['label'], t=t,
                baseline_conditional_correct=before['conditional_target_success'],
                delta_rms=float(after['terminal_rms']) - float(before['terminal_rms']),
                delta_q_null=float(after['null_target_probability']) - float(before['null_target_probability']),
                delta_q_conditional=float(after['conditional_target_probability']) - float(before['conditional_target_probability']),
            ))
        for correct in [None, True, False]:
            subset = [(a, b) for a, b in pairs if correct is None or (a['conditional_target_success'] == 'True') == correct]
            row = dict(t=t, baseline_teacher_group='all' if correct is None else 'correct' if correct else 'incorrect', n=len(subset))
            for key in ['terminal_rms', 'conditional_target_probability', 'null_target_probability']:
                row['mean_delta_' + key] = float(np.mean([float(b[key]) - float(a[key]) for a, b in subset]))
            row['residual_decreased_count'] = sum(float(b['terminal_rms']) < float(a['terminal_rms']) for a, b in subset)
            row['residual_decreased_and_null_q_increased_count'] = sum(float(b['terminal_rms']) < float(a['terminal_rms']) and float(b['null_target_probability']) > float(a['null_target_probability']) for a, b in subset)
            teachers.append(row)
    dump_csv('diagnostic_group_means.csv', summaries)
    dump_csv('root4_teacher_stratification.csv', teachers)
    dump_csv('root4_paired_changes.csv', pairs_out)
    return dict(source=str(source), sha256=digest(source), rows=len(rows), unique_seeds=len(seeds), note='Teacher groups are observational strata defined before intervention. Repeated times are not independent; no causal attribution to teacher correctness or significance test.')


def geometry():
    # Linear U: inverse transport and terminal-metric gradient agree exactly.
    matrix = np.array([[2., 1.], [0., .5]])
    residual = np.ones(2)
    euclidean = matrix.T @ residual
    natural = np.linalg.solve(matrix.T @ matrix, euclidean)
    inverse = np.linalg.solve(matrix, residual)
    identity_error = float(np.max(np.abs(natural - inverse)))
    assert identity_error < 1e-12

    # Global nonlinear diffeomorphism, also the exact time-1 shear flow.
    kappa = .6
    x = np.array([.7, -.4])
    velocity_c = np.array([.8, .3])

    def U(z, horizon=1.):
        return np.array([z[0], z[1] + horizon * kappa * z[0] ** 2])

    def U_inv(z, horizon=1.):
        return np.array([z[0], z[1] - horizon * kappa * z[0] ** 2])

    anchor = x + velocity_c
    base_output = U(x)
    residual_nl = anchor - base_output
    jac = np.array([[1., 0.], [2 * kappa * x[0], 1.]])
    tangent = np.linalg.solve(jac, residual_nl)
    final = U_inv(anchor)
    initial_loss = .5 * float(residual_nl @ residual_nl)
    geodesics = []
    max_loss_error = 0.
    for s in np.linspace(0., 1., 21):
        state = U_inv(base_output + s * residual_nl)
        error = U(state) - anchor
        loss = .5 * float(error @ error)
        expected_loss = (1. - s) ** 2 * initial_loss
        max_loss_error = max(max_loss_error, abs(loss - expected_loss))
        geodesics.append(dict(s=float(s), x1=float(state[0]), x2=float(state[1]), anchored_loss=loss, predicted_anchored_loss=expected_loss))
    eps = 1e-5
    tangent_fd = (U_inv(base_output + eps * residual_nl) - U_inv(base_output - eps * residual_nl)) / (2 * eps)
    tangent_error = float(np.max(np.abs(tangent_fd - tangent)))
    assert max_loss_error < 1e-12 and tangent_error < 1e-8
    dump_csv('nonlinear_geodesic.csv', geodesics)

    local_gap = velocity_c - np.array([0., kappa * x[0] ** 2])
    horizon_rows = []
    for h in [.5, .25, .125, .0625, .03125, .015625, .0078125, .00390625]:
        delta = U_inv(x + h * velocity_c, h) - x
        horizon_rows.append(dict(horizon=h, normalized_x1=float(delta[0] / h), normalized_x2=float(delta[1] / h), local_gap_x1=float(local_gap[0]), local_gap_x2=float(local_gap[1]), error=float(np.linalg.norm(delta / h - local_gap))))
    assert all(a['error'] > b['error'] for a, b in zip(horizon_rows, horizon_rows[1:]))
    dump_csv('short_horizon_generator.csv', horizon_rows)

    # A valid finite future kernel where h_1 is constant but h_2 is not.
    # This is an abstract probability example, not a Gaussian FM benchmark.
    likelihood = np.array([0., .5, 1.])

    def future_weights(state):
        mass = .2 + .05 * np.tanh(state)
        return np.array([mass, 1. - 2. * mass, mass])

    def h_w(state, power):
        return float(future_weights(state) @ (likelihood ** power))

    grad_h1 = (np.log(h_w(eps, 1)) - np.log(h_w(-eps, 1))) / (2 * eps)
    grad_h2 = (np.log(h_w(eps, 2)) - np.log(h_w(-eps, 2))) / (2 * eps)
    assert abs(grad_h1) < 1e-10 and abs(grad_h2 - 1. / 14.) < 1e-8

    # Initial law matters when doing a terminal h-transform of a path law.
    initial = np.array([.5, .5])
    kernel = np.array([[.9, .1], [.2, .8]])
    weight = np.array([1., 3.])
    h_initial = kernel @ weight
    transformed_kernel = kernel * weight[None, :] / h_initial[:, None]
    terminal = initial @ kernel
    target_tilt = terminal * weight / (terminal @ weight)
    same_initial_terminal = initial @ transformed_kernel
    tilted_initial = initial * h_initial / (initial @ h_initial)
    corrected_terminal = tilted_initial @ transformed_kernel
    assert np.max(np.abs(corrected_terminal - target_tilt)) < 1e-12
    assert np.max(np.abs(same_initial_terminal - target_tilt)) > .1

    result = {
        'linear': dict(U_matrix=matrix.tolist(), residual=residual.tolist(), negative_euclidean_gradient=euclidean.tolist(), negative_natural_gradient=natural.tolist(), inverse_displacement=inverse.tolist(), identity_max_abs_error=identity_error, euclidean_inverse_cosine=float(euclidean @ inverse / (np.linalg.norm(euclidean) * np.linalg.norm(inverse)))),
        'nonlinear': dict(x=x.tolist(), conditional_anchor=anchor.tolist(), initial_null_output=base_output.tolist(), natural_gradient_tangent=tangent.tolist(), exact_inverse_state=final.tolist(), finite_displacement=(final - x).tolist(), loss_identity_max_abs_error=max_loss_error, tangent_finite_difference_error=tangent_error, final_anchor_loss=.5 * float(np.sum((U(final) - anchor) ** 2))),
        'moving_target_example': dict(U='identity', C='x + H * b', residual='H * b, independent of x', moving_target_loss_gradient='0', frozen_target_negative_gradient='H * b', fsg_displacement='H * b'),
        'noncommuting_future_power': dict(h1_at_zero=h_w(0., 1), h2_at_zero=h_w(0., 2), h1_squared=h_w(0., 1) ** 2, local_log_h1_gradient=float(grad_h1), true_log_h2_gradient=float(grad_h2), caveat='Abstract finite future kernel, not a diffusion image-quality experiment.'),
        'initial_normalization': dict(h_initial=h_initial.tolist(), target_terminal_tilt=target_tilt.tolist(), fixed_initial_h_transform_terminal=same_initial_terminal.tolist(), reweighted_initial_h_transform_terminal=corrected_terminal.tolist()),
        'all_checks_passed': True,
    }
    dump_json('geometry_audit.json', result)
    return result


def density_transport():
    """Independent normalized-density check in an exact compatible FM mixture."""
    nodes, weights = np.polynomial.legendre.leggauss(768)
    grid, integration_weights = 12. * nodes, 12. * weights
    mu, sigma = 2., .5

    def fields(t, x):
        q = (1. - t) ** 2 + (sigma * t) ** 2
        m = mu * t
        norm = -.5 * np.log(2. * np.pi * q)
        log_c = norm - (x - m) ** 2 / (2. * q)
        log_minus = norm - (x + m) ** 2 / (2. * q)
        log_u = np.logaddexp(log_c, log_minus) - np.log(2.)
        score_c = -(x - m) / q
        score_u = -x / q + (m / q) * np.tanh(m * x / q)
        beta = (1. - t) / t
        v_c, v_u = x / t + beta * score_c, x / t + beta * score_u
        return log_c, log_u, score_c, score_u, v_c, v_u

    def normalized(t, x, w):
        lc, lu, *_ = fields(t, grid)
        unnormalized = np.exp(w * lc + (1. - w) * lu)
        z = integration_weights @ unnormalized
        lc_x, lu_x, *_ = fields(t, x)
        return w * lc_x + (1. - w) * lu_x - np.log(z), unnormalized / z

    rows = []
    eps = 1e-5
    for t in [.2, .5, .8]:
        beta = (1. - t) / t
        for w in [1., 1.5, 3.]:
            query = np.linspace(-3., 3., 25)
            _, pi = normalized(t, query, w)
            _, _, _, _, vc_grid, vu_grid = fields(t, grid)
            expected_gap2 = integration_weights @ (pi * (vc_grid - vu_grid) ** 2)
            lc, lu, sc, su, vc, vu = fields(t, query)
            velocity = vu + w * (vc - vu)
            score_pi = su + w * (sc - su)
            log_plus, _ = normalized(t + eps, query, w)
            log_minus, _ = normalized(t - eps, query, w)
            log_plus_half, _ = normalized(t + eps / 2., query, w)
            log_minus_half, _ = normalized(t - eps / 2., query, w)
            time_coarse = (log_plus - log_minus) / (2. * eps)
            time_fine = (log_plus_half - log_minus_half) / eps
            time_derivative = (4. * time_fine - time_coarse) / 3.
            *_, vc_plus, vu_plus = fields(t, query + eps)
            *_, vc_minus, vu_minus = fields(t, query - eps)
            *_, vc_plus_half, vu_plus_half = fields(t, query + eps / 2.)
            *_, vc_minus_half, vu_minus_half = fields(t, query - eps / 2.)
            div_coarse = ((1. - w) * (vu_plus - vu_minus) + w * (vc_plus - vc_minus)) / (2. * eps)
            div_fine = ((1. - w) * (vu_plus_half - vu_minus_half) + w * (vc_plus_half - vc_minus_half)) / eps
            div_velocity = (4. * div_fine - div_coarse) / 3.
            residual = time_derivative + velocity * score_pi + div_velocity
            predicted = w * (w - 1.) / beta * ((vc - vu) ** 2 - expected_gap2)
            error = float(np.max(np.abs(residual - predicted)))
            assert error < 2e-6, (t, w, error)
            rows.append(dict(t=t, guidance_multiplier=w, expected_gap_squared=float(expected_gap2), max_identity_error=error, max_absolute_density_defect=float(np.max(np.abs(predicted)))))
    dump_csv('density_transport_identity.csv', rows)

    rotation = np.array([[0., -1.], [1., 0.]])
    assert np.max(np.abs(rotation @ rotation.T - np.eye(2))) == 0
    coupling = dict(initial_distribution='N(0, I_2)', U='identity', C='90-degree rotation', same_terminal_law=True, expected_paired_squared_residual=float(np.sum((rotation - np.eye(2)) ** 2)), terminal_wasserstein_squared=0., caveat='Demonstrates coupling dependence for arbitrary smooth flows; rotation is outside the canonical Gaussian-FM gradient-field parameterization.')
    dump_json('coupling_example.json', coupling)
    return dict(density_checks_passed=True, max_identity_error=max(r['max_identity_error'] for r in rows), coupling_example=coupling)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = progress()
    diagnostics = diagnostic_analysis()
    audit = geometry()
    density = density_transport()
    dump_json('analysis_inventory.json', dict(script=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)), diagnostic_source=diagnostics, geometry_checks_passed=audit['all_checks_passed'], density=density, outputs={p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'analysis_inventory.json'}))
    print(json.dumps(dict(captured_at=snapshot['captured_at_utc'], completed=snapshot['csv_rows'], consistent_status=snapshot['csv_rows_equal_current_status_completed'], best_candidate=snapshot['best_candidate']['arm'], best_candidate_fid=snapshot['best_candidate']['fid'], geometry_checks_passed=audit['all_checks_passed'], density_identity_error=density['max_identity_error'], output=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
