"""Read only requests, step diagnostics, and frozen sampler source; no GPU/FID."""
import time
START_WALL = time.perf_counter()
START_CPU = time.process_time()
import hashlib
import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = []
CHECKS = []


def read(path):
    raw = path.read_bytes()
    SOURCES.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    return raw


def check(name, value):
    CHECKS.append({'name': name, 'passed': bool(value)})


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def runs(indices, rows):
    result = []
    for index in indices:
        if result and index == result[-1]['last_step_index'] + 1:
            result[-1]['last_step_index'] = index
            result[-1]['last_successor_time'] = rows[index]['following']
        else:
            result.append({'first_step_index': index, 'last_step_index': index,
                           'first_current_time': rows[index]['current'],
                           'first_successor_time': rows[index]['following'],
                           'last_successor_time': rows[index]['following']})
    return result


def largest(rows, getter, absolute=False):
    values = [getter(row) for row in rows]
    index = max(range(len(values)), key=lambda i: abs(values[i]) if absolute else values[i])
    return {'value': values[index], 'step_index': index, 'successor_time': rows[index]['following']}


arms = {}
loaded = {}
for arm in ('global100', 'spatial100'):
    request = json.loads(read(ROOT / arm / 'request.json'))
    rows = [json.loads(line) for line in read(ROOT / arm / 'step_diagnostics.jsonl').splitlines()]
    frozen = request['sources']['experiments/sample_raev2_spatial_energy_balls.py']
    sampler = read(Path(frozen['path']))
    check(arm + '/frozen_sampler_sha', hashlib.sha256(sampler).hexdigest() == frozen['sha256'])
    check(arm + '/complete_indexed_100_steps', [r['step_index'] for r in rows] == list(range(100)))
    check(arm + '/request_N_B_steps', (request['sample_count'], request['batch_size'], request['num_steps']) == (1000, 8, 100))
    moments = request['clean_moments_dc_ac']
    grid = request['time_grid']
    for k, row in enumerate(rows):
        p = row['projection']
        prefix = f'{arm}/step{k:03d}/'
        check(prefix + 'successor_time', row['current'] == grid[k] and row['following'] == grid[k+1] == p['following'])
        check(prefix + 'full_cohort_shared_coefficients', p['cohort_count'] == 1000 and p['coefficients_are_shared_over_all_images_and_channels'] is True)
        check(prefix + 'forward_calls', row['stage2_forward_calls'] == 125)
        check(prefix + 'mode', p['mode'] == request['mode'])
        t = grid[k+1]
        budgets = [(1-t)**2*moments[0] + t*t/256, (1-t)**2*moments[1] + t*t*255/256]
        a = p['energy_before_dc_ac']
        if arm == 'global100':
            ideal_lambda = [min(1.0, math.sqrt(sum(budgets)/sum(a)))] * 2
        else:
            ideal_lambda = [1.0 if v == 0 else min(1.0, math.sqrt(b/v)) for v, b in zip(a, budgets)]
        check(prefix + 'budget_formula', budgets == p['bridge_budget_dc_ac'])
        check(prefix + 'ideal_lambda_formula', ideal_lambda == p['lambda_fp64'])
        applied = [fp32(v) for v in ideal_lambda]
        check(prefix + 'applied_fp32_lambda', applied == p['lambda_applied_fp32'])
        check(prefix + 'no_expansion', all(0 <= v <= 1 for v in applied))
        ideal_after = [v*l*l for v, l in zip(a, ideal_lambda)]
        after = p['energy_after_dc_ac']
        check(prefix + 'ideal_after_formula', ideal_after == p['ideal_energy_after_dc_ac'])
        check(prefix + 'residual_arithmetic',
              [v-b for v, b in zip(after, budgets)] == p['budget_residual_dc_ac']
              and [v-b for v, b in zip(after, ideal_after)] == p['rounding_energy_residual_dc_ac']
              and sum(after)-sum(budgets) == p['global_budget_residual'])
        check(prefix + 'activity_and_bypass_flags',
              p['ideal_active'] == any(v < 1 for v in ideal_lambda)
              and p['actual_applied_active'] == any(v < 1 for v in applied)
              and p['exact_all_one_bypass'] == all(v == 1 for v in applied))
        check(prefix + 'finite_nonnegative_energies', all(math.isfinite(v) and v >= 0 for v in a + after))
        if all(v == 1 for v in applied):
            check(prefix + 'all_one_energy_bypass', a == after)
    component = {}
    for j, name in enumerate(('DC', 'AC')):
        active = [k for k, row in enumerate(rows) if row['projection']['lambda_applied_fp32'][j] < 1]
        ideal_active = [k for k, row in enumerate(rows) if row['projection']['lambda_fp64'][j] < 1]
        minimum_index = min(range(100), key=lambda k: rows[k]['projection']['lambda_applied_fp32'][j])
        minimum = rows[minimum_index]['projection']['lambda_applied_fp32'][j]
        component[name] = {
            'actual_active_steps': len(active), 'actual_active_runs_inclusive_zero_based': runs(active, rows),
            'ideal_active_steps': len(ideal_active),
            'ideal_but_not_actual_active_steps': sorted(set(ideal_active)-set(active)),
            'minimum_applied_lambda': minimum, 'maximum_amplitude_contraction_percent': 100*(1-minimum),
            'maximum_contraction_step_index': minimum_index,
            'maximum_contraction_successor_time': rows[minimum_index]['following'],
            'maximum_absolute_rounding_energy_residual': largest(rows, lambda r: r['projection']['rounding_energy_residual_dc_ac'][j], True),
            'largest_postprojection_budget_residual': largest(rows, lambda r: r['projection']['budget_residual_dc_ac'][j]),
            'component_budget_is_individually_constrained': arm == 'spatial100',
        }
    final = rows[-1]['projection']
    arms[arm] = {
        'mode': request['mode'], 'step_count': len(rows), 'cohort_size': request['sample_count'],
        'forward_calls_sum': sum(row['stage2_forward_calls'] for row in rows),
        'sample_forwards': sum(row['stage2_forward_calls'] for row in rows)*request['batch_size'],
        'components': component,
        'any_component_actual_active_steps': sum(row['projection']['actual_applied_active'] for row in rows),
        'all_one_bypass_steps': sum(row['projection']['exact_all_one_bypass'] for row in rows),
        'largest_global_budget_residual': largest(rows, lambda r: r['projection']['global_budget_residual']),
        'maximum_recorded_state_rounding_rms': largest(rows, lambda r: r['projection']['float_state_residual_rms']),
        'maximum_recorded_state_rounding_abs': largest(rows, lambda r: r['projection']['float_state_residual_max_abs']),
        'endpoint_same_controlled_trajectory_pre_vs_post': {
            'successor_time': 0.0, 'budget_dc_ac': final['bridge_budget_dc_ac'],
            'preprojection_energy_dc_ac': final['energy_before_dc_ac'],
            'postprojection_energy_dc_ac': final['energy_after_dc_ac'],
            'applied_lambda_dc_ac': final['lambda_applied_fp32'],
            'preprojection_ratio_to_budget_dc_ac': [a/b for a,b in zip(final['energy_before_dc_ac'], final['bridge_budget_dc_ac'])],
            'postprojection_ratio_to_budget_dc_ac': [a/b for a,b in zip(final['energy_after_dc_ac'], final['bridge_budget_dc_ac'])],
            'postprojection_budget_residual_dc_ac': final['budget_residual_dc_ac'],
            'preprojection_total': sum(final['energy_before_dc_ac']),
            'postprojection_total': sum(final['energy_after_dc_ac']),
            'total_budget': sum(final['bridge_budget_dc_ac']),
        },
        'diagnostic_timer_sums_only': {
            'model_euler_wall_seconds': math.fsum(row['model_and_euler_wall_seconds'] for row in rows),
            'projection_diagnostics_wall_seconds': math.fsum(row['projection_and_diagnostics_wall_seconds'] for row in rows),
            'scope': 'Sums of step timer spans; excludes callbacks, surrounding trajectory overhead, loads, decoder and storage; not end-to-end time or GPU kernel busy time.'
        }
    }
    loaded[arm] = (request, rows)

check('same_requests_grid_moments_seed', all(loaded['global100'][0][key] == loaded['spatial100'][0][key]
                                          for key in ('time_grid', 'clean_moments_dc_ac', 'seed', 'sample_count', 'batch_size')))
check('same_first_actual_euler_energy_before_control', loaded['global100'][1][0]['projection']['energy_before_dc_ac'] == loaded['spatial100'][1][0]['projection']['energy_before_dc_ac'])
out = {
    'protocol': 'spatial_energy_balls_independent_step_diagnostic_review_v1',
    'passed': all(check['passed'] for check in CHECKS), 'check_count': len(CHECKS),
    'failed_checks': [check for check in CHECKS if not check['passed']], 'checks': CHECKS,
    'inputs': SOURCES, 'arms': arms,
    'source_reading': {
        'scope': 'Frozen sampler lines 115-269 read statically, never imported.',
        'full_cohort': 'cohort_energies sums FP64 spatial energies over all B8 blocks and divides by state.numel(); time_major updates all Euler successors before calling project_cohort once.',
        'actual_trajectory': 'project_cohort writes replacement into the live state tensor; next Euler step consumes that corrected tensor.',
        'float_reference': 'RMS/max residuals compare actual FP32 application against the FP64 ideal projection of the same preprojection FP32 state. This review validates stored scalar arithmetic, not a new tensor-level recomputation.',
        'metadata_qualification': 'constraint_note is the same global_ball wording in both modes. It is informative for global100 only; spatial100 has individually constrained component budgets as verified by mode-specific formulas. Frozen records were not changed.'
    },
    'interpretation_limits': [
        'energy_before is after Euler on THIS controlled trajectory, before THIS projection; it is not the independently generated official trajectory.',
        'Global100 and spatial100 have different recursive states after their first intervention. Endpoint pre/post comparisons within one arm must not be described as official-to-candidate causal effects.',
        'Raw states at all steps were not reloaded. Recorded energy/state-error measurements are producer evidence; this independent review checks their formulas, indexing, consistency and source semantics.',
        'No FID/IS/feature/evaluation files were opened or computed. No GPU, model, sampler, frozen-source edits or protocol changes.',
        'Activation intervals are retrospective diagnostics; no windows, thresholds, gains, reference changes or new candidate selected.'
    ],
    'review_cost': {'wall_seconds': time.perf_counter()-START_WALL, 'cpu_seconds': time.process_time()-START_CPU,
                   'scope': 'This CPU script from start through review computation, before JSON serialization and process exit; excludes prior interactive source/request reading.'}
}
output_path = HERE / 'review.json'
output_path.write_text(json.dumps(out, indent=2) + '\n')
manifest = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
            for path in (Path(__file__).resolve(), output_path)]
(HERE/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'passed': out['passed'], 'check_count': out['check_count'], 'failed_checks': out['failed_checks'],
                  'arms': arms, 'review_cost': out['review_cost'], 'manifest': manifest}, indent=2))
