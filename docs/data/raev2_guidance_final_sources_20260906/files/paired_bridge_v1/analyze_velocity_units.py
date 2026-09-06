"""Post-hoc descriptive conversion from normalized labels to bridge velocity.

No model, GPU, sampler decision, new time selection or confidence interval.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

P = Path(__file__).resolve().parent

def artifact(path):
    payload = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}

def main():
    start, cpu = time.perf_counter(), time.process_time()
    output = P/'analysis_velocity_units_v1'
    if output.exists():
        raise FileExistsError('analysis already exists; do not overwrite')
    path = P/'validate/teacher_regression.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10000
    assert {int(row['step_index']) for row in rows} == set(range(100))
    for index in range(100):
        assert sum(int(row['step_index']) == index for row in rows) == 100
    results = []
    for name in ('candidate_tau0.0', 'candidate_tau0.5', 'candidate_tau1.0', 'control_tau0.0'):
        def mean(key):
            return math.fsum(float(row['beta'])**2*float(row[name+'_'+key]) for row in rows)/len(rows)
        target, prediction, cross = (mean(key) for key in ('target_energy', 'prediction_energy', 'cross'))
        gain = 2*cross-prediction
        assert all(math.isfinite(x) for x in (target, prediction, cross, gain))
        results.append({'teacher_query': name, 'target_energy': target, 'prediction_energy': prediction,
                        'cross': cross, 'risk_gain_over_zero': gain,
                        'relative_risk_gain_percent': 100*gain/target})
    value = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': artifact(path), 'script': artifact(Path(__file__)),
        'rows': len(rows), 'time_points': 100, 'records_per_time': 100,
        'scope': 'Post-hoc descriptive units conversion, after fixed screen launch but before any screen FID. Does not modify its frozen formula, weights, time coverage or cost rule.',
        'formula': 'w_hat=beta*f, R=beta*normalized_target. Multiply stored target_energy, prediction_energy and cross by beta^2 per row, then mean all rows.',
        'rounding_boundary': 'Reconstructed from stored FP32 reductions and beta scalars; not a new raw-tensor recomputation of R.',
        'interpretation': 'All four physical bridge-velocity risks remain worse than zero prediction. This is a teacher-interpolant regression diagnostic, not actual deployment-midpoint label risk, KL or FID.',
        'no_time_or_class_selection': True, 'no_independence_or_CI_claim': True,
        'model_calls': 0, 'gpu_calls': 0, 'results': results,
        'wall_seconds_before_write': time.perf_counter()-start,
        'cpu_seconds_before_write': time.process_time()-cpu}
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(value, ensure_ascii=False))

if __name__ == '__main__':
    main()
