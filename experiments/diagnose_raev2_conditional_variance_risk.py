"""Separate state conditioning from a new time-MSE estimate; no image selection."""
import json
from pathlib import Path
import numpy as np
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    root = DATA/'conditional_variance_features'
    execution = json.loads((root/'execution.json').read_text())
    fit = json.loads((DATA/'conditional_variance_fit/execution.json').read_text())
    assert execution['complete'] and fit['complete']
    assert sha(root/'execution.json') == fit['feature_execution_sha256']
    for split in ['train', 'validation']:
        assert sha(root/split/'metadata.npz') == execution['splits'][split]['files']['metadata.npz']
    records_path = DATA/'conditional_variance_fit/validation_records.npz'
    assert sha(records_path) == fit['validation_records_sha256']
    train, validation = [np.load(root/split/'metadata.npz') for split in ['train', 'validation']]
    records = np.load(records_path)
    calibration_path = DATA/'guided_reverse_variance/calibration.json'
    assert sha(calibration_path) == fit['calibration_sha256']
    mse0 = np.asarray([r['mse'] for r in json.loads(calibration_path.read_text())['rows']])
    # Each of the original100 query times has exactly640 training residuals.
    # These are descriptive CPU controls, never another sampling schedule.
    mse_time = np.bincount(train['query_indices'], weights=train['residual_mse'], minlength=100)/640
    np.testing.assert_array_equal(np.bincount(train['query_indices'], minlength=100), 640)
    index = validation['query_indices']
    observed_ratio = validation['residual_mse']/mse0[index]
    np.testing.assert_allclose(observed_ratio, records['residual_ratio'], rtol=0, atol=0)
    h_time = np.log(mse_time[index]/mse0[index])
    time_loss = .5*(h_time+observed_ratio*np.expm1(-h_time))
    head_loss = records['nll_change']
    def describe(loss):
        classes = loss.reshape(8, 1000).mean(0)
        mean, se = float(classes.mean()), float(classes.std(ddof=1)/np.sqrt(1000))
        return {'mean': mean, 'class_standard_error': se, 'upper_two_standard_errors': mean+2*se}
    result = {'complete': True, 'goal_achieved': False, 'source_sha256': sha(Path(__file__).resolve()),
              'feature_execution_sha256': sha(root/'execution.json'), 'checkpoint_sha256': fit['checkpoint_sha256'],
              'head_vs_old_time_mse': describe(head_loss), 'new_train_time_mse_vs_old': describe(time_loss),
              'head_vs_new_train_time_mse': describe(head_loss-time_loss),
              'time_mean_estimator': 'One identical MSE sample-mean formula at every original100 time,640 train examples each',
              'variance_time_ratio_quantiles': np.quantile(mse_time/mse0, [0, .01, .5, .99, 1]).tolist(),
              'head_checkpoint_and_sampling_plan_unchanged': True, 'new_time_baseline_not_sampled': True,
              'not_used_as_a_retroactive_gate_or_for_fid_parameter_selection': True,
              'limitation': 'Finite-sample plug-in time means, descriptive class SE; difference is not a population conditional-information proof'}
    output = ROOT/'experiments/results/raev2_guidance_20260907/conditional_variance_risk_diagnostic.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
