"""Prepare, without applying, the one-scalar directional covariance branch."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from experiments.prepare_raev2_conditional_variance_integration import replace_once

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'experiments/locks/raev2_directional_variance_20260907'
    folder.mkdir(exist_ok=False)
    sampler = ROOT/'experiments/sample_raev2_ancestral_guidance.py'
    study = ROOT/'experiments/run_raev2_ancestral_study.py'
    old = {p: p.read_text() for p in [sampler, study]}
    value = old[sampler]
    value = replace_once(value, 'from experiments.raev2_conditional_variance import ConditionalVarianceHead, NativePrefixTap',
        'from experiments.raev2_conditional_variance import ConditionalVarianceHead, NativePrefixTap\nfrom experiments.raev2_directional_variance import colored_noise, PLAN as DIRECTIONAL_PLAN')
    value = replace_once(value, "'conditional_variance_zero'])", "'conditional_variance_zero','directional_variance','directional_variance_unit'])")
    value = replace_once(value, "if any(m.startswith('conditional_variance') for m in args.modes):", "if any(m.startswith(('conditional_variance','directional_variance')) for m in args.modes):")
    value = replace_once(value, '    weak_model = SharedPrefixWeak(model)', '''    directional_calibration = None
    if any(m.startswith('directional_variance') for m in args.modes):
        directional_path = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/directional_variance_moments/calibration.json')
        directional_calibration = json.loads(directional_path.read_text())
        if not directional_calibration['complete'] or not directional_calibration['validation']['entry_condition_passed']:
            raise ValueError('directional variance requires the original held-out Gaussian NLL gate')
        if directional_calibration['plan'] != DIRECTIONAL_PLAN or directional_calibration['spherical_head_sha256'] != file_sha256(variance_path):
            raise ValueError('directional covariance or frozen spherical head identity mismatch')
        if 'directional_variance_unit' in args.modes and (args.samples != 8 or args.shards != 1):
            raise ValueError('unit covariance ratio is only an 8-image parity check')
    weak_model = SharedPrefixWeak(model)''')
    value = replace_once(value, "'experiments/raev2_conditional_variance.py']},", "'experiments/raev2_conditional_variance.py','experiments/raev2_directional_variance.py']},")
    value = replace_once(value, "    put(out/'request.json',request)", '''    if directional_calibration is not None:
        request['directional_variance'] = {'calibration_sha256': file_sha256(directional_path),
            'kappa': directional_calibration['kappa'], 'plan': DIRECTIONAL_PLAN,
            'spherical_head_sha256': file_sha256(variance_path),
            'global_fitted_parameters': 1, 'all_100_times': True,
            'no_extra_model_calls_or_input_backward': True}
    put(out/'request.json',request)''')
    value = replace_once(value, "        variance_tap=NativePrefixTap(model) if mode.startswith('conditional_variance') else None",
        "        directional_calls=0\n        variance_tap=NativePrefixTap(model) if mode.startswith(('conditional_variance','directional_variance')) else None")
    value = replace_once(value, "                if variance_tap is not None:\n                    noise=torch.randn(state.shape,device='cuda',generator=refresh)\n",
        "                if variance_tap is not None:\n                    noise=torch.randn(state.shape,device='cuda',generator=refresh)\n                    if mode.startswith('directional_variance'):\n                        noise,_=colored_noise(noise,full,base,1. if mode=='directional_variance_unit' else directional_calibration['kappa'])\n                        directional_calls+=1\n")
    value = replace_once(value, "            'variance_feature_queries':variance_calls,'sample_variance_feature_queries':variance_calls*args.batch,",
        "            'directional_covariance_queries':directional_calls,'sample_directional_covariance_queries':directional_calls*args.batch,\n            'variance_feature_queries':variance_calls,'sample_variance_feature_queries':variance_calls*args.batch,")
    updated = {sampler: value}
    value = old[study]
    value = replace_once(value, "ROOT/'experiments/raev2_conditional_variance.py',Path(__file__).resolve()", "ROOT/'experiments/raev2_conditional_variance.py',ROOT/'experiments/raev2_directional_variance.py',Path(__file__).resolve()")
    value = replace_once(value, "            'sample_variance_feature_queries':sum", "            'sample_directional_covariance_queries':sum(s.get('sample_directional_covariance_queries',0) for s in summaries),\n            'sample_variance_feature_queries':sum")
    updated[study] = value
    manifest = {'applied': False, 'requires_completed_directional_moment_parent_before_apply': True, 'files': {}}
    patch = ''
    for path, value in updated.items():
        ast.parse(value)
        relative = str(path.relative_to(ROOT))
        patch += ''.join(difflib.unified_diff(old[path].splitlines(True), value.splitlines(True), fromfile='a/'+relative, tofile='b/'+relative))
        manifest['files'][relative] = {'before_sha256': hashlib.sha256(old[path].encode()).hexdigest(), 'after_sha256': hashlib.sha256(value.encode()).hexdigest()}
        (folder/(path.name+'.pending')).write_text(value)
    (folder/'integration.patch').write_text(patch)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
