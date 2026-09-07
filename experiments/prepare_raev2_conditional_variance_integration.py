"""Prepare a reviewable patch; never mutate a sampler used by live studies."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    folder = ROOT/'experiments/locks/raev2_conditional_variance_20260907'
    folder.mkdir(exist_ok=False)
    sampler = ROOT/'experiments/sample_raev2_ancestral_guidance.py'
    study = ROOT/'experiments/run_raev2_ancestral_study.py'
    old = {p: p.read_text() for p in [sampler, study]}
    value = old[sampler]
    value = replace_once(value, 'from experiments.raev2_prefix_ratio_guidance import PrefixRatioHead',
        'from experiments.raev2_prefix_ratio_guidance import PrefixRatioHead\nfrom experiments.raev2_conditional_variance import ConditionalVarianceHead, NativePrefixTap')
    value = replace_once(value, "'actual_ratio','prefix_ratio64k'])", "'actual_ratio','prefix_ratio64k','conditional_variance','conditional_variance_zero'])")
    value = replace_once(value, '    weak_model = SharedPrefixWeak(model)', '''    variance_head = None
    if any(m.startswith('conditional_variance') for m in args.modes):
        variance_path = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/conditional_variance_fit/head.pt')
        variance_fit = json.loads((variance_path.parent/'execution.json').read_text())
        if args.steps != 100 or not variance_fit['complete'] or not variance_fit['optimizer_converged']:
            raise ValueError('conditional variance requires the fixed complete 100-step experiment')
        if file_sha256(variance_path) != variance_fit['checkpoint_sha256']:
            raise ValueError('conditional variance checkpoint identity mismatch')
        if 'conditional_variance_zero' in args.modes and (args.samples != 8 or args.shards != 1):
            raise ValueError('zero head is only an 8-image implementation parity check')
        variance_head = ConditionalVarianceHead(variance_path).cuda().eval().requires_grad_(False)
        if variance_head.calibration_sha256 != file_sha256(args.variance_calibration):
            raise ValueError('conditional variance calibration identity mismatch')
    weak_model = SharedPrefixWeak(model)''')
    value = replace_once(value, "if 'calibrated' in args.modes or critic is not None:",
                         "if 'calibrated' in args.modes or critic is not None or variance_head is not None:")
    value = replace_once(value, "'experiments/raev2_prefix_ratio_features.py']},", "'experiments/raev2_prefix_ratio_features.py','experiments/raev2_conditional_variance.py']},")
    value = replace_once(value, "    put(out/'request.json',request)", '''    if variance_head is not None:
        request['conditional_variance'] = {'checkpoint_sha256': file_sha256(variance_path),
            'plan': variance_head.plan, 'held_out_validation': variance_head.validation,
            'calibration_sha256': variance_head.calibration_sha256,
            'formula': 'native Euler + q sqrt(mse0(t) exp(h(native features))) xi',
            'all_100_times': True, 'no_additional_model_calls_or_input_backward': True,
            'feature_precision': 'native BF16 autocast block7; FP32 pooled statistics',
            'head_precision': 'FP64 normalization and fixed weights, FP32 final noise coefficient',
            'zero_head_allowed_only_for_8_image_parity': True}
    put(out/'request.json',request)''')
    value = replace_once(value, '        prefix_calls=0\n', '        prefix_calls=0\n        variance_calls=0\n        variance_tap=NativePrefixTap(model) if mode.startswith(\'conditional_variance\') else None\n')
    value = replace_once(value, "                if mode=='two_mode':\n", "                variance_features=variance_tap.take() if variance_tap is not None else None\n                if mode=='two_mode':\n")
    value = replace_once(value, "                if mode=='calibrated':\n", '''                if variance_tap is not None:
                    noise=torch.randn(state.shape,device='cuda',generator=refresh)
                    euler=state-(t-s)*((state-clean)/t)
                    coefficient,log_ratio=variance_head.noise_coefficient(variance_features,
                        calibration['rows'][step]['mse'],(t-s)/t,zero=mode=='conditional_variance_zero')
                    state=euler+coefficient[:,None,None,None]*noise
                    variance_calls+=1
                    if batch_id==batch_ids[0]:
                        critic_records.append({'step':step,'time':t,'mean_log_variance_ratio':float(log_ratio.mean()),
                            'min_log_variance_ratio':float(log_ratio.min()),'max_log_variance_ratio':float(log_ratio.max())})
                elif mode=='calibrated':
''')
    value = replace_once(value, '        images=np.concatenate(all_images)', '''        if variance_tap is not None:
            assert variance_tap.calls == variance_calls == len(batch_ids)*args.steps
            variance_tap.close()
        images=np.concatenate(all_images)''')
    value = replace_once(value, "            'initial_noise':records,'max_memory_allocated':torch.cuda.max_memory_allocated()})",
        "            'variance_feature_queries':variance_calls,'sample_variance_feature_queries':variance_calls*args.batch,\n            'initial_noise':records,'max_memory_allocated':torch.cuda.max_memory_allocated()})")
    updated = {sampler: value}
    value = old[study]
    value = replace_once(value, "ROOT/'experiments/raev2_prefix_ratio_features.py',Path(__file__).resolve()", "ROOT/'experiments/raev2_prefix_ratio_features.py',ROOT/'experiments/raev2_conditional_variance.py',Path(__file__).resolve()")
    value = replace_once(value, "            'paired_noise_labels_sha256':hashlib.sha256", "            'sample_variance_feature_queries':sum(s.get('sample_variance_feature_queries',0) for s in summaries),\n            'paired_noise_labels_sha256':hashlib.sha256")
    updated[study] = value
    manifest = {'applied': False, 'requires_completed_legacy_and_feature_parent_before_apply': True,
                'files': {}, 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    patch = ''
    for path, value in updated.items():
        ast.parse(value)
        relative = str(path.relative_to(ROOT))
        patch += ''.join(difflib.unified_diff(old[path].splitlines(True), value.splitlines(True), fromfile='a/'+relative, tofile='b/'+relative))
        manifest['files'][relative] = {'before_sha256': hashlib.sha256(old[path].encode()).hexdigest(),
                                      'after_sha256': hashlib.sha256(value.encode()).hexdigest()}
        (folder/(path.name+'.pending')).write_text(value)
    (folder/'integration.patch').write_text(patch)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
