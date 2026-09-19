"""Independent review checks. Never changes the training run or its sources."""

import argparse
import hashlib
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
DATA = Path('/home/zhoushunyu/data/eqvae/experiments')
RUN = DATA / 'adversarial_weak_training_20260915/endpoint_gan_v1'
OUT = WORK / 'docs/data/adversarial_weak_audit_20260915'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, result):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def cpu():
    import numpy as np
    import torch
    from experiments.adversarial_guidance_endpoint_20260915.validate import validate
    from experiments.adversarial_guidance_endpoint_20260915.objectives import guided_generator_loss
    torch.set_num_threads(2)
    result = {'existing_validation_rerun': validate()}
    request = json.loads((RUN / 'request.json').read_text())
    result['frozen_sources_match'] = all(sha(p) == h for p, h in request['sources'].items())
    result['snapshot_sources'] = len(request['sources'])
    deps = [WORK / 'experiments/guidance_dynamic_50k_20260915' / (n + '.py')
            for n in ('models', 'sampling', 'data', 'endpoints', 'config')]
    deps += [WORK / 'experiments/advfd_cleanroom/feature_extractors.py',
             WORK / 'experiments/guidance_distribution_20260912/local_head.py']
    result['core_dependencies_absent_from_run_snapshot'] = [str(p) for p in deps if str(p) not in request['sources']]
    result['dependency_hashes_at_audit'] = {str(p): sha(p) for p in deps}
    state = torch.load(RUN / 'checkpoint_000200.pt', map_location='cpu', weights_only=False)
    initial = torch.load(request['initial_head'], map_location='cpu', weights_only=False)['ema']
    result['checkpoint_sha_matches_latest'] = sha(RUN / 'checkpoint_000200.pt') == json.loads((RUN / 'latest.json').read_text())['checkpoint_sha256']
    result['checkpoint_step'] = state['step']
    result['optimizer_w_steps'] = sorted(set(int(v['step']) for v in state['optimizer_w']['state'].values()))
    result['optimizer_d_steps'] = sorted(set(int(v['step']) for v in state['optimizer_d']['state'].values()))
    result['head_changes'] = {}
    for kind in ('head', 'ema'):
        keys = [k for k in initial if k.endswith(('.weight', '.bias'))]
        diff = torch.cat([(state[kind][k] - initial[k]).flatten() for k in keys])
        base = torch.cat([initial[k].flatten() for k in keys])
        result['head_changes'][kind] = dict(l2=float(diff.norm()), relative_l2=float(diff.norm()/base.norm()),
                                          max_abs=float(diff.abs().max()))
    result['head_buffers_identical_to_initial'] = all(torch.equal(state['head'][k], v) for k, v in initial.items() if k.endswith(('_mean', '_std')) or k == 'positions')
    # Analytic one-step countercheck: x = 2*S-W, a real-preferring critic wants x to rise.
    # The gradient must DECREASE W under gradient descent, including the AG minus sign.
    weak = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    x = 2. * torch.tensor(.2) - weak
    logits = torch.stack((x, x*0, x*0, -x))[None]
    loss = guided_generator_loss(logits)
    gradient, = torch.autograd.grad(loss, weak)
    after_x = 2. * .2 - float(weak.detach() - .01 * gradient)
    result['analytic_generator_sign'] = dict(dloss_dweak=float(gradient), endpoint_increases=after_x > float(x.detach()))
    assert gradient > 0 and after_x > float(x.detach())
    rows = [json.loads(l) for l in (RUN / 'train.jsonl').read_text().splitlines()]
    norms = [v['head_gradient_norm'] for v in rows if v['generator_updated']]
    result['training'] = dict(rows=len(rows), weak_updates=len(norms), clipping_count=sum(v>1 for v in norms),
                              gradient_min=min(norms), gradient_median=float(np.median(norms)), gradient_max=max(norms))
    # Document deterministic future-path defects without launching or mutating evaluation.
    result['progress_phase'] = json.loads((RUN / 'progress.json').read_text())['phase']
    result['complete_marker'] = json.loads((RUN / 'complete.json').read_text())['complete']
    labels = np.load(DATA / 'guidance_dynamic_50k_20260915/sit_small/data/train_labels.npy')
    quality = np.load(DATA / 'guidance_dynamic_50k_20260915/sit_small/quality_inputs/labels.npy')
    result['data'] = dict(train_samples=len(labels), train_classes=len(np.unique(labels)),
                          eval_1k_class_counts=np.unique(np.bincount(quality[:1000])).tolist())
    save('cpu.json', result)


def gpu(strict_fp32=False, math_attention=False):
    # One presently idle device, under the same advisory lease as the experiment.
    import fcntl
    from experiments.weak_reference_loss_20260914 import idle
    lease = None
    for row in idle.gpu_snapshot():
        if not idle.eligible(row):
            continue
        candidate = Path('/tmp', f"eqvae_idle_{row['uuid']}.lock").open('a')
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close()
            continue
        current = {v['uuid']: v for v in idle.gpu_snapshot()}
        if row['uuid'] in current and idle.eligible(current[row['uuid']]):
            lease = candidate
            break
        candidate.close()
    if lease is None:
        raise RuntimeError('No idle leased GPU; no running job touched.')
    os.environ['CUDA_VISIBLE_DEVICES'] = row['uuid']
    import torch
    from torch.utils.checkpoint import checkpoint
    from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
    from experiments.guidance_dynamic_50k_20260915.sampling import integrate
    from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import recomputed_rollout
    from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps
    from experiments.adversarial_guidance_endpoint_20260915.objectives import guided_generator_loss
    from experiments.adversarial_weak_training_20260915.critic import SourceCritic
    from experiments.adversarial_weak_training_20260915.data import decoded
    from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    torch.set_num_threads(2)
    torch.manual_seed(2026091599)
    adapter = Adapter('sit_small')
    if strict_fp32:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if math_attention:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    state = torch.load(RUN / 'checkpoint_000200.pt', map_location='cpu', weights_only=False)
    head = adapter.loaded_head('real').requires_grad_(True)
    head.load_state_dict(state['head'])
    critic = SourceCritic().cuda().eval().requires_grad_(False)
    critic.load_state_dict(state['critic'])
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    modules = dict(strong=adapter.model, vae=adapter.rt.vae, head=head, critic=critic, inception=feature)
    before = {k: fingerprint(v) for k, v in modules.items()}
    noise = torch.randn(1,4,32,32,device='cuda').requires_grad_(True)
    labels = torch.tensor([17],device='cuda')
    steps = GuidanceSteps(adapter,head,labels,1.,method='real')
    parameters = tuple(head.parameters())
    def terminal(z):
        return guided_generator_loss(critic(feature(decoded(adapter,z)),labels))
    # Independent ordinary chain rule, using PyTorch activation checkpointing
    # solely for memory. No custom backward or truncated path in the reference.
    z = noise
    for i in range(64):
        z = checkpoint(lambda x, i=i: steps(i,x), z, use_reentrant=False)
    reference = z.detach().clone()
    loss_ref = terminal(z)
    grad_ref = torch.autograd.grad(loss_ref,(noise,*parameters))
    adapter.values.clear()
    z = recomputed_rollout(steps,64,noise,parameters)
    loss_actual = terminal(z)
    grad_actual = torch.autograd.grad(loss_actual,(noise,*parameters))
    flat_ref = torch.cat([v.flatten() for v in grad_ref[1:]])
    flat_actual = torch.cat([v.flatten() for v in grad_actual[1:]])
    error = float((flat_actual-flat_ref).norm()/flat_ref.norm())
    torch.testing.assert_close(z,reference,rtol=0,atol=0)
    with torch.no_grad():
        deployed,_ = integrate(adapter,head,'real',1.,noise.detach(),labels)
        torch.testing.assert_close(z,deployed,rtol=0,atol=0)
        pixels = decoded(adapter,z)
        f_unclamped = feature(pixels)
        f_clamped = feature(pixels.clamp(0,1))
        clip_stats = dict(fraction=float(((pixels<0)|(pixels>1)).float().mean()),
                         pixel_min=float(pixels.min()),pixel_max=float(pixels.max()),
                         feature_relative_change=float((f_unclamped-f_clamped).norm()/f_unclamped.norm()),
                         loss_continuous=float(guided_generator_loss(critic(f_unclamped,labels))),
                         loss_clamped=float(guided_generator_loss(critic(f_clamped,labels))))
    after = {k: fingerprint(v) for k,v in modules.items()}
    assert before == after
    assert not any(p.grad is not None for m in modules.values() for p in m.parameters())
    result = dict(kind='real_model_full_64_step_gradient_audit_not_quality_trial',
                  gpu=row['uuid'], checkpoint_sha256=sha(RUN/'checkpoint_000200.pt'),
                  head_weights='online_head', discriminator='trained_200_step_critic',
                  coefficient=1., batch=1, seed=2026091599, class_label=17,
                  strict_fp32=strict_fp32,
                  math_attention=math_attention,
                  full64_checkpoint_reference_relative_gradient_error=error,
                  head_gradient_cosine=float(torch.nn.functional.cosine_similarity(flat_actual,flat_ref,dim=0)),
                  gradient_relative_error_under_2e_minus4=error<2e-4,
                  initial_noise_gradient_relative_error=float((grad_actual[0]-grad_ref[0]).norm()/grad_ref[0].norm()),
                  head_gradient_norm=float(flat_actual.norm()),
                  forward_bitwise_equal=True, deployed_forward_bitwise_equal=True,
                  loss=float(loss_actual.detach()), no_module_changed=True, fingerprints=after,
                  pixel_clamp_single_sample=clip_stats,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    filename = 'gpu_math_attention.json' if math_attention else 'gpu_strict_fp32.json' if strict_fp32 else 'gpu.json'
    save(filename,result)
    lease.close()
    if math_attention:
        assert error < 2e-4, 'Full-chain math-attention derivative check failed; see saved result.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['cpu','gpu'])
    parser.add_argument('--strict-fp32',action='store_true')
    parser.add_argument('--math-attention',action='store_true')
    args = parser.parse_args()
    if args.mode == 'cpu':
        cpu()
    else:
        gpu(args.strict_fp32,args.math_attention)
