"""Paired CFG geometry screen. Noise-to-data SiT; no image feedback operation.

Every controller freezes accepted-step memory throughout both Heun queries and
commits the left-stage proposal only after accepting the step. The public
CFG-CTRL rule is ported to this convention, not claimed as a paper replication.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.guidance_pasted_20260912 import common as c

ROOT = c.EXPS / 'cfg_invariants_20260913/images'
HERE = Path(__file__).resolve()
REPORT = HERE.with_name('image_results.md')
PYTHON = c.PYTHON
FID_PYTHON = '/data/shared/envs/adm-fid/bin/python'
REFERENCE = c.EXPS.parent / 'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
CLASS_MAP = c.EXPS.parent / 'imagenet_sit_flow/imagenet100_cmc/manifest.json'
CLASSIFIER = Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')
AUTHOR = c.WORK / 'readings/fsg_pasted_audit_20260910/common_cfg_ctrl.py'
N, BATCH, CAL_N, SEED = 400, 16, 100, 2026091391
ALPHA, K, LAMBDA, LAMBDA_TIME = 1.25, .3, .05, 3.2
ARMS = ('cfg_base', 'cfg_half', 'ctrl_author', 'ctrl_gap_project',
        'ctrl_direction_matched', 'ctrl_physical', 'fixedset_projection',
        'norm_feedback', 'norm_time_mean', 'clean_evidence_proxy',
        'evidence_time_mean', 'gap_time_mean', 'fixedset_time_mean')
TRACE_COLUMNS = ('gap_rms', 'effective_parallel_extra', 'extra_to_native_norm',
                 'off_gap_extra_fraction', 'proxy_target_probability', 'used_gain')


def atomic_npz(path, **kwargs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, **kwargs)
    temp.replace(path)


def rms(x):
    return x.square().flatten(1).mean(1).sqrt()


def expand(x):
    return x[:, None, None, None]


def coefficient(vector, gap):
    return (vector * gap).flatten(1).sum(1) / gap.square().flatten(1).sum(1).clamp_min(1e-20)


def project(vector, gap):
    return expand(coefficient(vector, gap)) * gap


def fixedset(vector, gap):
    """Projection onto frozen S={a gap:0<=a<=ALPHA}; single-gap scalar only."""
    return expand(coefficient(vector, gap).clamp(0., ALPHA)) * gap


def author_gap(gap, previous):
    old = gap if previous is None else previous
    sliding = gap - old + LAMBDA * old
    return gap - K * sliding.sign()


def physical_gap(gap, previous, t, previous_t, parameter='velocity'):
    """Canonical velocity derivative with current-unit transported raw history.

    clean=a*v+z, a=1-t. The affine common z cancels in the gap. History is
    transported by a/a_prev; K is multiplied by |a|. This is also valid for
    epsilon gaps with a=-t away from the singular endpoint. Raw history and
    derivative use accepted left-stage physical times, unlike the author rule.
    """
    scale = 1. if parameter == 'velocity' else 1.-t if parameter == 'clean' else -t
    assert abs(scale) > 1e-8
    current = scale * gap
    if previous is None:
        old, derivative = current, torch.zeros_like(current)
    else:
        old_scale = 1. if parameter == 'velocity' else 1.-previous_t if parameter == 'clean' else -previous_t
        old_stored = old_scale * previous
        old = old_stored * (scale / old_scale)
        derivative = (current - old) / (t - previous_t)
    sliding = derivative + LAMBDA_TIME * old
    return (current - abs(scale) * K * sliding.sign()) / scale


def cpu_check():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(SEED)
    g = torch.randn((5, 4, 8, 8), generator=generator, dtype=torch.float64)
    old = torch.randn(g.shape, generator=generator, dtype=g.dtype)
    r = torch.randn(g.shape, generator=generator, dtype=g.dtype)
    spec = importlib.util.spec_from_file_location('invariant_author_cfgctrl', AUTHOR)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    state = module.CFGCtrlState(prev_guidance_eps=old.clone())
    params = module.CFGCtrlParams(smc_cfg_enable=True)
    actual = module.CFGCtrlMixin()._cfg_ctrl_apply(noise_pred_posi=g, noise_pred_nega=torch.zeros_like(g),
                 cfg_scale=1., progress_id=2, params=params, state=state)
    assert torch.equal(actual, author_gap(g, old))
    checks = dict(author_local_rule_exact=True,
        fixedset_idempotence_max=float((fixedset(fixedset(r, g), g)-fixedset(r, g)).abs().max()),
        gap_projection_idempotence_max=float((project(project(r, g), g)-project(r, g)).abs().max()))
    for parameter in ('clean', 'epsilon'):
        a = physical_gap(g, old, .4, .375, parameter)
        b = physical_gap(g, old, .4, .375)
        checks[parameter+'_physical_transport_max_error'] = float((a-b).abs().max())
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    assert checks['fixedset_idempotence_max'] < 1e-12
    zero = torch.zeros_like(g)
    checks['current_zero_gap_stale_author_correction_rms'] = float(rms(author_gap(zero, old)).mean())
    checks['projected_current_zero_gap_is_zero'] = bool(torch.equal(project(r, zero), zero))
    assert checks['projected_current_zero_gap_is_zero']
    checks['history_contract'] = 'freeze through Heun stages; commit left proposal once after accepted step'
    checks['cuda_used'] = False
    return checks


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    checks = cpu_check(); c.atomic(ROOT/'cpu_check.json', checks)
    for name, number, seed in [('screen', N, SEED), ('calibration', CAL_N, SEED+1)]:
        path = ROOT/(name+'_bank.npz')
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((number, 4, 32, 32), dtype=np.float32)
        labels = (np.arange(number) % 100).astype(np.int64); rng.shuffle(labels)
        if path.exists():
            with np.load(path) as data:
                np.testing.assert_array_equal(data['noise'], noise)
                np.testing.assert_array_equal(data['labels'], labels)
        else: atomic_npz(path, noise=noise, labels=labels)
    sources = [HERE, Path(c.__file__).resolve(), AUTHOR,
               c.WORK/'experiments/lifting_scale_sweep_20260909.py']
    request = dict(samples=N, batch=BATCH, calibration_samples=CAL_N, seed=SEED, arms=list(ARMS),
        model='sit_small', solver='Heun64, t=0 noise to t=1 data', active_left_time='t<.75',
        alpha=ALPHA, total_cfg_weight=1+ALPHA, ctrl_lambda=LAMBDA, ctrl_K=K,
        full_branch_forwards_per_output=224, prefix_forwards_per_output=0,
        sources={str(p):c.sha(p) for p in sources},
        assets={str(p):c.sha(p) for p in [*c.asset_paths('sit_small'), CLASSIFIER, CLASS_MAP, REFERENCE]},
        inputs={str(ROOT/(name+'_bank.npz')):c.sha(ROOT/(name+'_bank.npz')) for name in ('screen','calibration')},
        history=checks['history_contract'],
        fixedset='orthogonal projection of extra vector onto {a*current_gap:0<=a<=1.25}; scalar for one gap',
        physical='raw previous gap, physical derivative, lambda_time=3.2; canonical velocity units and transported history',
        evidence='clean conditional estimate decoded at left steps 12,24,36; ImageNet1K ConvNeXt target probability; held between reads; NOT noisy or model posterior',
        matched_controls='time means from independent 100-image candidate trajectories; norm target from independent baseline',
        snapshots='5 ordinary sampling states and guided clean predictions; no image feedback/identity experiment',
        fid_interpretation='ADM FID400 screening only; not a population-FID improvement claim')
    path = ROOT/'request.json'
    if path.exists(): assert c.read(path) == request, 'request is frozen'
    else: c.atomic(path, request)
    print(json.dumps(dict(prepared=True, root=str(ROOT), checks=checks)), flush=True)


def verify():
    request = c.read(ROOT/'request.json')
    for key in ('sources','assets','inputs'):
        for path, digest in request[key].items():
            assert c.sha(path) == digest, path
    return request


class CleanClassifier:
    def __init__(self):
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.model = convnext_tiny(weights=None).cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(CLASSIFIER, map_location='cpu', weights_only=True), strict=True)
        self.transform = self.weights.transforms()
        records = c.read(CLASS_MAP)['classes']
        self.mapping = np.array([next(r['original_imagenet_label'] for r in records if r['label']==i) for i in range(100)])

    @torch.inference_mode()
    def classify(self, pixels):
        x = torch.from_numpy(pixels).permute(0,3,1,2)
        return self.model(self.transform(x).cuda()).softmax(-1)

    def target_probability(self, pixels, labels):
        p = self.classify(pixels)
        target = torch.as_tensor(self.mapping, device='cuda')[labels]
        return p[torch.arange(len(p), device='cuda'), target]


@torch.inference_mode()
def sample(rt, classifier, noise, labels, arm, profiles=None, snapshots=False):
    rt.labels=labels; z=noise.clone(); previous=None; previous_t=None
    probability=z.new_full((len(z),), .5); trace=[]; states=[]; clean_states=[]; snap_times=[]
    before=rt.counts.copy(); extra_decoded=0
    for step,(t,u) in enumerate(zip(rt.grid[:-1],rt.grid[1:])):
        left=float(t); h=u-t; proposal=None
        def field(x, now, stage):
            nonlocal proposal, probability, extra_decoded
            cond=rt.field(x, now, 'full')
            if left >= .75: return cond
            rt.labels=torch.full_like(labels,100)
            try: uncond=rt.field(x, now, 'full')
            finally: rt.labels=labels
            gap=cond-uncond; grms=rms(gap); index=2*step+stage
            gain=x.new_full((len(x),), ALPHA)
            if arm=='cfg_half': gain=gain*.5
            if arm.startswith('ctrl_') or arm=='fixedset_projection':
                if arm=='ctrl_physical':
                    modified=physical_gap(gap,previous,float(now),previous_t)
                else: modified=author_gap(gap,previous)
                if arm=='ctrl_gap_project': modified=gap+project(modified-gap,gap)
                extra=(1+ALPHA)*modified-gap
                if arm=='fixedset_projection': extra=fixedset(extra,gap)
                if arm=='ctrl_direction_matched':
                    extra=extra*expand(ALPHA*grms/rms(extra).clamp_min(1e-12))
                if stage==0:
                    proposal=gap.detach().clone() if arm=='ctrl_physical' else modified.detach().clone()
            else:
                if arm=='norm_feedback':
                    gain=ALPHA*(float(profiles['gap_rms'][index])/grms.clamp_min(1e-8)).clamp(.25,2.)
                elif arm=='clean_evidence_proxy':
                    if stage==0 and step in (12,24,36):
                        pixels=rt.decode(x+(1-float(now))*cond)
                        probability=classifier.target_probability(pixels,labels)
                        extra_decoded+=len(x)
                    gain=2*ALPHA*(1-probability)
                elif arm.endswith('_time_mean'):
                    source={'norm_time_mean':'norm_feedback', 'evidence_time_mean':'clean_evidence_proxy',
                            'gap_time_mean':'ctrl_gap_project','fixedset_time_mean':'fixedset_projection'}[arm]
                    gain=x.new_full((len(x),),float(profiles[source][index]))
                extra=expand(gain)*gap
            parallel=coefficient(extra,gap)
            extra_rms=rms(extra)
            perp=rms(extra-expand(parallel)*gap)/extra_rms.clamp_min(1e-12)
            trace.append(torch.stack((grms,parallel,extra_rms/(ALPHA*grms).clamp_min(1e-12),
                                      perp,probability,gain),-1))
            return cond+extra
        first=field(z,t,0)
        if snapshots and step in (0,16,32,48):
            states.append(z[:8].detach().clone()); clean_states.append((z+(1-left)*first)[:8].detach().clone()); snap_times.append(left)
        second=field(z+h*first,u,1)
        z=z+(h/2)*(first+second)
        if proposal is not None: previous=proposal; previous_t=left
        if not torch.isfinite(z).all(): raise FloatingPointError((arm,step))
    if snapshots:
        states.append(z[:8]); clean_states.append(z[:8]); snap_times.append(1.)
    counts={k:rt.counts[k]-before[k] for k in before}
    assert counts==dict(full=224,prefix=0), counts
    return dict(latents=z, trace=torch.stack(trace,1).cpu().numpy(), counts=counts,
                states=states,clean_states=clean_states,times=snap_times,proxy_decodes=extra_decoded)


def load_bank(name):
    with np.load(ROOT/(name+'_bank.npz')) as data:
        return data['noise'],data['labels']


@torch.inference_mode()
def calibrate():
    verify();rt=c.runtime('sit_small');classifier=CleanClassifier();noise,labels=load_bank('calibration')
    profiles={}; costs=[]
    for arm in ('cfg_base','ctrl_gap_project','fixedset_projection','norm_feedback','clean_evidence_proxy'):
        rows=[]; begin=time.perf_counter()
        for start in range(0,CAL_N,BATCH):
            result=sample(rt,classifier,c.cuda(noise[start:start+BATCH]),c.cuda(labels[start:start+BATCH]),arm,profiles)
            rows.append(result['trace'])
        rows=np.concatenate(rows); profiles['gap_rms' if arm=='cfg_base' else arm]=rows[:,:,0 if arm=='cfg_base' else 1].mean(0)
        torch.cuda.synchronize(); costs.append(dict(arm=arm,seconds=time.perf_counter()-begin,
                                                    mean_parallel_extra=float(rows[:,:,1].mean())))
        print('calibrated',arm,costs[-1],flush=True)
    atomic_npz(ROOT/'profiles.npz',**profiles)
    c.atomic(ROOT/'calibration.json',dict(complete=True, costs=costs, samples_per_arm=CAL_N,
             generated_paths=CAL_N*len(costs),sha256=c.sha(ROOT/'profiles.npz'),request_sha256=c.sha(ROOT/'request.json')))
    c.atomic(ROOT/'runtime_sources.json',rt.sources)


def gallery(pixels,path,cols=8,labels=None,limit=64):
    pixels=pixels[:limit]; side=128; rows=(len(pixels)+cols-1)//cols
    canvas=Image.new('RGB',(cols*side,rows*(side+20)),(245,245,245)); draw=ImageDraw.Draw(canvas)
    for i,pixel in enumerate(pixels):
        x=(i%cols)*side;y=(i//cols)*(side+20)
        canvas.paste(Image.fromarray(pixel).resize((side,side)),(x,y))
        if labels is not None:draw.text((x+3,y+side+2),str(labels[i]),fill='black')
    canvas.save(path)


def collect(arm):
    root=ROOT/arm; request_hash=c.sha(ROOT/'request.json'); noise,labels=load_bank('screen')
    images=[]; latents=[]; traces=[]; costs=[]; proxies=0
    for start in range(0,N,BATCH):
        path=root/f'batch{start:04d}.npz'
        if not path.exists(): return False
        with np.load(path) as data:
            assert str(data['request_sha256'])==request_hash
            assert str(data['noise_sha256'])==c.array_sha(noise[start:start+BATCH])
            np.testing.assert_array_equal(data['labels'],labels[start:start+BATCH])
            images.append(data['pixels']);latents.append(data['latents']);traces.append(data['trace'])
            costs.append(float(data['seconds']));proxies+=int(data['proxy_decodes'])
    pixels=np.concatenate(images); latents=np.concatenate(latents); trace=np.concatenate(traces)
    assert len(pixels)==N and np.isfinite(latents).all()
    atomic_npz(root/'samples.npz',arr_0=pixels)
    atomic_npz(root/'diagnostics.npz',latents=latents,labels=labels,trace=trace,columns=np.array(TRACE_COLUMNS))
    gallery(pixels,root/'grid.png',labels=labels)
    summary=dict(complete=True,arm=arm,n=N,seconds=sum(costs),full_calls_per_output=224,
        proxy_extra_vae_decodes_per_output=proxies/N,proxy_extra_classifier_calls_per_output=proxies/N,
        trajectory_visualization_extra_vae_decodes=40,
        latent_mean=float(latents.mean()),latent_rms=float(np.sqrt(np.square(latents).mean())),
        latent_variance=float(latents.var(axis=0).mean()),
        mean_parallel_extra=float(trace[:,:,1].mean()),mean_extra_to_native_norm=float(trace[:,:,2].mean()),
        mean_off_gap_fraction=float(trace[:,:,3].mean()),
        saturation_fraction=float(((pixels==0)|(pixels==255)).mean()),
        request_sha256=request_hash,samples_sha256=c.sha(root/'samples.npz'))
    c.atomic(root/'summary.json',summary)
    print('collected',arm,summary,flush=True)
    return True


@torch.inference_mode()
def worker(rank):
    verify();rt=c.runtime('sit_small');classifier=CleanClassifier(); noise,labels=load_bank('screen')
    with np.load(ROOT/'profiles.npz') as data:profiles={k:data[k] for k in data.files}
    calibration=c.read(ROOT/'calibration.json'); assert c.sha(ROOT/'profiles.npz')==calibration['sha256']
    for arm in ARMS[rank::2]:
        root=ROOT/arm;root.mkdir(exist_ok=True)
        for start in range(0,N,BATCH):
            path=root/f'batch{start:04d}.npz'
            if path.exists():continue
            torch.cuda.synchronize();begin=time.perf_counter()
            value=sample(rt,classifier,c.cuda(noise[start:start+BATCH]),c.cuda(labels[start:start+BATCH]),
                         arm,profiles,snapshots=start==0)
            pixels=rt.decode(value['latents'])
            if start==0:
                states=torch.stack(value['states'],1);clean_states=torch.stack(value['clean_states'],1)
                snaps=rt.decode(clean_states.reshape(-1,4,32,32))
                atomic_npz(root/'trajectory_5_times.npz',states=states.cpu().numpy(),
                           clean_predictions=clean_states.cpu().numpy(),pixels=snaps.reshape(8,5,256,256,3),
                           times=np.array(value['times']),labels=labels[:8])
                gallery(snaps,root/'trajectory_grid.png',cols=5,labels=[f't={t:g}' for _ in range(8) for t in value['times']])
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            atomic_npz(path,pixels=pixels,latents=value['latents'].cpu().numpy(),trace=value['trace'],
                       seconds=seconds,labels=labels[start:start+BATCH],proxy_decodes=value['proxy_decodes'],
                       request_sha256=c.sha(ROOT/'request.json'),noise_sha256=c.array_sha(noise[start:start+BATCH]))
            c.atomic(ROOT/f'progress{rank}.json',dict(rank=rank,pid=os.getpid(),arm=arm,completed=start+len(pixels),total=N))
        assert collect(arm)
    c.atomic(ROOT/f'worker{rank}_complete.json',dict(complete=True,arms=list(ARMS[rank::2])))


@torch.inference_mode()
def classify_outputs(rank):
    verify();torch.set_num_threads(2);classifier=CleanClassifier();_,labels=load_bank('screen')
    targets=classifier.mapping[labels]
    for arm in ARMS[rank::2]:
        root=ROOT/arm
        if (root/'classifier.json').exists():continue
        pixels=np.load(root/'samples.npz')['arr_0'];prob=[];begin=time.perf_counter()
        for start in range(0,N,16):prob.append(classifier.classify(pixels[start:start+16]).cpu().numpy())
        p=np.concatenate(prob);ranks=np.argsort(-p,axis=1)
        target=p[np.arange(N),targets]
        top1=(ranks[:,0]==targets);top5=(ranks[:,:5]==targets[:,None]).any(1)
        atomic_npz(root/'classifier_probabilities.npz',target_probability=target,top1=top1,top5=top5,probabilities=p)
        c.atomic(root/'classifier.json',dict(target_top1=float(top1.mean()),target_top5=float(top5.mean()),
                  target_probability=float(target.mean()),seconds=time.perf_counter()-begin,
                  target_log_probability=float(np.log(target.clip(1e-12)).mean()),
                  classifier='ConvNeXt Tiny ImageNet1K; clean endpoint diagnostic, not ground truth quality'))
        print('classified',arm,flush=True)


def evaluate(rank):
    for arm in ARMS[rank::2]:
        root=ROOT/arm
        if (root/'fid.json').exists():continue
        command=[FID_PYTHON,str(c.WORK/'experiments/compute_adm_fid.py'),
            '--reference',str(REFERENCE),'--samples',str(root/'samples.npz'),'--batch-size','32',
            '--gpu-memory-fraction','.3','--output',str(root/'fid.json'),
            '--activations-output',str(root/'inception_activations.npz')]
        with (root/'fid.log').open('w') as log:
            subprocess.run(command,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        print('evaluated',arm,c.read(root/'fid.json')['fid'],flush=True)


def report():
    rows=[]; examples=[]
    for arm in ARMS:
        root=ROOT/arm
        if not (root/'summary.json').exists():continue
        row=c.read(root/'summary.json')
        for name in ('fid','classifier'):
            if (root/(name+'.json')).exists():row.update(c.read(root/(name+'.json')))
        rows.append(row)
        pixels=np.load(root/'samples.npz')['arr_0'][:8];examples.extend(pixels)
    output=io.StringIO();fields=['arm','fid','sfid','inception_score','target_top1','target_top5',
              'target_probability','mean_parallel_extra','mean_extra_to_native_norm','mean_off_gap_fraction',
              'latent_rms','latent_variance','saturation_fraction','seconds','full_calls_per_output',
              'proxy_extra_vae_decodes_per_output']
    writer=csv.DictWriter(output,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (ROOT/'results.csv').write_text(output.getvalue())
    if examples:gallery(np.stack(examples),ROOT/'comparison_grid.png',cols=8,
                        labels=[r['arm'] for r in rows for _ in range(8)],limit=len(examples))
    comparisons={}
    lookup={r['arm']:r for r in rows}
    for candidate,control in [('ctrl_gap_project','gap_time_mean'),('fixedset_projection','fixedset_time_mean'),
                              ('norm_feedback','norm_time_mean'),('clean_evidence_proxy','evidence_time_mean'),
                              ('ctrl_direction_matched','cfg_base'),('ctrl_author','cfg_base'),('ctrl_physical','ctrl_author')]:
        if candidate not in lookup or control not in lookup:continue
        a,b=lookup[candidate],lookup[control]
        record=dict(candidate=candidate,control=control)
        if 'fid' in a and 'fid' in b:record['fid_delta_candidate_minus_control']=a['fid']-b['fid']
        paths=[ROOT/name/'classifier_probabilities.npz' for name in (candidate,control)]
        if all(p.exists() for p in paths):
            va,vb=[dict(np.load(p)) for p in paths]
            diff=va['target_probability']-vb['target_probability'];rng=np.random.default_rng(SEED+2)
            boot=diff[rng.integers(0,N,(1000,N))].mean(1)
            record.update(target_probability_paired_difference=float(diff.mean()),
                          target_probability_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
        comparisons[candidate+'__'+control]=record
    c.atomic(ROOT/'comparisons.json',comparisons)
    lines=['# CFG 不变量：真实图像筛查', '',
       f'实际完成 {len(rows)}/{len(ARMS)} 个配置；每个 {N} 张，共同初始噪声和类别。',
       'SiT-S/2，Heun64，t=0 为噪声、t=1 为图像；guidance 在左端 t<0.75 开启。',
       '每输出固定 224 次单分支前向；CFG 默认 extra α=1.25（总权重 w=2.25），half 对照 α=.625。',
       '', '这是普通生成采样轨迹的 5 个时间点，不是回灌，未假定图片恒等。',
       '作者 CFG-CTRL 使用公开代码默认 λ=.05、K=.3，移植到 SiT velocity；Heun 两次查询冻结历史，接受后只提交左端 proposal。不是论文设置复现。',
       'ctrl_physical 使用原始 gap 的物理时间导数和 λ_time=3.2；历史按当前参数单位运输，velocity/clean/epsilon 等价由 CPU 检查。它与作者版本还存在 raw vs modified 历史区别。',
       'fixedset_projection 是当前固定集合 {a·gap:0≤a≤1.25} 的正交投影；单 gap 时仅为标量截断，不能宣称新方向。跨时间集合变化，不承诺整条轨迹幂等。',
       'ctrl_direction_matched 将相对 conditional 的额外向量范数逐样本配平至 vanilla 的 α||gap||；它用于检查方向效应。',
       'gap/fixedset/norm/evidence 的 time-mean 对照来自独立 100 张校准 bank 的各自轨迹，未用质量指标选参；只能配平时间平均增益，不能使各样本轨迹相同。',
       'clean_evidence_proxy 在 t=12/64、24/64、36/64 解码条件 clean 预测并用 ConvNeXt 读取目标类概率，gain=2α(1−q)，更新之间保持；初值 q=.5。这是 clean 图像代理，不是 noisy 后验，也不是生成模型自身后验。每输出额外 3 次 VAE 与分类器读取。',
       '', 'FID400 只用于小样本筛选，有显著样本数偏差；分类器读数也不是图像质量真值。evidence arm 的终点分类器与反馈分类器相同，不能当独立语义验证。100 张校准每类别仅 1 张，时间平均配平精度有限。需优先比较配平对照，不能仅将 guidance 变弱记为结构收益。', '',
       '|arm|FID400↓|top1↑|target p↑|mean extra ∥|extra norm/native|off-gap|',
       '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(row,key):return '—' if key not in row else f'{row[key]:.4f}'
    for row in rows:
        lines.append('|'+row['arm']+'|'+'|'.join(fmt(row,key) for key in ('fid','target_top1','target_probability',
                         'mean_parallel_extra','mean_extra_to_native_norm','mean_off_gap_fraction'))+'|')
    lines+=['',f'产物：[CSV](<{ROOT}/results.csv>)、[配平对比](<{ROOT}/comparisons.json>)、[同 seed 图集](<{ROOT}/comparison_grid.png>)。',
            '每个 arm 下有 grid.png、trajectory_grid.png、trajectory_5_times.npz、samples.npz、diagnostics.npz、FID 与 classifier 文件。',
            '运行：tmux session cfg_invariants_images_0913；GPU 0/1；旧队列未恢复。','']
    text='\n'.join(lines);REPORT.write_text(text);(ROOT/'image_results.md').write_text(text)
    c.atomic(ROOT/'status.json',dict(complete=len(rows)==len(ARMS) and all('fid' in r and 'target_top1' in r for r in rows),
              collected=len(rows),fid_evaluated=sum('fid' in r for r in rows),planned=len(ARMS)))
    print(json.dumps(dict(rows=len(rows),comparisons=comparisons)),flush=True)


def pipeline():
    verify()
    with (ROOT/'calibration.log').open('w') as log:
        subprocess.run([PYTHON,'-m','experiments.cfg_invariants_20260913.image_screen','calibrate'],
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
    for phase in ('worker','classify','evaluate'):
        processes=[]; logs=[]
        for rank in (0,1):
            log=(ROOT/f'{phase}{rank}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([PYTHON,'-m','experiments.cfg_invariants_20260913.image_screen',phase,'--rank',str(rank)],
                             env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                             stdout=log,stderr=subprocess.STDOUT,cwd=c.WORK))
        while any(process.poll() is None for process in processes):
            time.sleep(20)
        codes=[process.returncode for process in processes]
        for log in logs:log.close()
        report()
        if any(codes):raise RuntimeError((phase,codes))
    report()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','calibrate','worker','classify','evaluate','report','pipeline','cpu-check'])
    parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    if args.command=='cpu-check':print(json.dumps(cpu_check(),indent=2))
    elif args.command=='prepare':prepare()
    elif args.command=='calibrate':calibrate()
    elif args.command=='worker':worker(args.rank)
    elif args.command=='classify':classify_outputs(args.rank)
    elif args.command=='evaluate':evaluate(args.rank)
    elif args.command=='report':report()
    else:pipeline()
