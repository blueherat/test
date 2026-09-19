from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import catalog as c
from experiments.sit_measure_guidance_20260912 import data
from experiments import imagenet100_sit_internal_v_head as heads
from experiments.lifting_scale_sweep_20260909 import Runtime, SMALL_CKPT, SMALL_HEAD, atomic, read, sha


def training_sources():
    return [Path(__file__).resolve(), Path(c.__file__).resolve(), Path(data.__file__).resolve(),
        Path(heads.__file__).resolve(), c.PROTOCOL, Path(__file__).resolve().parent / '__init__.py']


def prepare():
    c.ROOT.mkdir(parents=True, exist_ok=True)
    if (c.ROOT / 'training_request.json').exists():
        return verify()
    parent = read(c.MEASURE / 'training_request.json')
    for key in ('sources', 'assets', 'input_files'):
        for path, digest in parent[key].items():
            assert sha(path) == digest, path
    sources = dict(parent['sources'])
    sources.update({str(p): sha(p) for p in training_sources()})
    stops = [c.ROOT.parent / name / 'STOP_AFTER_CURRENT' for name in (
        'sit_refined_priority_20260911', 'sit_control_output_50ideas_20260910',
        'sit_apg_mechanism_extension_20260911', 'sit_broad_resume_20260912')]
    request = dict(sources=sources, assets={str(p): sha(p) for p in (SMALL_CKPT, SMALL_HEAD)},
        input_files=parent['input_files'], old_stop_markers={str(p): sha(p) for p in stops},
        methods=c.METHODS, steps=c.STEPS, batch=c.BATCH, learning_rate=c.LR,
        ema=c.EMA, adamw_betas=[.9,.999], gradient_clip=1., weight_decay=0.,
        seed=c.SEED, time_ranges=dict(ig=[.01,.5], cfg=[.01,.75]),
        precision='FP32 frozen features and readout, TF32 enabled',
        strong_frozen=True, selected_by_validation=False, created_unix=time.time())
    atomic(c.ROOT / 'training_request.json', request)
    snapshot = c.ROOT / 'training_sources'; snapshot.mkdir(exist_ok=True)
    for i, path in enumerate(sorted(sources)):
        (snapshot / f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    return request


def verify():
    request = read(c.ROOT / 'training_request.json')
    for key in ('sources', 'assets', 'input_files', 'old_stop_markers'):
        for path, digest in request[key].items():
            assert sha(path) == digest, (key, path)
    return request


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_head(rt, source):
    module = copy.deepcopy(rt.head.module if source == 'ig' else rt.model.final_layer)
    for layer in module.modules():
        layer._forward_hooks.clear(); layer._forward_pre_hooks.clear()
    return module.eval()


def amounts(source, times):
    if source == 'cfg':
        return torch.full_like(times, 1.25)[:, None, None, None]
    return torch.where(times < .25, .8 * 6 / 7, .8)[:, None, None, None]


@torch.no_grad()
def features_and_strong(rt, z, times, labels, source):
    if source == 'ig':
        f, context = heads.extract_internal_features(rt.model, z, times, labels, internal_depth=4)
        strong = heads.full_velocity_from_features(rt.model, f, context, internal_depth=4, latent_channels=4)
    else:
        strong = rt.model(z, times, labels)[:, :4]
        f, context = heads.extract_internal_features(rt.model, z, times, torch.full_like(labels,100),
            internal_depth=len(rt.model.blocks))
    return f, context, strong


def project(rt, head, f, context):
    tokens = head(f, context)
    patch = int(rt.model.x_embedder.patch_size[0])
    channels = tokens.shape[-1] // (patch * patch)
    return heads.unpatchify_channels(rt.model, tokens, channels=channels)[:, :4]


@torch.no_grad()
def validate(rt, head, pool, source):
    generator = torch.Generator(device='cuda').manual_seed(c.SEED + 200)
    sums = np.zeros(3, dtype=np.float64); count = 0
    for start in range(0, 800, 20):
        labels = torch.arange(start, start + 20, device='cuda') % 100
        clean, labels = pool.draw('native', generator, 20, labels=labels)
        eps = torch.randn(clean.shape, device='cuda', generator=generator)
        times = .01 + ((.5 if source == 'ig' else .75) - .01) * torch.rand(20, device='cuda', generator=generator)
        z = times[:,None,None,None] * clean + (1-times[:,None,None,None]) * eps
        f, context, strong = features_and_strong(rt,z,times,labels,source)
        weak = project(rt,head,f,context); target = clean-eps
        guided = strong + amounts(source,times) * (strong-weak)
        sums += np.array([float((value.double()-target.double()).square().sum()) for value in (weak,strong,guided)])
        count += target.numel()
    return dict(images=800, weak_mse=sums[0]/count, strong_mse=sums[1]/count, guided_mse=sums[2]/count)


def train(method):
    verify(); source,target_kind = method.split('_'); rh = sha(c.ROOT/'training_request.json')
    out = c.ROOT/'training'/method; out.mkdir(parents=True,exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            done=read(out/'complete.json')
            assert done['request_sha256']==rh and done['checkpoint_sha256']==sha(out/'model.pt')
            return
        if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        rt=Runtime('sit_small');torch.set_num_threads(4);torch.manual_seed(c.SEED)
        head=initial_head(rt,source).requires_grad_(True)
        ema=copy.deepcopy(head).requires_grad_(False)
        strong_hash=state_sha(rt.model)
        train_pool,heldout=data.Pool('train'),data.Pool('validation')
        before=validate(rt,ema,heldout,source)
        optimizer=torch.optim.AdamW(head.parameters(),lr=c.LR,betas=(.9,.999),weight_decay=0.)
        generator=torch.Generator(device='cuda').manual_seed(c.SEED)
        step0,elapsed0,fingerprints=0,0.,[]
        if (out/'resume.pt').exists():
            saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
            assert saved['request_sha256']==rh
            head.load_state_dict(saved['head']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['generator']);step0=saved['step'];elapsed0=saved['elapsed'];fingerprints=saved['fingerprints']
        losses=[];begin=time.perf_counter()
        for step in range(step0+1,c.STEPS+1):
            clean,labels=train_pool.draw('native',generator,c.BATCH)
            eps=torch.randn(clean.shape,device='cuda',generator=generator)
            times=.01+((.5 if source=='ig' else .75)-.01)*torch.rand(len(clean),device='cuda',generator=generator)
            dropped=torch.rand(len(clean),device='cuda',generator=generator)<.1
            if source=='ig':labels=torch.where(dropped,100,labels)
            z=times[:,None,None,None]*clean+(1-times[:,None,None,None])*eps
            f,context,strong=features_and_strong(rt,z,times,labels,source)
            truth=clean-eps;a=amounts(source,times)
            target=truth if target_kind=='native' else strong+(strong-truth)/a
            fingerprints.append(hashlib.sha256(z.cpu().numpy().tobytes()+times.cpu().numpy().tobytes()+labels.cpu().numpy().tobytes()).hexdigest())
            optimizer.zero_grad(set_to_none=True)
            value=project(rt,head,f,context)
            loss=(value-target).square().mean();assert torch.isfinite(loss)
            if step==1:
                if target_kind=='direct':
                    alternate=((strong+a*(strong-value)-truth)/a).square().mean()
                    torch.testing.assert_close(loss,alternate,rtol=1e-6,atol=1e-7)
                assert not f.requires_grad and not strong.requires_grad
            loss.backward();assert all(p.grad is not None for p in head.parameters())
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),head.parameters()):ep.lerp_(p,1-c.EMA)
            losses.append(float(loss.detach()));elapsed=elapsed0+time.perf_counter()-begin
            if step==1 or step%50==0:
                record=dict(phase='training',method=method,step=step,total=c.STEPS,loss=float(np.mean(losses[-50:])),
                    gradient_norm=float(norm),elapsed_seconds=elapsed,pid=os.getpid())
                atomic(out/'status.json',record);print(record,flush=True)
            stopping=(c.ROOT/'STOP_AFTER_CURRENT').exists()
            if step%250==0 or stopping:
                temp=out/'resume.tmp.pt'
                torch.save(dict(head=head.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                    generator=generator.get_state(),step=step,elapsed=elapsed,fingerprints=fingerprints,request_sha256=rh),temp)
                temp.replace(out/'resume.pt')
            if stopping:
                atomic(out/'status.json',dict(phase='paused',step=step));return
        torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
        assert all(p.grad is None for p in rt.model.parameters()) and state_sha(rt.model)==strong_hash
        after=validate(rt,ema,heldout,source);assert before['strong_mse']==after['strong_mse']
        temp=out/'model.tmp.pt'
        torch.save(dict(ema={k:v.cpu() for k,v in ema.state_dict().items()},step=c.STEPS,method=method,request_sha256=rh),temp)
        temp.replace(out/'model.pt');atomic(out/'input_fingerprints.json',fingerprints)
        done=dict(passed=True,method=method,step=c.STEPS,training_seconds=elapsed,
            trainable_parameters=sum(p.numel() for p in head.parameters()),validation_before=before,validation_after=after,
            strong_state_sha256=strong_hash,strong_unchanged=True,strong_gradients_absent=True,
            checkpoint_sha256=sha(out/'model.pt'),request_sha256=rh,input_fingerprints_sha256=sha(out/'input_fingerprints.json'),
            selected_by_validation=False)
        atomic(out/'complete.json',done);atomic(out/'status.json',dict(phase='complete',**done));print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--train',choices=c.METHODS)
    args=parser.parse_args()
    if args.prepare:prepare()
    else:train(args.train)
