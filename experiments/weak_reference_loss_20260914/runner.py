"""Training, endpoint weighting, sampling, and evaluation for the frozen protocol."""
import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import config as k
from .objectives import excess_weights, smoothing_pair, weighted_mse
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local


def runtime():
    c.ROOT = k.ROOT
    return c.runtime(k.MODEL)


def tensor_fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)
    k.atomic(path.with_suffix('.json'), dict(sha256=k.sha(path), request_sha256=k.sha(k.ROOT / 'request.json')))


def load_torch(path):
    meta = k.read(Path(path).with_suffix('.json'))
    assert k.sha(path) == meta['sha256']
    assert meta['request_sha256'] == k.sha(k.ROOT / 'request.json')
    return torch.load(path, map_location='cpu', weights_only=False)


def make_head(rt, old=False):
    torch.manual_seed(k.SEED)
    head = local.make_head(rt)
    state = torch.load(k.OLD / 'head.pt', map_location='cpu', weights_only=True)['ema']['context']
    if old:
        head.load_state_dict(state, strict=True)
    else:
        with torch.no_grad():
            for name in ('token_mean', 'token_std', 'condition_mean', 'condition_std', 'positions'):
                getattr(head, name).copy_(state[name])
    return head


def trained_head(rt, arm):
    if arm == 'context':
        return make_head(rt, old=True).eval().requires_grad_(False)
    arm = arm.removesuffix('_half')
    out = k.ROOT / 'training' / arm
    done = k.read(out / 'complete.json')
    assert done['complete'] and done['head_sha256'] == k.sha(out / 'head.pt')
    state = load_torch(out / 'head.pt')
    assert state['step'] == k.STEPS
    head = make_head(rt)
    head.load_state_dict(state['ema'], strict=True)
    return head.eval().requires_grad_(False)


class CallCounter:
    def __init__(self, rt, head=None):
        self.blocks = [0] * len(rt.model.blocks)
        self.head = 0
        self.handles = []
        for index, block in enumerate(rt.model.blocks):
            def mark(module, args, index=index):
                self.blocks[index] += 1
            self.handles.append(block.register_forward_pre_hook(mark))
        if head is not None:
            def mark_head(module, args):
                self.head += 1
            self.handles.append(head.register_forward_pre_hook(mark_head))

    def close(self):
        for handle in self.handles:
            handle.remove()


def field(rt, head, capture, arm, z, t, left, factor=1.):
    factor *= .5 if arm.endswith('_half') else 1.
    amount = c.amount(rt, left, 'ig', factor)
    if not amount:
        return rt.field(z, t, 'full')
    if arm == 'native':
        full, weak = rt.pair(z, t)
    else:
        full = rt.field(z, t, 'full')
        weak = local.unpatchify(rt, head(capture.values['context'], capture.values['condition'])).float()
    return full + amount * (full - weak)


@torch.inference_mode()
def gpu_preflight():
    k.verify()
    rt = runtime()
    head = make_head(rt, old=True).eval().requires_grad_(False)
    rng = torch.Generator(device='cuda').manual_seed(k.SEED + 3)
    z = torch.randn((2, 4, 32, 32), device='cuda', generator=rng)
    y = torch.tensor([0, 1], device='cuda')
    t = torch.tensor(.47, device='cuda')
    rt.labels = y
    full = rt.field(z, t, 'full')
    capture = local.Capture(rt)
    full_capture, _ = rt.pair(z, t)
    assert torch.equal(full, full_capture)
    retained = {key: value.clone() for key, value in capture.values.items()}
    direct = local.features(rt, z, t.expand(len(z)), y)
    assert all(torch.equal(retained[key], direct[key]) for key in direct)
    for arm, old_kind in [('native', 'native_base'), ('context', 'context_base')]:
        counter = CallCounter(rt, head if arm == 'context' else None)
        actual, counts = c.integrate(rt, z, y, lambda x, time_, left, i, j:
            field(rt, head, capture, arm, x, time_, left))
        assert counts == dict(full=128, prefix=0) and counter.blocks == [128] * len(rt.model.blocks)
        assert counter.head == (64 if arm == 'context' else 0)
        counter.close()
        expected, _ = c.integrate(rt, z, y, lambda x, time_, left, i, j:
            local.field(rt, {'context': head}, capture, old_kind, x, time_, left))
        assert torch.equal(actual, expected), arm
    zero, _ = c.integrate(rt, z, y, lambda x, time_, left, i, j:
        field(rt, head, capture, 'context', x, time_, left, factor=0.))
    native_zero, _ = c.integrate(rt, z, y, lambda x, time_, left, i, j: rt.field(x, time_, 'full'))
    assert torch.equal(zero, native_zero)
    capture.close()
    assert all(not p.requires_grad for p in rt.model.parameters())
    k.atomic(k.ROOT / 'gpu_preflight.json', dict(passed=True, shared_features_exact=True,
        native_and_context_trajectories_exact=True, zero_guidance_exact=True, full_calls=128,
        prefix_calls=0, new_head_calls=64, strong_fingerprint=tensor_fingerprint(rt.model),
        runtime_sources=rt.sources, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        offline_check_paths=12, request_sha256=k.sha(k.ROOT / 'request.json')))
    print('GPU preflight passed', flush=True)


def train(arm):
    request = k.verify()
    assert k.read(k.ROOT / 'gpu_preflight.json')['passed']
    spec = k.SPECS[arm]
    root = k.ROOT / 'training' / arm
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        done = k.read(root / 'complete.json')
        assert done['complete'] and done['head_sha256'] == k.sha(root / 'head.pt')
        return
    rt = runtime()
    strong_before = tensor_fingerprint(rt.model)
    head = make_head(rt)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=request['lr'], weight_decay=request['weight_decay'],
                                  betas=tuple(request['adam_betas']))
    values = torch.from_numpy(np.load(k.DATA / f"{spec['dataset']}_clean.npy")).cuda()
    weights = torch.ones((100, 20), device='cuda')
    if spec['weights']:
        receipt = k.read(k.ROOT / 'weights/complete.json')
        assert receipt['complete'] and not receipt['degenerate']
        path = k.ROOT / f"weights/{spec['weights']}.npy"
        assert k.sha(path) == receipt['files'][str(path)]
        weights = torch.from_numpy(np.load(path)).cuda()
    generator = torch.Generator(device='cuda').manual_seed(k.SEED)
    history, elapsed, first = [], 0., 1
    latest = root / 'latest.pt'
    if latest.exists():
        state = load_torch(latest)
        assert state['arm'] == arm
        head.load_state_dict(state['online'], strict=True)
        ema.load_state_dict(state['ema'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['data_rng'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        first = state['step'] + 1
        history, elapsed = state['history'], state['training_seconds']
    torch.cuda.synchronize()
    begin = time.perf_counter()
    last = first - 1

    def checkpoint(step):
        torch.cuda.synchronize()
        value = dict(arm=arm, step=step, online=head.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), data_rng=generator.get_state(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(), history=history,
            training_seconds=elapsed + time.perf_counter() - begin)
        save_torch(latest, value)

    try:
        for step in range(first, k.STEPS + 1):
            k.check_stop()
            labels = torch.randint(100, (k.BATCH,), device='cuda', generator=generator)
            ids = torch.randint(20, (k.BATCH,), device='cuda', generator=generator)
            clean = values[labels, ids]
            t = .01 + .98 * torch.rand(k.BATCH, device='cuda', generator=generator)
            noise = torch.randn(clean.shape, device='cuda', generator=generator)
            coin = torch.rand(k.BATCH, device='cuda', generator=generator)
            smooth = coin < spec['smooth_probability']
            z, target = smoothing_pair(clean, t, noise, request['tau'], smooth)
            feats = local.features(rt, z, t, labels)
            prediction = head(feats['context'], feats['condition'])
            loss = weighted_mse(prediction, local.patchify(rt, target), weights[labels, ids])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step()
            with torch.no_grad():
                for averaged, online in zip(ema.parameters(), head.parameters()):
                    averaged.lerp_(online, 1 - request['ema'])
            last = step
            if step == 1 or step % 100 == 0:
                row = dict(step=step, loss=float(loss.detach()), smoothed_examples=int(smooth.sum()))
                history.append(row)
                k.atomic(root / 'progress.json', dict(pid=os.getpid(), arm=arm, **row))
                print(arm, row, flush=True)
            if step % 500 == 0:
                checkpoint(step)
    except k.RequestedStop:
        checkpoint(last)
        raise
    checkpoint(k.STEPS)
    assert tensor_fingerprint(rt.model) == strong_before
    assert all(p.grad is None for p in rt.model.parameters())
    final = dict(step=k.STEPS, arm=arm, ema=ema.state_dict())
    save_torch(root / 'head.pt', final)
    k.atomic(root / 'complete.json', dict(complete=True, arm=arm, steps=k.STEPS,
        head_sha256=k.sha(root / 'head.pt'), request_sha256=k.sha(k.ROOT / 'request.json'),
        training_seconds=load_torch(latest)['training_seconds'],
        history=history, strong_fingerprint_before=strong_before, strong_unchanged=True,
        parameters=sum(p.numel() for p in head.parameters()), data_source=spec['dataset']))
    print(arm, 'training complete', flush=True)


@torch.inference_mode()
def endpoint_features():
    k.verify()
    assert k.read(k.ROOT / 'gpu_preflight.json')['passed']
    out = k.ROOT / 'weights'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'features_complete.json').exists():
        record = k.read(out / 'features_complete.json')
        assert all(k.sha(path) == digest for path, digest in record['files'].items())
        return
    rt = runtime()
    all_features, all_source, all_labels, all_ids = [], [], [], []
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for source, name in enumerate(('real', 'strong')):
        values = np.load(k.DATA / f'{name}_clean.npy').reshape(2000, 4, 32, 32)
        labels = np.repeat(np.arange(100), 20)
        for start in range(0, len(values), 32):
            k.check_stop()
            x = c.cuda(values[start:start+32])
            y = c.cuda(labels[start:start+32])
            t = torch.full((len(x),), .99, device='cuda')
            feats = local.features(rt, x, t, y)['context'].float()
            vector = torch.cat([feats.mean(1), feats.std(1, correction=0)], 1)
            all_features.append(vector.cpu().numpy())
            all_source.extend([source] * len(x))
            all_labels.extend(labels[start:start+len(x)].tolist())
            all_ids.extend((np.arange(start, start+len(x)) % 20).tolist())
    torch.cuda.synchronize()
    path = out / 'endpoint_features.npz'
    np.savez(path, features=np.concatenate(all_features), source=np.array(all_source),
        labels=np.array(all_labels), endpoint_index=np.array(all_ids))
    k.atomic(out / 'features_complete.json', dict(complete=True, samples=4000,
        seconds=time.perf_counter()-start_time, request_sha256=k.sha(k.ROOT / 'request.json'),
        files={str(path):k.sha(path)}, feature_definition=k.read(k.ROOT / 'request.json')['source_classifier']['features']))


def fit_weights():
    k.verify()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    out = k.ROOT / 'weights'
    path = out / 'endpoint_features.npz'
    assert k.read(out / 'features_complete.json')['files'][str(path)] == k.sha(path)
    if (out / 'complete.json').exists():
        done = k.read(out / 'complete.json')
        assert all(k.sha(path) == digest for path, digest in done['files'].items())
        return
    with np.load(path) as data:
        x, source, labels, ids = [data[key] for key in ('features','source','labels','endpoint_index')]
    probability = np.full(len(x), np.nan)
    records = []
    start_time = time.perf_counter()
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    for fold in (0, 1):
        valid = ids % 2 == fold
        train = ~valid
        assert all(np.sum(train & (source == s) & (labels == c_)) == 10 for s in (0,1) for c_ in range(100))
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=1., max_iter=2000, random_state=k.SEED+40)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(scaler.transform(x[train]), source[train])
        probability[valid] = model.predict_proba(scaler.transform(x[valid]))[:, 1]
        np.savez(out / f'classifier_fold{fold}.npz', coef=model.coef_, intercept=model.intercept_,
            mean=scaler.mean_, scale=scaler.scale_, training_ids=np.flatnonzero(train), validation_ids=np.flatnonzero(valid))
        records.append(dict(fold=fold, heldout=len(x[valid]), auc=float(roc_auc_score(source[valid],probability[valid])),
            log_loss=float(log_loss(source[valid],probability[valid])), iterations=int(model.n_iter_[0])))
    assert np.isfinite(probability).all()
    generated = source == 1
    raw = excess_weights(torch.from_numpy(probability[generated]), 1.).numpy().reshape(100,20)
    weights = (raw / raw.mean(axis=1, keepdims=True)).astype(np.float32)
    shuffled = weights.copy()
    rng = np.random.default_rng(k.SEED+41)
    for row in shuffled:
        rng.shuffle(row)
    assert all(np.array_equal(np.sort(a), np.sort(b)) for a,b in zip(weights,shuffled))
    for name, array in [('probability',probability),('raw_excess',raw),('excess',weights),('shuffled',shuffled)]:
        np.save(out / (name+'.npy'), array)
    files = [path, *out.glob('*.npy'), *out.glob('classifier_fold*.npz')]
    k.atomic(out / 'complete.json', dict(complete=True, degenerate=bool(weights.std()<1e-3),
        normalized_weight_std=float(weights.std()), raw_weight_min=float(raw.min()), raw_weight_max=float(raw.max()),
        raw_range_valid=bool(raw.min()>=1 and raw.max()<=2), cross_fitted=True, feature_ratio_only=True,
        folds=records, seconds=time.perf_counter()-start_time, files={str(p):k.sha(p) for p in files},
        request_sha256=k.sha(k.ROOT / 'request.json')))
    print('Offline endpoint weights complete', records, 'weight std', weights.std(), flush=True)


def prepare_bank(stage, samples, seed):
    out = k.ROOT / k.MODEL / stage / 'inputs'
    if (out / 'complete.json').exists():
        record = k.read(out / 'complete.json')
        assert record['samples'] == samples and record['seed'] == seed
        assert all(k.sha(path)==digest for path,digest in record['files'].items())
        return out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((samples,4,32,32), dtype=np.float32)
    labels = np.arange(samples, dtype=np.int64) % 100
    rng.shuffle(labels)
    np.save(out/'noise.npy', noise)
    np.save(out/'labels.npy', labels)
    k.atomic(out/'complete.json', dict(samples=samples, seed=seed, files={str(out/name):k.sha(out/name) for name in ('noise.npy','labels.npy')}))
    return out


@torch.inference_mode()
def sample(arm, stage='screen1000'):
    k.verify()
    n, seed = (1000,k.SCREEN_SEED) if stage=='screen1000' else (5000,k.CONFIRM_SEED)
    bank = prepare_bank(stage,n,seed)
    root = k.ROOT/k.MODEL/stage/arm
    if (root/'summary.json').exists():
        collect(arm,stage,n,bank)
        return
    rt = runtime()
    head = None if arm=='native' else trained_head(rt,arm)
    capture = local.Capture(rt)
    noise, labels = np.load(bank/'noise.npy',mmap_mode='r'),np.load(bank/'labels.npy')
    rh = k.sha(k.ROOT/'request.json')
    for start in range(0,n,rt.batch):
        k.check_stop()
        path = root/'batches'/f'{start:05d}.npz'
        if path.exists():
            meta = k.read(path.with_suffix('.json'))
            assert meta['sha256']==k.sha(path) and meta['request_sha256']==rh
            continue
        x,y=c.cuda(noise[start:start+rt.batch]),c.cuda(labels[start:start+rt.batch])
        counter=CallCounter(rt,head)
        torch.cuda.synchronize()
        begin=time.perf_counter()
        latent,counts=c.integrate(rt,x,y,lambda z,t,left,i,j:field(rt,head,capture,arm,z,t,left))
        assert counts==dict(full=128,prefix=0) and counter.blocks==[128]*len(rt.model.blocks)
        assert counter.head==(0 if arm=='native' else 64)
        pixels=rt.decode(latent)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        c.save_batch(path,pixels,latent.float().cpu().numpy(),y.cpu().numpy(),
            dict(seconds=seconds,full_calls=128,prefix_calls=0,head_calls=counter.head,block_calls=np.array(counter.blocks)),
            rh,start,k.array_sha(noise[start:start+len(y)]))
        counter.close()
        if start%80==0:
            k.atomic(root/'progress.json',dict(pid=os.getpid(),completed=start+len(y),total=n,arm=arm,stage=stage))
    capture.close()
    collect(arm,stage,n,bank)
    print(stage,arm,'sampling complete',flush=True)


def collect(arm,stage,n,bank):
    root=k.ROOT/k.MODEL/stage/arm
    noise,labels=np.load(bank/'noise.npy',mmap_mode='r'),np.load(bank/'labels.npy')
    paths=sorted((root/'batches').glob('*.npz'))
    images,coverage,records=[],[],[]
    seconds=0.
    rh=k.sha(k.ROOT/'request.json')
    for path in paths:
        meta=k.read(path.with_suffix('.json'))
        assert meta['sha256']==k.sha(path) and meta['request_sha256']==rh
        with np.load(path) as d:
            start,count=int(d['start']),len(d['labels'])
            coverage.extend(range(start,start+count))
            np.testing.assert_array_equal(d['labels'],labels[start:start+count])
            assert str(d['noise_sha256'])==k.array_sha(noise[start:start+count]) and str(d['request_sha256'])==rh
            assert int(d['full_calls'])==128 and int(d['prefix_calls'])==0
            assert np.array_equal(d['block_calls'],np.full(12,128))
            assert int(d['head_calls'])==(0 if arm=='native' else 64)
            assert np.isfinite(d['latents']).all() and d['arr_0'].shape==(count,256,256,3)
            assert d['arr_0'].dtype==np.uint8
            images.append(d['arr_0'])
            seconds+=float(d['seconds'])
        records.append(dict(file=str(path),sha256=meta['sha256']))
    assert coverage==list(range(n)),(arm,len(coverage),n)
    output=root/'samples.npz'
    pixels=np.concatenate(images)
    del images
    if not output.exists():
        temp=output.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez(stream,arr_0=pixels)
        temp.replace(output)
    else:
        with np.load(output) as saved:
            np.testing.assert_array_equal(saved['arr_0'],pixels)
    k.atomic(root/'summary.json',dict(complete=True,model=k.MODEL,stage=stage,arm=arm,
        primary_samples=n,generated_paths=n,full_calls_per_output=128,prefix_calls_at_inference=0,
        head_calls_per_output=0 if arm=='native' else 64,seconds=seconds,
        samples_sha256=k.sha(output),request_sha256=rh,bank_sha256=k.sha(bank/'complete.json'),records=records))


def evaluate(arm,stage='screen1000'):
    k.verify()
    root=k.ROOT/k.MODEL/stage/arm
    summary=k.read(root/'summary.json')
    assert summary['complete'] and summary['samples_sha256']==k.sha(root/'samples.npz')
    output=root/'metrics.json'
    if output.exists():
        row=k.read(output)
        assert row['request_sha256']==k.sha(k.ROOT/'request.json') and row['samples_sha256']==summary['samples_sha256']
        return
    reference=k.EXPS.parent/'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
    command=['/data/shared/envs/adm-fid/bin/python',str(k.WORK/'experiments/compute_adm_fid.py'),
        '--reference',str(reference),'--samples',str(root/'samples.npz'),'--output',str(root/'adm.json'),
        '--activations-output',str(root/'activations.npz'),'--batch-size','32']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
             TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1')
    with (root/'evaluation.log').open('a') as stream:
        subprocess.run(command,cwd=k.WORK,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    metric=k.read(root/'adm.json')
    assert np.isfinite(metric['fid']) and np.isfinite(metric['inception_score'])
    k.atomic(output,dict(summary,fid=metric['fid'],inception_score=metric['inception_score'],metrics=metric))
    print(stage,arm,metric['fid'],metric['inception_score'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['gpu_preflight','train','features','weights','sample','evaluate'])
    parser.add_argument('--arm',choices=k.SCREEN_ARMS)
    parser.add_argument('--stage',default='screen1000',choices=['screen1000','confirm5000'])
    args=parser.parse_args()
    if args.action=='gpu_preflight':gpu_preflight()
    elif args.action=='train':train(args.arm)
    elif args.action=='features':endpoint_features()
    elif args.action=='weights':fit_weights()
    elif args.action=='sample':sample(args.arm,args.stage)
    else:evaluate(args.arm,args.stage)


if __name__=='__main__':
    try:
        main()
    except k.RequestedStop as error:
        print(error,flush=True)
        raise SystemExit(75)
