"""Rebuild the retained 3K state exactly, then train Context to total 20K."""
import argparse
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as old

ROOT = c.EXPS / 'raev2_context_20k_20260914'
TRAIN = ROOT / 'raev2/training'
OLD = old.ROOT / 'raev2/input_local_training'
PROTOCOL = c.WORK / 'docs/RAEV2_CONTEXT_20K_PROTOCOL_20260914_ZH.md'
STEPS = 20000
MILESTONES = (3000, 5000, 10000, 15000, 20000)
VALIDATION_SEEDS = {'validation':2026091401, 'train':2026091402}


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def prepare():
    old.verify_request(OLD / 'request.json')
    prior = c.read(OLD / 'request.json')
    request = dict(model='raev2', total_steps=STEPS, previous_steps=3000,
        restoration='exact replay of original 3K, followed by Context-only continuation',
        batch=8, lr=.0003, weight_decay=.0001, ema=.995, seed=old.TRAIN_SEED,
        validation_seeds=VALIDATION_SEEDS, validation_cases_per_split=1000,
        validation_steps=list(range(3000, STEPS+1, 1000)),
        sources={**prior['sources'], str(Path(__file__).resolve()):c.sha(Path(__file__)), str(PROTOCOL):c.sha(PROTOCOL)},
        assets=prior['assets'], data=prior['data'],
        heads={str(OLD/f):c.sha(OLD/f) for f in ('head.pt','summary.json','request.json')})
    path = TRAIN / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


@torch.no_grad()
def fixed_cache(rt, data):
    """One deterministic case per class; tensors are cached on CPU at native precision."""
    from experiments.train_raev2_observable_potential import batch_from_bank
    cache = {}
    identities = {}
    for split, seed in VALIDATION_SEEDS.items():
        bank = data.banks[split]
        labels = bank[1]['labels']
        indices = np.array([np.flatnonzero(labels == cls)[0] for cls in range(1000)], dtype=np.int64)
        assert len(np.unique(indices)) == 1000
        g = torch.Generator(device='cuda').manual_seed(seed)
        batches, times, noise_hashes = [], [], []
        for start in range(0, 1000, 8):
            clean, y = batch_from_bank(bank, indices[start:start+8], 'cuda')
            t = .01 + .98 * torch.rand(8, device='cuda', generator=g)
            eps = torch.randn(clean.shape, device='cuda', generator=g)
            z = (1-t[:,None,None,None])*clean + t[:,None,None,None]*eps
            with rt.context():
                features = old.features(rt, z, t, y)
            batches.append((features['context'].cpu(), features['condition'].cpu(), old.patchify(rt, clean).cpu()))
            times.extend(t.cpu().tolist())
            noise_hashes.append(c.array_sha(eps.cpu().numpy()))
        cache[split] = batches
        identities[split] = dict(seed=seed, bank_indices=indices.tolist(), labels=labels[indices].tolist(),
            times=times, noise_batch_sha256=noise_hashes, count=1000,
            context_dtype=str(batches[0][0].dtype), target_dtype=str(batches[0][2].dtype))
    path = TRAIN/'fixed_loss_cases.json'
    if path.exists():
        assert c.read(path) == identities
    else:
        c.atomic(path, identities)
    return cache


@torch.no_grad()
def evaluate_fixed(rt, ema, cache, step):
    rows = {}
    for split, batches in cache.items():
        values = []
        for tokens, conditions, targets in batches:
            with rt.context():
                pred = ema(tokens.cuda(), conditions.cuda())
            mse = (pred.float()-targets.cuda()).square().flatten(1).mean(1)
            values.extend(mse.cpu().tolist())
        assert len(values) == 1000 and np.isfinite(values).all()
        rows[split] = dict(mean=float(np.mean(values)), values=values, cases=1000)
    record = dict(step=step, model_weights='EMA', splits=rows)
    path = TRAIN/'fixed_validation'/f'step_{step:05d}.json'
    if path.exists():
        assert c.read(path) == record
    else:
        c.atomic(path, record)
    print('fixed EMA loss',step,{k:v['mean'] for k,v in rows.items()},flush=True)
    return {k:v['mean'] for k,v in rows.items()}


def main(parent=0):
    rp = prepare()
    old.verify_request(rp)
    if (TRAIN/'summary.json').exists():
        assert c.read(TRAIN/'summary.json')['complete']
        return
    torch.manual_seed(old.TRAIN_SEED)
    rt = c.runtime('raev2')
    data = old.RealData('raev2')
    heads = {'local':old.make_head(rt)}
    heads['context'] = copy.deepcopy(heads['local'])
    g = torch.Generator(device='cuda').manual_seed(old.TRAIN_SEED)
    c.atomic(TRAIN/'checks_before.json',old.checks(rt, heads))
    sums = {k:torch.zeros(old.shape(rt)[0],device='cuda',dtype=torch.float64) for k in ('local','context','condition')}
    squares = {k:v.clone() for k,v in sums.items()}
    counts = {k:0 for k in sums}
    with torch.no_grad():
        for _ in range(32):
            feats,_,_,_,_ = old.training_batch(rt,data,'train',g,8)
            for key,x in feats.items():
                x = x.reshape(-1,x.shape[-1]).double()
                sums[key] += x.sum(0)
                squares[key] += x.square().sum(0)
                counts[key] += len(x)
        for key,head in heads.items():
            for feat,prefix in ((key,'token'),('condition','condition')):
                mean = sums[feat]/counts[feat]
                std = (squares[feat]/counts[feat]-mean.square()).clamp_min(1e-8).sqrt()
                getattr(head,prefix+'_mean').copy_(mean.float())
                getattr(head,prefix+'_std').copy_(std.float())
    ema = {k:copy.deepcopy(h).eval().requires_grad_(False) for k,h in heads.items()}
    opts = {k:torch.optim.AdamW(h.parameters(),lr=.0003,weight_decay=.0001) for k,h in heads.items()}
    prior = torch.load(OLD/'head.pt',map_location='cpu',weights_only=True)
    old_history = {r['step']:r for r in c.read(OLD/'summary.json')['history']}
    history, validation = [], []
    first_step, previous_seconds, replay_exact = 1, 0., False
    latest = TRAIN/'latest.pt'
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        assert state['request_sha256'] == c.sha(rp)
        for k in heads:
            heads[k].load_state_dict(state['online'][k])
            ema[k].load_state_dict(state['ema'][k])
            opts[k].load_state_dict(state['optimizers'][k])
        g.set_state(state['data_rng'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        first_step = state['step']+1
        history, validation = state['history'],state['validation']
        previous_seconds,replay_exact = state['elapsed_seconds'],state['replay_exact']
    cache = None
    if first_step > 3000:
        assert replay_exact
        before = g.get_state().clone()
        with torch.random.fork_rng(devices=[0]):
            cache = fixed_cache(rt,data)
        assert torch.equal(g.get_state(),before)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for step in range(first_step,STEPS+1):
        if step % 100 == 1:
            c.check_parent(parent)
        feats,target,_,_,_ = old.training_batch(rt,data,'train',g,8)
        losses = {}
        active = ('local','context') if step <= 3000 else ('context',)
        for key in active:
            head = heads[key]
            opts[key].zero_grad(set_to_none=True)
            with rt.context():
                pred = head(feats[key],feats['condition'])
            loss = (pred.float()-target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            opts[key].step()
            with torch.no_grad():
                for p,q in zip(ema[key].parameters(),head.parameters()):
                    p.lerp_(q,.005)
            losses[key] = float(loss.detach())
        if step in old_history:
            for key in active:
                assert losses[key] == old_history[step][key], (step,key,losses[key],old_history[step][key])
        history.append(dict(step=step,**losses))
        if step == 3000:
            differences = {k:{name:float((value.cpu()-prior['ema'][k][name]).abs().max())
                for name,value in ema[k].state_dict().items()} for k in ema}
            assert all(d == 0 for vals in differences.values() for d in vals.values()),differences
            replay_exact = True
            c.atomic(TRAIN/'replay_3000.json',dict(passed=True,all_ema_tensors_exact=True,
                recorded_losses_exact=True,recorded_loss_points=len(old_history),differences=differences,
                optimizer_reconstructed=True,training_data_rng_preserved=True))
            before = g.get_state().clone()
            with torch.random.fork_rng(devices=[0]):
                vg = torch.Generator(device='cuda').manual_seed(old.TRAIN_SEED+7)
                original_validation = []
                with torch.no_grad():
                    for _ in range(32):
                        f,target_old,_,_,_ = old.training_batch(rt,data,'validation',vg,8)
                        with rt.context():
                            pred_old = ema['context'](f['context'],f['condition'])
                        original_validation.append(float((pred_old.float()-target_old).square().mean()))
                old_mean = c.read(OLD/'summary.json')['validation_mse']['context']
                assert float(np.mean(original_validation)) == old_mean
                c.atomic(TRAIN/'old_validation_replay.json',dict(passed=True,original_mse=old_mean,
                    replay_mse=float(np.mean(original_validation)),batches=32,batch=8))
                cache = fixed_cache(rt,data)
            assert torch.equal(g.get_state(),before)
        if step >= 3000 and step % 1000 == 0:
            assert replay_exact and cache is not None
            before = g.get_state().clone()
            with torch.random.fork_rng(devices=[0]):
                scores = evaluate_fixed(rt,ema['context'],cache,step)
            assert torch.equal(g.get_state(),before)
            validation.append(dict(step=step,**scores))
        if step == 1 or step % 100 == 0:
            mean100 = float(np.mean([r['context'] for r in history[-100:]]))
            record = dict(pid=os.getpid(),step=step,total=STEPS,context=losses['context'],
                mean_last_100=mean100,replay_exact=replay_exact,validation=validation[-1] if validation else None)
            c.atomic(TRAIN/'progress.json',record)
            if step == 1 or step % 1000 == 0:
                print('training',record,flush=True)
        if step % 1000 == 0:
            assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
            torch.cuda.synchronize()
            state = dict(step=step,online={k:h.state_dict() for k,h in heads.items()},
                ema={k:h.state_dict() for k,h in ema.items()},optimizers={k:o.state_dict() for k,o in opts.items()},
                data_rng=g.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                history=history,validation=validation,elapsed_seconds=previous_seconds+time.perf_counter()-begin,
                request_sha256=c.sha(rp),replay_exact=replay_exact)
            save_torch(latest,state)
            if step in MILESTONES:
                save_torch(TRAIN/'checkpoints'/f'step_{step:05d}.pt',state)
    assert replay_exact
    c.atomic(TRAIN/'checks_after.json',old.checks(rt,ema))
    head_path = TRAIN/'head.pt'
    save_torch(head_path,dict(ema={'context':ema['context'].state_dict()},step=STEPS,model='raev2',request_sha256=c.sha(rp)))
    c.atomic(TRAIN/'summary.json',dict(complete=True,model='raev2',steps=STEPS,previous_steps=3000,
        replay_exact=True,continuation_steps=17000,batch=8,parameters_per_head=sum(p.numel() for p in ema['context'].parameters()),
        elapsed_seconds=previous_seconds+time.perf_counter()-begin,includes_fixed_evaluation_and_checkpoint_io=True,
        history=history,validation=validation,head_sha256=c.sha(head_path),request_sha256=c.sha(rp),
        strong_frozen=True,optimizer_and_rng_saved=True,head_selection='fixed final 20K EMA'))
    print('20K training complete',validation[-1],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent',type=int,default=0)
    args = parser.parse_args()
    main(args.parent)
