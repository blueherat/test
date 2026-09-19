from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from . import catalog as c
from experiments.sit_sampler_reference_20260912 import train as parent_train
from experiments.sit_guided_objective_20260912 import core as original
from experiments.lifting_scale_sweep_20260909 import Runtime, atomic, read, sha


def verify():
    request=read(c.ROOT/'data_request.json')
    for key in ('sources','assets','input_files','old_stop_markers'):
        for path,digest in request[key].items():assert sha(path)==digest,(key,path)
    return request


def prepare():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    if (c.ROOT/'data_request.json').exists():return verify()
    parent=parent_train.verify()
    assert read(c.PARENT_ROOT/'status.json')['final_audit_passed']
    sources=dict(parent['sources'])
    sources.update({str(p):sha(p) for p in (Path(__file__).resolve(),Path(c.__file__).resolve(),c.PROTOCOL)})
    inputs=dict(parent['input_files'])
    for method in c.REUSED_METHODS:
        out=c.head_folder(method);done=read(out/'complete.json')
        assert done['passed'] and done['checkpoint_sha256']==sha(out/'model.pt')
        assert done['request_sha256']==sha(c.PARENT_ROOT/'training_request.json')
        for name in ('complete.json','model.pt','input_fingerprints.json'):inputs[str(out/name)]=sha(out/name)
    config=next(x for x in c.planned() if x['arm']=='strong_00')
    assert config['strength']==0 and config['solver']=='heun64'
    request=dict(sources=sources,assets=parent['assets'],input_files=inputs,
        old_stop_markers=parent['old_stop_markers'],seed=c.DATA_SEED,samples=c.DATA_SAMPLES,
        sampling_config=config,strong_only=True,training_bank_is_separate=True,
        selection_by_quality=False,created_unix=time.time())
    atomic(c.ROOT/'data_request.json',request)
    folder=c.ROOT/'data';folder.mkdir(exist_ok=True)
    for name in ('noise.npy','labels.npy','real_clean.npy','real_pool_positions.npy'):
        shutil.copyfile(c.PARENT_ROOT/'data'/name,folder/name)
        assert sha(folder/name)==sha(c.PARENT_ROOT/'data'/name)
    assert not np.array_equal(np.load(folder/'noise.npy')[:1000],np.load(c.OLD/'inputs/noise.npy'))
    atomic(folder/'inputs.json',dict(noise_sha256=sha(folder/'noise.npy'),labels_sha256=sha(folder/'labels.npy'),
        same_training_noise_as_guided_source_experiment=True))
    snapshot=c.ROOT/'data_sources';snapshot.mkdir(exist_ok=True)
    for i,path in enumerate(sorted(sources)):(snapshot/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    return request


@torch.inference_mode()
def generate(rank):
    assert rank in range(4)
    request=verify();folder=c.ROOT/'data';inputs=read(folder/'inputs.json')
    assert sha(folder/'noise.npy')==inputs['noise_sha256'] and sha(folder/'labels.npy')==inputs['labels_sha256']
    out=folder/f'rank{rank}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        receipt=read(out/'complete.json');assert receipt['data_request_sha256']==sha(c.ROOT/'data_request.json')
        assert all(sha(path)==digest for path,digest in receipt['raw_hashes'].items());return
    rt=Runtime('sit_small');torch.set_num_threads(4)
    noise=np.load(folder/'noise.npy');labels=np.load(folder/'labels.npy')
    # One original full forward per evaluation, and no guidance in the dataset.
    n=torch.from_numpy(noise[:8].copy()).cuda();y=torch.from_numpy(labels[:8].copy()).cuda()
    z,stats=original.sample(rt,n,y,request['sampling_config'])
    with np.load(c.OLD/'strong_00/rank0/batch0000.npz') as saved:
        quality_n=torch.from_numpy(np.load(c.OLD/'inputs/noise.npy')[:8].copy()).cuda()
        quality_y=torch.from_numpy(np.load(c.OLD/'inputs/labels.npy')[:8].copy()).cuda()
        golden,_=original.sample(rt,quality_n,quality_y,request['sampling_config'])
        np.testing.assert_array_equal(golden.cpu().numpy(),saved['latents'])
    assert stats['full_calls']==128 and stats['prefix_calls']==0
    raw_hashes={};elapsed=0.;coverage=[];full=prefix=0
    for start in range(rank*8,c.DATA_SAMPLES,32):
        if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        ids=np.arange(start,min(start+8,c.DATA_SAMPLES));path=out/f'batch{start:04d}.npz'
        n=torch.from_numpy(noise[ids].copy()).cuda();y=torch.from_numpy(labels[ids].copy()).cuda()
        torch.cuda.synchronize();begin=time.perf_counter()
        z,stats=original.sample(rt,n,y,request['sampling_config'])
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        assert torch.isfinite(z).all() and stats['full_calls']==128 and stats['prefix_calls']==0
        np.savez(path,indices=ids,labels=labels[ids],latents=z.cpu().numpy(),seconds=seconds,
            full_calls=stats['full_calls'],prefix_calls=stats['prefix_calls'])
        raw_hashes[str(path)]=sha(path);coverage.extend(ids.tolist());elapsed+=seconds
        full+=stats['full_calls']*len(ids);prefix+=stats['prefix_calls']*len(ids)
        if len(coverage)%200==0:atomic(out/'status.json',dict(source='strong',completed=len(coverage),gpu_seconds=elapsed))
    atomic(out/'complete.json',dict(source='strong',rank=rank,images=len(coverage),indices=coverage,
        raw_hashes=raw_hashes,gpu_seconds=elapsed,full_calls_per_image=full/len(coverage),
        prefix_calls_per_image=prefix/len(coverage),unguided_golden_exact=True,
        data_request_sha256=sha(c.ROOT/'data_request.json')))


def assemble():
    verify();folder=c.ROOT/'data';receipts=[read(folder/f'rank{r}/complete.json') for r in range(4)]
    labels=np.load(folder/'labels.npy');assert np.all(np.bincount(labels,minlength=100)==20)
    values=np.empty((c.DATA_SAMPLES,4,32,32),np.float32);covered=[]
    for receipt in receipts:
        assert receipt['data_request_sha256']==sha(c.ROOT/'data_request.json')
        for path,digest in receipt['raw_hashes'].items():
            assert sha(path)==digest
            with np.load(path) as batch:
                ids=batch['indices'];np.testing.assert_array_equal(batch['labels'],labels[ids])
                assert int(batch['full_calls'])==128 and int(batch['prefix_calls'])==0
                values[ids]=batch['latents'];covered.extend(ids.tolist())
    assert sorted(covered)==list(range(c.DATA_SAMPLES)) and np.isfinite(values).all()
    np.save(folder/'strong_clean.npy',values.reshape(20,100,4,32,32).transpose(1,0,2,3,4).copy())
    files=list(folder.glob('*.npy'))+[folder/'inputs.json']+[folder/f'rank{r}/complete.json' for r in range(4)]
    atomic(folder/'complete.json',dict(passed=True,raw_coverage_and_hashes_passed=True,samples_per_dataset=c.DATA_SAMPLES,
        per_class=20,training_bank_separate_from_fid=True,selection_by_quality=False,
        files={str(p):sha(p) for p in files},generation_gpu_seconds={'strong':sum(r['gpu_seconds'] for r in receipts)},
        data_request_sha256=sha(c.ROOT/'data_request.json')))


class Pool:
    def __init__(self,dataset,device='cuda'):
        self.values=torch.from_numpy(np.load(c.ROOT/f'data/{dataset}_clean.npy')).to(device)
        self.device=device

    def draw(self,method,generator,count=32,labels=None):
        assert method=='native'
        labels=torch.randint(100,(count,),device=self.device,generator=generator) if labels is None else labels
        indices=torch.randint(20,(count,),device=self.device,generator=generator)
        return self.values[labels,indices],labels


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--rank',type=int);g.add_argument('--assemble',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.assemble:assemble()
    else:generate(args.rank)
