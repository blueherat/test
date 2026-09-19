from pathlib import Path
import argparse
import time
import numpy as np
import torch
from . import catalog as c,models
from experiments.sit_reference_compilation_20260912 import data as previous,train as previous_train,core as previous_core
from experiments.lifting_scale_sweep_20260909 import Runtime,atomic,read,sha


def sources():
    return sorted(set([*previous.sources(),*Path(__file__).resolve().parent.glob('*.py'),c.PROTOCOL]))


def prepare():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    if (c.ROOT/'data_request.json').exists():return verify()
    parent=previous_train.verify();status=read(c.PARENT_ROOT/'status.json')
    assert status['phase']=='complete' and status['final_audit_passed']
    assert read(c.PARENT_ROOT/'training/ig_shallow/complete.json')['checkpoint_sha256']==sha(c.PARENT_ROOT/'training/ig_shallow/model.pt')
    request=dict(parent);request['sources']=dict(parent['sources']);request['assets']=dict(parent['assets'])
    request['sources'].update({str(p):sha(p) for p in sources()})
    for p in (c.PARENT_ROOT/'training/ig_shallow/model.pt',c.PARENT_ROOT/'training/ig_shallow/complete.json',
              c.PARENT_ROOT/'screen_review.json',c.PARENT_ROOT/previous.c.STAGE/'results.json'):
        request['assets'][str(p)]=sha(p)
    request.update(data_seed=c.DATA_SEED,methods=c.METHODS,states='paired fresh teacher/student rollouts',created_unix=time.time())
    atomic(c.ROOT/'data_request.json',request);snapshots=c.ROOT/'sources';snapshots.mkdir(exist_ok=True)
    for i,path in enumerate(sorted(request['sources'])):(snapshots/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    return request


def verify():
    request=read(c.ROOT/'data_request.json')
    for key in ('sources','assets','input_files','old_stop_markers'):
        for path,digest in request[key].items():assert sha(path)==digest,(key,path)
    return request


@torch.no_grad()
def generate(rank):
    verify();out=c.ROOT/'data';out.mkdir(exist_ok=True)
    rt=Runtime('sit_small');teacher=previous.prefix_core.get_weak(rt,'native')
    student=previous_core.get_head(rt,'ig_shallow');strong_hash=previous_train.state_sha(rt.model)
    generator=torch.Generator(device='cuda').manual_seed(c.DATA_SEED)
    noises=torch.randn((c.TRAJECTORIES,4,32,32),device='cuda',generator=generator)
    for source in ('teacher','student'):
        path=out/f'{source}_rank{rank}.pt'
        if path.exists():assert read(path.with_suffix('.json'))['sha256']==sha(path);continue
        records={k:[] for k in ('index','step','f4','context','teacher','strong')}
        torch.cuda.synchronize();begin=time.perf_counter()
        with models.Capture(rt,depths=(4,)) as capture:
            for start in range(rank*8,c.TRAJECTORIES,32):
                ids=torch.arange(start,min(start+8,c.TRAJECTORIES),device='cuda');labels=ids%100
                rt.labels=labels;z=noises[ids].clone();offsets=(ids//100+3*labels)%8
                for k in range(32):
                    t,h=k/64,1/64;ts=z.new_full((len(z),),t);a=.8*(6/7 if t<.25 else 1.)
                    strong=rt.field(z,z.new_tensor(t),'full')
                    weak=(teacher(z,ts,labels) if source=='teacher' else models.unpatchify(rt,student(capture.values[4],capture.context)))
                    select=offsets==k%8
                    if select.any():
                        target=weak if source=='teacher' else teacher(z,ts,labels)
                        row=dict(index=ids,step=torch.full_like(ids,k),f4=capture.values[4],context=capture.context,
                            teacher=target,strong=strong)
                        for key,value in row.items():records[key].append(value[select].cpu())
                    first=strong+a*(strong-weak);pred=z+h*first
                    strong2=rt.field(pred,pred.new_tensor(t+h),'full')
                    weak2=(teacher(pred,ts+h,labels) if source=='teacher' else models.unpatchify(rt,student(capture.values[4],capture.context)))
                    z=z+(h/2)*(first+strong2+a*(strong2-weak2));assert torch.isfinite(z).all()
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        assert previous_train.state_sha(rt.model)==strong_hash
        torch.save({k:torch.cat(v) for k,v in records.items()},path)
        atomic(path.with_suffix('.json'),dict(passed=True,sha256=sha(path),seconds=elapsed,source=source,
            states=sum(len(v) for v in records['index']),strong_unchanged=True,
            data_request_sha256=sha(c.ROOT/'data_request.json')))


def assemble():
    verify();files={};paired={}
    for source in ('teacher','student'):
        indices=[];steps=[]
        for rank in range(4):
            p=c.ROOT/'data'/f'{source}_rank{rank}.pt';done=read(p.with_suffix('.json'))
            assert done['passed'] and done['sha256']==sha(p) and done['strong_unchanged']
            assert done['data_request_sha256']==sha(c.ROOT/'data_request.json')
            value=torch.load(p,map_location='cpu',weights_only=False)
            indices.append(value['index']);steps.append(value['step'])
            for path in (p,p.with_suffix('.json')):files[str(path)]=sha(path)
            del value
        ix=torch.cat(indices).numpy();st=torch.cat(steps).numpy()
        assert np.all(np.bincount(ix,minlength=c.TRAJECTORIES)==4)
        counts=np.zeros((100,32),int);mask=ix<c.TRAIN_TRAJECTORIES;np.add.at(counts,(ix[mask]%100,st[mask]),1)
        assert np.all(counts==1);paired[source]=(ix,st)
    assert all(np.array_equal(x,y) for x,y in zip(paired['teacher'],paired['student']))
    atomic(c.ROOT/'data/complete.json',dict(passed=True,files=files,paired_index_time_exact=True,
        data_request_sha256=sha(c.ROOT/'data_request.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);generate(p.parse_args().rank)
