"""Quality sampling and frozen clean-endpoint bank generation."""
import time
import numpy as np
import torch

from . import config as k
from . import components as x


def head_provenance(arm):
    if arm in ('native','cfg_native'):
        return {}
    root=k.OLD if arm=='context' else k.ROOT/'training'/arm.removesuffix('_half')
    return {str(root/'head.pt'):k.sha(root/'head.pt')}


def verify_batch(path, request_hash, provenance):
    meta=k.read(path.with_suffix('.json'))
    assert meta['sha256']==k.sha(path) and meta['request_sha256']==request_hash
    assert meta['head_provenance']==provenance
    return meta


class Counter(x.legacy.CallCounter):
    def __init__(self, rt, head):
        super().__init__(rt,head)
        self.native_head=0
        def mark(module,args):
            self.native_head+=1
        self.handles.append(rt.head.module.register_forward_pre_hook(mark))


@torch.inference_mode()
def sample(arm, stage='screen1000'):
    k.verify()
    n,seed=(1000,k.SCREEN_SEED) if stage=='screen1000' else (5000,k.CONFIRM_SEED)
    bank=x.legacy.prepare_bank(stage,n,seed)
    root=k.ROOT/k.MODEL/stage/arm
    provenance=head_provenance(arm)
    if (root/'summary.json').exists():
        row=k.read(root/'summary.json')
        assert row['head_provenance']==provenance
        collect(arm,stage,n,bank,provenance)
        return
    rt=x.legacy.runtime()
    head=None if arm in ('native','cfg_native') else x.legacy.trained_head(rt,arm)
    capture=x.local.Capture(rt)
    noise,labels=np.load(bank/'noise.npy',mmap_mode='r'),np.load(bank/'labels.npy')
    expected=k.inference_counts(arm)
    rh=k.sha(k.ROOT/'request.json')
    for start in range(0,n,rt.batch):
        k.check_stop()
        path=root/'batches'/f'{start:05d}.npz'
        if path.exists() and not path.with_suffix('.json').exists():
            path.unlink()  # Uncommitted batch from an interrupted write; regenerate it.
        if path.exists():
            verify_batch(path,rh,provenance)
            continue
        z,y=x.c.cuda(noise[start:start+rt.batch]),x.c.cuda(labels[start:start+rt.batch])
        counter=Counter(rt,head)
        torch.cuda.synchronize()
        begin=time.perf_counter()
        latent,counts=x.c.integrate(rt,z,y,lambda state,t,left,i,j:x.guided_field(rt,head,capture,arm,state,t,left))
        assert counts==dict(full=expected['full'],prefix=0)
        assert counter.blocks==[expected['full']]*12
        assert counter.head==expected['head'] and counter.native_head==expected['native_head']
        pixels=rt.decode(latent)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream,arr_0=pixels,latents=latent.float().cpu().numpy(),labels=y.cpu().numpy(),
                seconds=seconds,full_calls=expected['full'],prefix_calls=0,head_calls=counter.head,
                native_head_calls=counter.native_head,block_calls=np.array(counter.blocks),
                request_sha256=rh,start=start,noise_sha256=k.array_sha(noise[start:start+len(y)]))
        temporary.replace(path)
        k.atomic(path.with_suffix('.json'),dict(sha256=k.sha(path),request_sha256=rh,head_provenance=provenance))
        counter.close()
        if start%80==0:
            k.atomic(root/'progress.json',dict(pid=__import__('os').getpid(),arm=arm,stage=stage,completed=start+len(y),total=n))
    capture.close()
    collect(arm,stage,n,bank,provenance)


def collect(arm,stage,n,bank,provenance):
    root=k.ROOT/k.MODEL/stage/arm
    noise,labels=np.load(bank/'noise.npy',mmap_mode='r'),np.load(bank/'labels.npy')
    expected=k.inference_counts(arm)
    coverage,images,records=[],[],[]
    seconds=0.
    rh=k.sha(k.ROOT/'request.json')
    for path in sorted((root/'batches').glob('*.npz')):
        meta=verify_batch(path,rh,provenance)
        with np.load(path) as d:
            start,count=int(d['start']),len(d['labels'])
            coverage.extend(range(start,start+count))
            np.testing.assert_array_equal(d['labels'],labels[start:start+count])
            assert str(d['noise_sha256'])==k.array_sha(noise[start:start+count])
            assert str(d['request_sha256'])==rh
            assert int(d['full_calls'])==expected['full'] and int(d['prefix_calls'])==0
            assert int(d['head_calls'])==expected['head'] and int(d['native_head_calls'])==expected['native_head']
            np.testing.assert_array_equal(d['block_calls'],np.full(12,expected['full']))
            assert d['arr_0'].shape==(count,256,256,3) and d['arr_0'].dtype==np.uint8
            assert np.isfinite(d['latents']).all()
            images.append(d['arr_0'])
            seconds+=float(d['seconds'])
        records.append(dict(file=str(path),sha256=meta['sha256']))
    assert coverage==list(range(n))
    pixels=np.concatenate(images)
    del images
    output=root/'samples.npz'
    if output.exists():
        with np.load(output) as old:np.testing.assert_array_equal(old['arr_0'],pixels)
    else:
        temporary=output.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez(stream,arr_0=pixels)
        temporary.replace(output)
    k.atomic(root/'summary.json',dict(complete=True,model=k.MODEL,stage=stage,arm=arm,
        primary_samples=n,generated_paths=n,full_calls_per_output=expected['full'],prefix_calls_at_inference=0,
        head_calls_per_output=expected['head'],native_head_calls_per_output=expected['native_head'],
        head_provenance=provenance,seconds=seconds,samples_sha256=k.sha(output),request_sha256=rh,
        bank_sha256=k.sha(bank/'complete.json'),records=records,
        training_steps=k.STEPS if arm in k.ARMS or arm=='gaussian_half' else None))


@torch.inference_mode()
def endpoints(source):
    """Each source is frozen for one round; no image decode/re-encode or clipping."""
    k.verify()
    root=k.ROOT/'endpoints'/source
    if (root/'complete.json').exists():
        x.read_negative(source)
        return
    n,seed=2000,k.SEED+{'weak':301,'ig':302,'cfg':303}[source]
    bank=x.legacy.prepare_bank('endpoint_'+source,n,seed)
    noise,labels=np.load(bank/'noise.npy',mmap_mode='r'),np.load(bank/'labels.npy')
    rt=x.legacy.runtime()
    head=x.legacy.trained_head(rt,'self') if source=='weak' else None
    capture=x.local.Capture(rt)
    rh=k.sha(k.ROOT/'request.json')
    dependencies=head_provenance('self') if source=='weak' else {}
    for start in range(0,n,rt.batch):
        k.check_stop()
        path=root/'batches'/f'{start:05d}.npz'
        if path.exists() and not path.with_suffix('.json').exists():
            path.unlink()
        if path.exists():
            verify_batch(path,rh,dependencies)
            continue
        z,y=x.c.cuda(noise[start:start+rt.batch]),x.c.cuda(labels[start:start+rt.batch])
        counter=Counter(rt,head)
        def velocity(state,t,left,i,j):
            if source!='weak':
                arm='native' if source=='ig' else 'cfg_native'
                return x.guided_field(rt,None,capture,arm,state,t,left)
            rt.counts['prefix']+=1
            features=x.local.features(rt,state,t.expand(len(state)),y)
            return x.local.unpatchify(rt,head(features['context'],features['condition'])).float()
        latent,counts=x.c.integrate(rt,z,y,velocity)
        if source=='weak':
            assert counts==dict(full=0,prefix=128) and counter.blocks==[128]*4+[0]*8
            assert counter.head==128 and counter.native_head==0
        else:
            expected=k.inference_counts('native' if source=='ig' else 'cfg_native')
            assert counts==dict(full=expected['full'],prefix=0) and counter.blocks==[expected['full']]*12
            assert counter.native_head==expected['native_head'] and counter.head==0
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream,latents=latent.float().cpu().numpy(),labels=y.cpu().numpy(),start=start,
                request_sha256=rh,noise_sha256=k.array_sha(noise[start:start+len(y)]),
                full_calls=counts['full'],prefix_calls=counts['prefix'],head_calls=counter.head,
                native_head_calls=counter.native_head,block_calls=np.array(counter.blocks))
        temporary.replace(path)
        k.atomic(path.with_suffix('.json'),dict(sha256=k.sha(path),request_sha256=rh,head_provenance=dependencies))
        counter.close()
    capture.close()
    bank_values=np.empty((100,20,4,32,32),np.float32)
    filled=np.zeros(100,dtype=np.int64)
    coverage=[];records=[]
    for path in sorted((root/'batches').glob('*.npz')):
        meta=verify_batch(path,rh,dependencies)
        with np.load(path) as d:
            start=int(d['start']);size=len(d['labels'])
            coverage.extend(range(start,start+size))
            np.testing.assert_array_equal(d['labels'],labels[start:start+size])
            assert str(d['request_sha256'])==rh and str(d['noise_sha256'])==k.array_sha(noise[start:start+size])
            assert d['latents'].shape==(size,4,32,32) and np.isfinite(d['latents']).all()
            expected_full=0 if source=='weak' else 128 if source=='ig' else 224
            assert int(d['full_calls'])==expected_full and int(d['prefix_calls'])==(128 if source=='weak' else 0)
            assert int(d['head_calls'])==(128 if source=='weak' else 0)
            assert int(d['native_head_calls'])==(64 if source=='ig' else 0)
            np.testing.assert_array_equal(d['block_calls'],[128]*4+[0]*8 if source=='weak' else [expected_full]*12)
            for value,label in zip(d['latents'],d['labels']):
                bank_values[label,filled[label]]=value
                filled[label]+=1
        records.append(dict(file=str(path),sha256=meta['sha256']))
    assert coverage==list(range(n)) and np.all(filled==20)
    output=root/'clean.npy'
    temporary=output.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.save(stream,bank_values)
    temporary.replace(output)
    k.atomic(root/'complete.json',dict(complete=True,source=source,samples=n,seed=seed,steps=64,
        request_sha256=rh,files={str(output):k.sha(output)},dependencies=dependencies,records=records,
        endpoint_policy='continuous clean latent, no clipping; first 18/class train, last 2/class validation'))
