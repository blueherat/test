"""Fixed 16-image transfer check on the existing official SiT-XL/2 full branch."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core,study,pipeline as small
from experiments import lifting_scale_sweep_20260909 as infrastructure
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.run_official_sit_native_sde_20260909 import state_sha
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha,array_sha,WORK,EXPS

ROOT=EXPS/'sit_xl_fsg_ctrl_examples_20260911'
PROTOCOL=WORK/'docs/SIT_XL_FSG_CTRL_EXAMPLES_PROTOCOL_20260911_ZH.md'
METHODS=dict(conditional=dict(kind='cfg',amount=0.),cfg_low=dict(kind='cfg',amount=.5),
    **{name:dict(core.METHODS[name]) for name in ('cfg_tuned','cfg_high','fsg_high','fsg_length_high',
        'fsg_debiased_high','smc_high','instant_high','instant01_high','norm_high')},
    smc01_high=dict(kind='smc',amount=2.75,gain=.1,decay=5.))


class Runtime:
    """Expose increasing denoising time and proxy class labels to the tested core."""
    def __init__(self):
        torch.set_num_threads(4);torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.model,self.metadata=load_model(repo=infrastructure.XL_REPO,
            checkpoint_path=infrastructure.XL_CKPT,model_name='SiT-XL/2',encoder_depth=8,
            state_key='ema',device=torch.device('cuda'))
        from diffusers import AutoencoderKL
        self.vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).cuda().eval().requires_grad_(False)
        rows=sorted(read(small.semantic.MANIFEST)['classes'],key=lambda row:row['label'])
        self.class_map=torch.tensor([r['original_imagenet_label'] for r in rows]+[1000],device='cuda')
        self.labels=None;self.counts=dict(full=0,prefix=0)

    def pair(self,z,t):
        self.counts['full']+=1
        native_time=torch.full((len(z),),1-float(t),device='cuda',dtype=torch.float32)
        value,auxiliary,_=self.model(z.float(),native_time,self.class_map[self.labels])
        # The released checkpoint's velocity follows noise time 1 -> 0.
        return -value.double(),-auxiliary.double()

    def field(self,z,t,kind):
        assert kind=='full','No auxiliary prediction is used as a guidance branch.'
        return self.pair(z,t)[0]

    def decode(self,z):
        image=self.vae.decode(z.float()/.18215).sample
        return (127.5*image+128).clamp(0,255).permute(0,2,3,1).to(torch.uint8).cpu().numpy()


def sources():
    return [Path(__file__).resolve(),PROTOCOL,Path(core.__file__).resolve(),Path(study.__file__).resolve(),
        Path(small.__file__).resolve(),Path(core.legacy_fsg.__file__).resolve(),
        WORK/'experiments/run_internal_guidance_sit_audit.py',infrastructure.XL_REPO/'models/sit.py',
        infrastructure.XL_REPO/'samplers.py']


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return verify()
    previous=read(EXPS/'official_sit_pfr_upper_20260909/request.json')
    assert sha(infrastructure.XL_CKPT)==previous['checkpoint_sha256']
    small.verify()
    noise=np.load(small.ROOT/'inputs/noise.npy')[:16].copy()
    labels=np.load(small.ROOT/'inputs/labels.npy')[:16].copy()
    np.savez(ROOT/'inputs.npz',noise=noise,labels=labels)
    request=dict(samples=16,batch=4,ranks=4,methods=METHODS,time_steps=[16,32,48],
        checkpoint=str(infrastructure.XL_CKPT),checkpoint_sha256=previous['checkpoint_sha256'],
        vae_state_sha256=previous['vae_state_sha256'],noise_sha256=array_sha(noise),label_sha256=array_sha(labels),
        inputs_sha256=sha(ROOT/'inputs.npz'),sources={str(path):sha(path) for path in sources()},
        small_request_sha256=sha(small.ROOT/'study_request.json'),
        assets={str(path):sha(path) for path in small.semantic.source_assets()},
        scope='16 predetermined matching-noise examples, no FID or XL parameter selection',
        precision='FP32 network and FP64 state, matrix and convolution TF32 disabled',
        branch='Official IG checkpoint full branch with true ImageNet-1K NULL class 1000',
        created_unix=time.time())
    atomic(ROOT/'request.json',request);return verify()


def verify():
    request=read(ROOT/'request.json')
    for group in ('sources','assets'):
        for path,digest in request[group].items():assert sha(path)==digest,path
    assert sha(ROOT/'inputs.npz')==request['inputs_sha256']
    assert sha(infrastructure.XL_CKPT)==request['checkpoint_sha256']
    return request


@torch.inference_mode()
def preflight(rt,noise,labels,request):
    assert state_sha(rt.vae)==request['vae_state_sha256']
    rt.labels=labels
    # Native official Euler is an exact check of label mapping and the time/velocity reversal.
    from samplers import euler_sampler
    native=euler_sampler(rt.model,noise.float(),rt.class_map[labels],num_steps=64,heun=False,cfg_scale=1.)
    mapped=noise.double()
    for k in range(64):mapped=mapped+(1/64)*rt.field(mapped,mapped.new_tensor(k/64),'full')
    assert torch.equal(mapped,native),float((mapped-native).abs().max())
    times=torch.full((len(noise),),.625,device='cuda')
    direct=-rt.model(noise.float(),times,torch.full_like(labels,1000))[0].double()
    null=core.field(rt,noise.double(),.375,labels,True)
    assert torch.equal(null,direct)
    c,u=core.pair(rt,noise.double(),.375,labels)
    raw,radius,gap=core.fsg_delta(rt,noise.double(),.375,labels,2.75)
    forward=noise.double()+.125*(c+2.75*(c-u))
    hand=forward-.125*core.field(rt,forward,.5,labels,True)-noise.double()
    assert torch.equal(raw,hand)
    core.limiting_checks(rt,noise.double(),labels)
    return dict(native_euler_time_label_replay_exact=True,true_null1000_exact=True,
        fsg_velocity_time_sign_checked=True,source_metadata=rt.metadata)


@torch.inference_mode()
def worker(rank):
    request=verify();request_hash=sha(ROOT/'request.json')
    rt=Runtime()
    with np.load(ROOT/'inputs.npz') as data:
        start=4*rank
        n=torch.from_numpy(data['noise'][start:start+4].copy()).cuda().double()
        y=torch.from_numpy(data['labels'][start:start+4].copy()).cuda()
    check=preflight(rt,n,y,request);atomic(ROOT/f'preflight_rank{rank}.json',dict(passed=True,**check))
    reader=study.Readouts(rt)
    for name,method in METHODS.items():
        path=ROOT/f'{name}_{rank}.json'
        if path.exists():
            assert read(path)['request_sha256']==request_hash
            continue
        before=time.perf_counter()
        end,stats,states,traces=core.trajectory(rt,n,y,method,capture=(16,32,48),trace=True)
        image_dir=ROOT/'images'/name
        values=reader.latents(end,y,image_dir/'guided',start)
        rows=study.encode_records(len(n),dict(start=start,method=name,k=64,suffix='guided'),values)
        arrays=dict(endpoint=end.cpu().numpy())
        for step,snapshot in states.items():
            arrays[f'x{step:02d}']=snapshot['state'].cpu().numpy()
            tails=('null','conditional') if step==32 else ('null',)
            for tail in tails:
                result=core.future(rt,snapshot['state'],step,y,tail)
                arrays[f'{tail}{step:02d}']=result.cpu().numpy()
                values=reader.latents(result,y,image_dir/f'{tail}_{step:02d}',start)
                rows+=study.encode_records(len(n),dict(start=start,method=name,k=step,suffix=tail),values)
        np.savez(path.with_suffix('.npz'),**arrays)
        trace_rows=[]
        for trace in traces:
            trace_rows+=study.encode_records(len(n),dict(start=start,method=name,k=trace['k']),
                {key:value for key,value in trace.items() if key!='k'})
        atomic(path,dict(request_sha256=request_hash,rows=rows,traces=trace_rows,
            full_calls=stats['full_calls'],arrays_sha256=sha(path.with_suffix('.npz')),
            seconds=time.perf_counter()-before))
        print(dict(rank=rank,method=name,seconds=round(time.perf_counter()-before,2)),flush=True)
    atomic(ROOT/f'rank{rank}.json',dict(complete=True,request_sha256=request_hash))


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2]!='Z'


def run(temporarily_pause_small=False):
    prepare();paused=None
    streams=[];processes=[]
    try:
        if temporarily_pause_small:
            status=read(small.ROOT/'status.json')
            assert status['phase'] in ('trajectories','jacobian','interventions'),status
            pid=status['controller_pid']
            assert 'sit_fsg_ctrl_hypothesis_20260911.pipeline' in Path(f'/proc/{pid}/cmdline').read_text()
            assert '--run' in Path(f'/proc/{pid}/cmdline').read_text()
            os.kill(pid,signal.SIGSTOP);paused=pid
            atomic(ROOT/'small_pause.json',dict(controller_pid=pid,phase=status['phase'],
                worker_pids=status['worker_pids'],started_unix=time.time(),purpose='User requested larger-model examples'))
            while any(alive(worker_pid) for worker_pid in status['worker_pids']):time.sleep(2)
            assert all(read(small.ROOT/status['phase']/f'rank{r}.json')['complete'] for r in range(4))
        for rank in range(4):
            if (ROOT/f'rank{rank}.json').exists() and read(ROOT/f'rank{rank}.json')['complete']:continue
            stream=(ROOT/f'worker{rank}.log').open('a');streams.append(stream)
            processes.append(subprocess.Popen([small.engine.PYTHON,'-u','-m',
                'experiments.run_sit_xl_fsg_ctrl_examples_20260911','--rank',str(rank)],cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
        atomic(ROOT/'status.json',dict(phase='running',worker_pids=[p.pid for p in processes],controller_pid=os.getpid()))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('XL example worker failed')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        assert all(read(ROOT/f'rank{r}.json')['complete'] for r in range(4))
        verify()
        atomic(ROOT/'status.json',dict(phase='complete',samples=16,methods=len(METHODS),finished_unix=time.time()))
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:process.wait(timeout=30)
        for stream in streams:stream.close()
        if paused is not None:
            os.kill(paused,signal.SIGCONT)
            atomic(ROOT/'small_resumed.json',dict(controller_pid=paused,resumed_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int)
    parser.add_argument('--pause-small',action='store_true');args=parser.parse_args()
    if args.rank is not None:worker(args.rank)
    else:run(args.pause_small)
