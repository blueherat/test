"""Alternating native/candidate sample-and-decode timings on the same free GPU."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from . import core as m


def wait_sampler(pid):
    while Path('/proc',str(pid),'cmdline').exists():
        command=Path('/proc',str(pid),'cmdline').read_bytes()
        if b'experiments.reflection_guidance_20260912.core' not in command:break
        time.sleep(5)


@torch.inference_mode()
def benchmark(model):
    m.configure();m.verify(model);rt=c.runtime(model)
    noise,_,labels=c.bank(model,m.STAGE);x=c.cuda(noise[:rt.batch]);y=c.cuda(labels[:rt.batch])
    records=[]
    for track in ('ig','cfg'):
        kinds=[track+'_native',track+'_abba'];times={k:[] for k in kinds}
        for k in kinds:
            z,_=m.sample(rt,x,y,k);rt.decode(z)
        for repeat in range(3):
            for k in (kinds if repeat%2==0 else kinds[::-1]):
                torch.cuda.synchronize();start=time.perf_counter()
                z,counts=m.sample(rt,x,y,k);rt.decode(z)
                torch.cuda.synchronize();times[k].append(time.perf_counter()-start)
                assert counts==dict(full=m.expected(model,k),prefix=0)
        median={k:float(np.median(v)) for k,v in times.items()}
        records.append(dict(model=model,track=track,batch=rt.batch,repeats=3,seconds=times,median=median,
            relative_median_change=median[kinds[1]]/median[kinds[0]]-1,full_calls=m.expected(model,kinds[0]),
            extra_prefix=0,extra_trainable_parameters=0))
    c.atomic(m.ROOT/model/'inference_benchmark.json',dict(complete=True,records=records,
        source_sha256=c.sha(Path(__file__)),method_sha256=c.sha(Path(m.__file__))))
    print(model,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid:wait_sampler(a.wait_pid)
    for model in ('sit_small','raev2'):benchmark(model)
