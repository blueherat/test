"""Check production batch memory and sampling throughput before freezing."""
import argparse
import time
import torch
from experiments.self_guidance_cross_model_20260913 import core as c

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,default=32);a=p.parse_args()
    rt=c.Runtime(a.model)
    x=torch.randn(a.batch,*c.SHAPES[a.model],device='cuda');y=torch.arange(a.batch,device='cuda')
    base='cfg' if a.model=='jit' else 'ig'
    config=dict(arm='bench',base=base,kind='sg',omega=1.,steps=100)
    torch.cuda.synchronize();start=time.perf_counter()
    z,counts=c.sample(rt,x,y,config);pixels=rt.decode(z)
    torch.cuda.synchronize()
    assert pixels.shape==(a.batch,256,256,3)
    result=dict(model=a.model,batch=a.batch,seconds=time.perf_counter()-start,
                peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,full=counts['full'])
    c.common.atomic(c.ROOT/'checks'/f'{a.model}_batch{a.batch}.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
