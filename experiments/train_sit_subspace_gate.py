"""Matched scalar/PCA state-dependent gates on a frozen real SiT cache."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.train_subspace_gate_pilot import core


class SiTGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=core.TimeEmbedding(32)
        self.label=nn.Embedding(100,8)
        self.net=nn.Sequential(nn.Linear(4136,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.head=nn.Linear(128,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,z,t,y,kind):
        h=self.net(torch.cat([z.flatten(1),self.time(t),self.label(y)],dim=1))
        logits=self.head(h)
        if kind=='scalar':logits=logits.mean(1,keepdim=True).expand(-1,2)
        elif kind!='pca':raise ValueError(kind)
        safe=t.clamp(.001,.999)
        return (logits+(torch.log1p(-safe)-torch.log(safe))[:,None]).sigmoid()


def quadratic_loss(g,q,include_constant=False):
    value=q[:,:,0]*g.square()+2*q[:,:,1]*g
    if include_constant:value=value+q[:,:,2]
    return value.sum(1)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    meta=json.loads((a.cache/'manifest.json').read_text());assert meta['complete']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(202609311)
    def load(name):return torch.from_numpy(np.load(a.cache/(name+'.npy'))).to(a.device)
    z,t,y,q,native=load('states'),load('times'),load('labels').long(),load('quadratics'),load('native_risk')
    if not all(torch.isfinite(x).all() for x in [z,t,q,native]):raise ValueError('nonfinite cache')
    base=SiTGate().to(a.device);models={k:copy.deepcopy(base) for k in ['scalar','pca']}
    opts={k:torch.optim.Adam(m.parameters(),lr=2e-4) for k,m in models.items()}
    gen=torch.Generator(device=a.device).manual_seed(202609312);fit=meta['fit_count']
    history=[]
    for step in range(1,5001):
        ii=torch.randint(fit,(256,),generator=gen,device=a.device);row={'step':step}
        for kind,m in models.items():
            opts[kind].zero_grad(set_to_none=True)
            per_sample=quadratic_loss(m(z[ii],t[ii],y[ii],kind),q[ii])
            if a.loss_weighting=='native':
                per_sample=per_sample*(t[ii]*(1-t[ii])).double().square()
            loss=per_sample.mean()
            if not torch.isfinite(loss):raise FloatingPointError(kind)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[kind].step()
            row[kind+'_gate_dependent_sum_loss']=float(loss.detach())
        if step%500==0:history.append(row);print(json.dumps(row),flush=True)
    torch.save({'models':{k:m.state_dict() for k,m in models.items()},'cache_manifest':meta,
                'loss_weighting':a.loss_weighting,
                'basis':torch.load(a.cache/'basis.pt',map_location='cpu',weights_only=False)},a.output/'checkpoint.pt')
    rows=[]
    with torch.no_grad():
        for split,lo,hi in [('fit',0,fit),('holdout',fit,len(z))]:
            row={'split':split,'count':hi-lo,'native_velocity_mse':float(native[lo:hi].mean()/4096)}
            for kind,m in models.items():
                risk=[];gates=[]
                for start in range(lo,hi,256):
                    s=slice(start,min(start+256,hi));g=m(z[s],t[s],y[s],kind)
                    risk.append(quadratic_loss(g,q[s],True));gates.append(g)
                row[kind+'_velocity_mse']=float(torch.cat(risk).mean()/4096)
                row[kind+'_mean_gates']=torch.cat(gates).mean(0).tolist()
            rows.append(row);print(json.dumps(row),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'complete':True,'rows':rows,'updates':5000,
        'parameters_per_gate':sum(p.numel() for p in base.parameters()),'cache_manifest':meta,
        'loss_weighting':a.loss_weighting},indent=2)+'\n')
    (a.output/'training.json').write_text(json.dumps(history,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2')
    p.add_argument('--loss-weighting',choices=['raw','native'],default='raw')
    run(p.parse_args())
