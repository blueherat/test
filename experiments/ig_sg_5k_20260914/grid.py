"""Full5K SG coefficient grid: step0.2, then step0.05 in top3 intervals."""
import argparse,os,sys
from pathlib import Path
import numpy as np
from experiments.ig_sg_5k_20260914 import run as r
from experiments.raev2_shallow_ig_20260914 import grid_common as g
c,ROOT,MODELS,m,old=r.c,r.ROOT,r.MODELS,r.m,r.old
MODULE='experiments.ig_sg_5k_20260914.grid'
STAGES=('grid_tune','grid_refine','confirm_5k')
PROTOCOL=c.WORK/'docs/GUIDANCE_5K_GRID_20260914_ZH.md'
verify=r.verify


def cfg(name,kind,w):return r.sg_config(name,kind,round(w-1,8))


def rows(phase,name):
    root=ROOT/phase/name;req=c.read(root/'request.json');assert req['samples']==5000
    assert c.read(ROOT/phase/'controller_complete.json')['complete']
    values=[]
    for x in req['configs']:
        if (root/x['arm']/'invalid.json').exists():continue
        values.append(dict(x,w=1+x.get('omega',0),fid=c.read(root/x['arm']/'metrics.json')[0]['fid'],source_phase=phase))
    return values


def choices(phase,name):
    base=m.configs(name)
    if phase=='grid_tune':return [base[0]]+[cfg(name,kind,w) for kind in ('sg','log') for w in g.COARSE if w>1],{}
    available=rows('grid_tune',name)
    if phase=='grid_refine':
        result=[];detail={}
        for kind in ('sg','log'):
            selected,values=g.best_intervals([x for x in available if x['kind'] in ('baseline',kind)])
            detail[kind]=dict(intervals=selected,fine_coefficients=values)
            result.extend(cfg(name,kind,w) for w in values)
        path=ROOT/'selection'/f'{name}_grid_refine.json';c.atomic(path,detail)
    else:
        available+=rows('grid_refine',name)
        detail={kind:min([x for x in available if x['kind']==kind],key=lambda x:(x['fid'],x['omega'])) for kind in ('sg','log')}
        path=ROOT/'selection'/f'{name}_grid_final.json';c.atomic(path,detail)
        result=[base[0],cfg(name,'sg',detail['sg']['w']),cfg(name,'log',detail['log']['w']),base[3],base[4]]
    return result,{str(path):c.sha(path)}


def prepare(phase):
    for name in MODELS:
        root=ROOT/phase/name;root.mkdir(parents=True,exist_ok=True)
        if (root/'request.json').exists():verify(phase,name);continue
        configs,selection=choices(phase,name)
        bank,record=r.input_bank('confirm_5k' if phase=='confirm_5k' else 'tune',name)
        prior=old.verify(m.ROOT/'screen_1k'/name);sources=dict(prior['sources'])
        for p in (Path(__file__).resolve(),Path(r.__file__).resolve(),Path(g.__file__).resolve(),PROTOCOL):sources[str(p)]=c.sha(p)
        if phase!='grid_tune':
            for earlier in (('grid_tune',) if phase=='grid_refine' else STAGES[:2]):
                parent=ROOT/earlier/name
                for p in [parent/'request.json',*sorted(parent.glob('*/metrics.json')),*sorted(parent.glob('*/invalid.json'))]:selection[str(p)]=c.sha(p)
        req=dict(model=name,phase=phase,samples=5000,seed=record['seed'],batch=16 if name=='sit_small' else 32,
            configs=configs,sources=sources,assets=prior['assets'],inputs=record['inputs'],input_origin=record['origin'],
            noise=str(bank/'noise.npy'),labels=str(bank/'labels.npy'),reused_arms={},selection=selection,
            reference=prior['reference'],ig_schedule=prior['ig_schedule'],precision=prior['precision'],
            definition='v_out=v_IG+(w_SG-1)*(v_strong-v_reference); fixed originalIG; only SG extrapolation changes',
            selection_rule='All5K: w1..2 step0.2; top3 adjacent intervals by mean5K endpoints, fine step0.05; positiveSG selected, IG-only separate',
            storage='Every pixel and feature retained; tuning latent SHA256/finite check plus first batch; all final latents retained')
        c.atomic(root/'request.json',req);print('Prepared',phase,name,len(configs),'arms x5000',flush=True)


def collect(phase,name,cfg,req):
    out=ROOT/phase/name/cfg['arm']
    if (out/'summary.json').exists():return
    images=[];records=[];full=[];seconds=0.;noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);coverage=[]
    rh=c.sha(ROOT/phase/name/'request.json')
    for start in range(0,5000,req['batch']):
        path=out/f'batch{start:06d}.npz'
        with np.load(path) as d:
            stop=start+len(d['pixels']);coverage.extend(range(start,stop))
            assert int(d['start'])==start and str(d['request_sha256'])==rh
            assert np.array_equal(d['labels'],labels[start:stop]) and str(d['noise_sha256'])==c.array_sha(noise[start:stop])
            g.check_latents(d,(stop-start,*m.SHAPES[name]));assert int(d['prefix'])==0
            images.append(d['pixels']);full.append(int(d['full']));seconds+=float(d['seconds'])
        records.append(dict(path=str(path),sha256=c.sha(path)))
    assert coverage==list(range(5000)) and len(set(full))==1
    g.raw_save(out/'samples.npz',arr_0=np.concatenate(images))
    c.atomic(out/'summary.json',dict(complete=True,samples=5000,reused=False,records=records,full_calls_per_output=full[0],prefix_calls_per_output=0,
        seconds=seconds,request_sha256=rh,samples_sha256=c.sha(out/'samples.npz')))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker','evaluate_one']);p.add_argument('--phase',default='grid_tune')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--model',choices=MODELS);p.add_argument('--arm');a=p.parse_args()
    if a.action=='prepare':prepare(a.phase)
    elif a.action=='worker':r.save_npz=g.writer(a.phase!='confirm_5k');r.worker(a.phase,a.rank)
    elif a.action=='controller':r.MODULE=MODULE;r.collect=collect;r.controller(a.phase)
    else:
        out=ROOT/a.phase/a.model/a.arm;old.evaluate(out,a.model);print(a.phase,a.model,a.arm,c.read(out/'metrics.json')[0]['fid'],flush=True)

if __name__=='__main__':main()
