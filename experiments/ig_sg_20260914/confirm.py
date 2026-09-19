"""Independent JiT confirmation, selected SG setting fixed after seed1407 screen."""
import numpy as np
from pathlib import Path
from experiments.ig_sg_20260914 import run as r
c=r.c.common


def prepare():
    original=r.c.ROOT/'screen_1k/jit';req=r.verify(original)
    base=c.read(original/'ig/metrics.json')[0]['fid']
    sg=c.read(original/'ig_sg_w1/metrics.json')[0]['fid']
    cost=c.read(original/'ig_sg_cost/metrics.json')[0]['fid']
    assert sg<min(base,cost),('confirmation trigger unmet',base,sg,cost)
    root=r.c.ROOT/'confirm_1k/jit';root.mkdir(parents=True,exist_ok=True)
    seed=2026091417;rng=np.random.default_rng(seed)
    path=root/'noise.npy'
    if not path.exists():
        bank=np.lib.format.open_memmap(path.with_suffix('.tmp'),mode='w+',dtype=np.float32,shape=(1000,*r.c.SHAPES['jit']))
        for start in range(0,1000,8):bank[start:start+8]=rng.standard_normal((8,*r.c.SHAPES['jit']),dtype=np.float32)
        bank.flush();del bank;path.with_suffix('.tmp').replace(path)
        labels=np.arange(1000,dtype=np.int64);rng.shuffle(labels);np.save(root/'labels.npy',labels)
    selected={'ig','ig_sg_w1','ig_sg_cost'}
    req.update(phase='confirm_1k',seed=seed,configs=[x for x in req['configs'] if x['arm'] in selected],
        inputs={str(root/f'{k}.npy'):c.sha(root/f'{k}.npy') for k in ('noise','labels')},input_origin={},reused_arms={},
        selection=dict(screen_request_sha256=c.sha(original/'request.json'),screen_fid=dict(ig=base,sg=sg,cost=cost),
            rule='Original SG beats baseline and equal-NFE IG on the screen; fixed omega1/shift.01, no confirmation tuning'),
        scope='Independent paired JiT 1K confirmation; new PCG64 seed2026091417; IG, IG+paper SG, equal-NFE IG; parameters frozen from screen.')
    req['sources'][str(Path(__file__).resolve())]=c.sha(__file__)
    dest=root/'request.json'
    if dest.exists():assert c.read(dest)==req
    else:c.atomic(dest,req)
    print('prepared independent JiT 3K',seed,flush=True)

if __name__=='__main__':prepare()
