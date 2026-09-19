"""Predeclare equal-224-call Z and tuned CFG/APG/CTRL comparisons."""
import json
from pathlib import Path

ALPHAS=(.75,1.125,1.25,1.5,2.)


def configurations():
    rows=[]
    # Paired interleaving makes each first result assessable against its own Z control.
    for a in ALPHAS:
        for wb in (0.,1.):
            ws=(1+a+wb)/2
            assert ws not in (0.,1.) and ws>wb
            for kind in ('z_release_euler','z_anchored_euler'):
                rows.append(dict(arm=f'{kind}_a{a:g}_b{wb:g}',kind=kind,alpha=a,
                                 backward_w=wb,forward_w=ws,steps=56,cutoff=.75))
    for a in ALPHAS:
        rows.append(dict(arm=f'cfg_heun64_a{a:g}',kind='cfg',alpha=a,steps=64,cutoff=.75))
        rows.append(dict(arm=f'cfg_euler128_a{a:g}',kind='cfg_euler',alpha=a,steps=128,cutoff=.75))
    for a in (1.5,2.,2.5):
        rows.append(dict(arm=f'apg_heun64_a{a:g}',kind='apg',alpha=a,steps=64,cutoff=.75,beta=-.5))
    for a in (2.25,2.75,3.25):
        rows.append(dict(arm=f'ctrl_heun64_a{a:g}',kind='ctrl',alpha=a,steps=64,cutoff=.75,lambda_ctrl=5.,K=.2))
    assert len(rows)==36 and len({r['arm'] for r in rows})==36
    return rows


if __name__=='__main__':
    path=Path(__file__).with_name('configs.json')
    if path.exists():
        assert json.loads(path.read_text())==configurations()
    else:
        path.write_text(json.dumps(configurations(),indent=2)+'\n')
    print(path)
