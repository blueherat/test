"""CPU-only live JiT coefficient/EMA curves, GAN health, and checkpoint index."""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time

import numpy as np


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def history(root,name):
    rows={}
    for path in sorted(root.glob(f'*/{name}')):
        if not (path.parent/'initial_state.json').exists():continue
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            rows[row['step']]=dict(row,source_run=path.parent.name)
    return [rows[key] for key in sorted(rows)]


def save_figure(fig,path):
    for suffix in ('png','pdf'):
        target=path.with_suffix('.'+suffix);temp=path.with_suffix('.tmp.'+suffix)
        fig.savefig(temp,dpi=150);temp.replace(target)


def publish_pngs(root,destination):
    """Publish real PNG files in the repository; replace each file atomically."""
    destination.mkdir(parents=True,exist_ok=True)
    for source_name,target_name in (('schedule.png','jit_gan_schedule.png'),
                                    ('gan_health.png','jit_gan_health.png')):
        source=root/source_name
        if source.exists():
            target=destination/target_name
            temporary=target.with_suffix('.tmp.png')
            temporary.write_bytes(source.read_bytes());temporary.replace(target)


def refresh(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=history(root,'train.jsonl')
    training=[row for row in rows if row['phase']=='training']
    if not training:return None
    probes=[row for row in history(root,'diagnostics.jsonl') if 'head_gradient' in row]
    last=training[-1];run=root/last['source_run']
    config=read(run/'initial_state.json')['config']
    raw=np.asarray(last['coefficients']);ema=last.get('ema_coefficients')
    latest=read(run/'latest.json',{})
    if ema is None and latest.get('step')==last['step']:ema=latest.get('ema_coefficients')
    x=np.asarray([row['coefficient_updates'] for row in training])
    curves=np.asarray([row['coefficients'] for row in training])
    edges=np.linspace(0,1,len(raw)+1)
    now=datetime.now(timezone.utc).isoformat()
    checkpoints=[]
    for path in sorted(root.glob('*/checkpoint_*.pt')):
        checkpoints.append(dict(step=int(path.stem.split('_')[-1]),path=str(path),
                                run=path.parent.name,bytes=path.stat().st_size))
    checkpoints.sort(key=lambda value:(value['step'],value['run']))
    atomic_json(root/'checkpoints.json',dict(updated_utc=now,retention='Keep all numbered checkpoints',
                checkpoints=checkpoints,total_bytes=sum(row['bytes'] for row in checkpoints)))
    status=read(root/'status.json',{})
    active=root/status.get('active_training',last['source_run'])
    exit_state=read(active/'exit.json',{})
    if exit_state:
        latest_active=read(active/'latest.json',{})
        status.update(phase='complete' if (active/'complete.json').exists() else
                      'paused' if latest_active.get('phase')=='paused' else 'error',
                      step=latest_active.get('step'),exit_code=exit_state['exit_code'])
        atomic_json(root/'status.json',status)
    target_updates=last['target_step']-config['warmup']
    health=dict(updated_utc=now,step=last['step'],coefficient_updates=last['coefficient_updates'],
                target_coefficient_updates=target_updates,source_run=last['source_run'],
                phase=status.get('phase'),global_batch=last['global_batch'],microbatch=last['microbatch'],
                world_size=last.get('world_size',1),coefficient_min=float(raw.min()),coefficient_max=float(raw.max()),
                coefficient_zero_or_negative_count=int((raw<=0).sum()),
                max_adjacent_abs_jump=float(np.abs(np.diff(raw)).max()),
                last_update_age_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(last['updated_utc'])).total_seconds(),
                recent_seconds_median=float(np.median([r['seconds'] for r in training[-10:] if r['source_run']==last['source_run']])),
                latest_diagnostic=probes[-1] if probes else None,checkpoint_count=len(checkpoints),
                all_coefficients_finite=bool(np.isfinite(curves).all()),
                interpretation='Activity diagnostics do not establish FID improvement or convergence.')
    atomic_json(root/'gan_health.json',health)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].stairs(np.full(len(raw),config['initial_extra_a']),edges,baseline=None,
                     color='gray',linestyle='--',label='Initial a = 0.50')
    axes[0,0].stairs(raw,edges,baseline=None,label='Current raw',color='#1764ab')
    if ema is not None:axes[0,0].stairs(ema,edges,baseline=None,label='EMA (0.99)',color='#da7c23')
    axes[0,0].set(xlabel='Sampling time: noise 0 to image 1',ylabel='Extra coefficient a (w = 1 + a)',
                  title=f"JiT 1-block: {last['coefficient_updates']} coefficient updates, step {last['step']}")
    axes[0,0].legend(fontsize=8)
    # Keep long-run refresh/render cost bounded. Original per-update values
    # remain in train.jsonl; the history heatmap displays at most 1000 rows.
    displayed=np.unique(np.linspace(0,len(x)-1,min(1000,len(x))).round().astype(int))
    im=axes[0,1].pcolormesh((edges[:-1]+edges[1:])/2,x[displayed],curves[displayed],
                           shading='nearest',cmap='viridis',rasterized=True)
    fig.colorbar(im,ax=axes[0,1],label='Extra coefficient a')
    axes[0,1].set(xlabel='Sampling time',ylabel='Coefficient update',title='Coefficient history')
    for key,label in [('schedule_gradient_norm','Total before clipping'),
                      ('gradient_first_half_norm','First half after clipping'),
                      ('gradient_second_half_norm','Last half after clipping')]:
        subset=[row for row in training if key in row]
        axes[1,0].plot([row['coefficient_updates'] for row in subset],[row[key] for row in subset],label=label,linewidth=.8)
    axes[1,0].set(yscale='log',xlabel='Coefficient update',ylabel='Gradient norm',title='Full-trajectory gradient')
    axes[1,0].legend(fontsize=8)
    for key,label in [('d_ce','D logistic loss'),('g_loss','Coefficient loss')]:
        values=np.asarray([row[key] for row in training]);width=min(10,len(values))
        axes[1,1].plot(x[width-1:],np.convolve(values,np.ones(width)/width,mode='valid'),label=label)
    axes[1,1].set(xlabel='Coefficient update',ylabel='10-update moving mean',title='Training losses (not image quality)')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    save_figure(fig,root/'schedule');plt.close(fig)
    with (root/'coefficients.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['solver_step','time_left','time_right','initial_a','raw_a','ema_a','training_step'])
        writer.writerows((i,edges[i],edges[i+1],config['initial_extra_a'],raw[i],ema[i] if ema is not None else '',last['step']) for i in range(len(raw)))
    if probes:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        groups=[(('real_probability_mean','Real before D'),('fake_probability_after_d_mean','Fake after D')),
                (('d_gradient_norm','D gradient'),('d_update_norm','D update')),
                (('g_feature_gradient_norm_mean','Feature gradient'),('g_endpoint_gradient_norm_mean','RGB endpoint gradient')),
                (('g_logit_gradient_mean','Loss derivative'),('head_update_norm','Coefficient update'))]
        titles=['Discriminator probabilities','Discriminator training','Feedback reaches RGB endpoint','Coefficient training signal']
        for ax,group,title in zip(axes.flat,groups,titles):
            for key,label in group:
                subset=[row for row in probes if key in row]
                ax.plot([row['step']-config['warmup'] for row in subset],[row[key] for row in subset],label=label)
            ax.set(xlabel='Coefficient update',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if ax is not axes[0,0]:ax.set_yscale('log')
        axes[0,0].set_ylim(0,1);save_figure(fig,root/'gan_health');plt.close(fig)
    links=''.join(f'<li>step {item["step"]}: <a href="{html.escape(str(Path(item["path"]).relative_to(root)))}">{html.escape(item["run"])}</a></li>' for item in checkpoints)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>JiT GAN 系数训练</title><style>body{{max-width:1200px;margin:24px auto;font-family:sans-serif}}img{{width:100%}}code{{background:#eee}}</style>
<h1>JiT 1-block：GAN 系数训练</h1><p>每30秒刷新 · step {last['step']} · 系数更新 {last['coefficient_updates']}/{target_updates} · {health['world_size']} GPU · global batch {last['global_batch']} · microbatch {last['microbatch']}</p>
<p>状态：{html.escape(str(health['phase']))} · 更新时间：{now} · 最近耗时中位数 {health['recent_seconds_median']:.2f} 秒/更新</p>
<img src="schedule.png?step={last['step']}" alt="系数当前值、EMA、历史、梯度及损失"><img src="gan_health.png?step={last['step']}" alt="判别器和系数训练信号">
<p>额外系数 a，总系数 w=1+a；曲线中的活动指标不代表 FID 改善。<a href="coefficients.csv">当前系数 CSV</a> · <a href="gan_health.json">诊断 JSON</a></p>
<h2>保留的完整 checkpoint</h2><ul>{links}</ul></html>'''
    temp=root/'index.tmp.html';temp.write_text(page);temp.replace(root/'index.html')
    return health


def main(args):
    while True:
        health=refresh(args.root)
        if health:publish_pngs(args.root,args.png_dir)
        if health:print(json.dumps({key:health[key] for key in ('step','phase','coefficient_updates','checkpoint_count')}),flush=True)
        if not args.watch or (health and health['phase'] in ('complete','paused','error')):break
        time.sleep(args.watch)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--watch',type=float,default=0)
    p.add_argument('--png-dir',type=Path,
                   default=Path(__file__).resolve().parents[1]/'docs/classifier_guidance/figures',
                   help='Repository directory for automatically refreshed PNG copies')
    main(p.parse_args())
