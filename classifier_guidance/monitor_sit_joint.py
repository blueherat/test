"""CPU-only live PNGs for joint SiT guidance, published as real repository files."""
import argparse
import json
from pathlib import Path
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows


def publish(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    fig.savefig(temporary,format='png',dpi=145,bbox_inches='tight')
    temporary.replace(path)
    plt.close(fig)


def render(runs,output):
    merged={}
    profiles=[]
    for run in runs:
        for row in read_rows(run/'train.jsonl'):merged[row['step']]=row
        profiles.extend(read_rows(run/'profiles.jsonl'))
    rows=[merged[k] for k in sorted(merged)]
    if not rows:return
    x=np.array([r['step'] for r in rows]);latest=rows[-1]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    t=(np.arange(64)+.5)/64
    # A standalone scale figure follows the pure-scale experiment layout.
    schedule_fig,schedule_axes=plt.subplots(2,2,figsize=(12,8))
    schedule_axes[0,0].plot(t,latest['coefficients'],label='Raw')
    schedule_axes[0,0].plot(t,latest['ema_coefficients'],label='EMA')
    schedule_axes[0,0].axhline(.75,color='gray',ls='--',label='Initial')
    schedule_axes[0,0].set_title('Latest signed extra coefficient');schedule_axes[0,0].legend()
    for index in np.unique(np.linspace(0,len(rows)-1,min(6,len(rows))).astype(int)):
        schedule_axes[0,1].plot(t,rows[index]['coefficients'],label=f"Step {rows[index]['step']}")
    schedule_axes[0,1].set_title('Schedule evolution');schedule_axes[0,1].legend(fontsize=8)
    stride=max(1,len(rows)//1000)
    schedule_heat=np.array([r['coefficients'] for r in rows[::stride]])
    heat_image=schedule_axes[1,0].imshow(schedule_heat,aspect='auto',origin='lower',
        extent=[0,1,x[0]-.5,x[-1]+.5],cmap='coolwarm')
    schedule_fig.colorbar(heat_image,ax=schedule_axes[1,0]);schedule_axes[1,0].set_ylabel('Global step')
    for key in ('coefficient_min','coefficient_max'):
        schedule_axes[1,1].plot(x,[r[key] for r in rows],label=key)
    schedule_axes[1,1].set_xlabel('Global step');schedule_axes[1,1].legend()
    for ax in (schedule_axes[0,0],schedule_axes[0,1],schedule_axes[1,0]):ax.set_xlabel('Sampler time (noise -> image)')
    schedule_fig.suptitle(f"SiT joint guidance | step {latest['step']} | joint updates {latest['joint_updates']} | GPUs {latest.get('world_size',1)}")
    schedule_fig.tight_layout();publish(schedule_fig,output/'sit_joint_schedule.png')
    axes[0,0].plot(t,latest['coefficients'],label='Raw')
    axes[0,0].plot(t,latest['ema_coefficients'],label='EMA')
    axes[0,0].axhline(.75,color='gray',ls='--',label='Initial 0.75')
    axes[0,0].axhline(0,color='gray',lw=.5)
    axes[0,0].set(xlabel='Sampler time (noise -> image)',ylabel='Signed extra coefficient a')
    axes[0,0].legend()
    stride=max(1,len(rows)//1000)
    heat=np.array([r['coefficients'] for r in rows[::stride]])
    im=axes[0,1].imshow(heat,aspect='auto',origin='lower',extent=[0,1,x[0]-.5,x[-1]+.5],cmap='coolwarm')
    axes[0,1].set(xlabel='Sampler time',ylabel='Global training step')
    fig.colorbar(im,ax=axes[0,1])
    if profiles:
        profile=max(profiles,key=lambda p:p['step'])
        ps=profile['rows'];pt=[p['time'] for p in ps]
        axes[1,0].plot(pt,[p['gap_rms'] for p in ps],label='Learned gap RMS')
        axes[1,0].plot(pt,[p['reference_gap_rms'] for p in ps],label='Initial gap RMS')
        axes[1,0].legend();axes[1,0].set_title(f"Fresh real-interpolant probe, step {profile['step']}")
        axes[1,1].plot(pt,[p['correction_rms'] for p in ps],label='|a| * gap RMS')
        axes[1,1].plot(pt,[.75*p['reference_gap_rms'] for p in ps],label='Initial correction RMS')
        axes[1,1].legend()
    for ax in axes[1]:ax.set_xlabel('Probe time');ax.set_ylabel('RMS')
    fig.suptitle(f"SiT 1-block joint head + scale | step {latest['step']} | joint updates {latest['joint_updates']}")
    fig.tight_layout();publish(fig,output/'sit_joint_guidance.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    groups=[('d_ce','g_loss'),('real_accuracy','fake_accuracy'),
            ('weak_gan_gradient_norm','norm_gradient_norm','coefficient_gradient_norm'),
            ('weak_update_norm','coefficient_update_norm','d_update_norm'),
            ('gap_ratio','gap_cosine'),('seconds','peak_allocated_gib')]
    for ax,keys in zip(axes.flat,groups):
        for key in keys:
            selected=[(r['step'],r[key]) for r in rows if key in r]
            if selected:ax.plot(*zip(*selected),label=key,lw=.8)
        if ax.lines:ax.legend(fontsize=8)
        ax.set_xlabel('Global step');ax.grid(alpha=.2)
    axes[0,2].set_yscale('symlog',linthresh=1e-8)
    axes[1,0].set_yscale('log')
    fig.suptitle('SiT joint GAN health (training diagnostics, not image-quality evaluation)')
    fig.tight_layout();publish(fig,output/'sit_joint_gan_health.png')
    status=dict(step=latest['step'],joint_updates=latest['joint_updates'],updated_utc=latest['updated_utc'],
                runs=[str(r) for r in runs])
    temporary=output/f'sit_joint_status.{os.getpid()}.tmp';temporary.write_text(json.dumps(status,indent=2))
    temporary.replace(output/'sit_joint_status.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/classifier_guidance/figures')
    p.add_argument('--interval',type=float,default=30)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    while True:
        try:render(args.runs,args.output)
        except Exception as exc:print(f'Monitor error: {exc}',flush=True)
        if args.once:break
        time.sleep(args.interval)
