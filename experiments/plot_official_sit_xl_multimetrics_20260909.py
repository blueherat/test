"""Plot fixed exploratory semantic, coverage, and kernel-distance readouts."""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.raev2_training_core import file_sha256


def main():
    out=ROOT/'docs/data/guidance_goal_20260909'
    source=out/'official_sit_xl_multimetrics.csv'
    audit=out/'official_sit_xl_multimetrics_audit.json'
    assert json.loads(audit.read_text())['passed']
    with source.open() as f: rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.1),layout='constrained')
    names={'ordinary':'IG, 115 full','time_only':'PFR time, 100 full + 50 prefix','projected':'PFR projected, 100 full + 50 prefix'}
    for method,label in names.items():
        rr=sorted((r for r in rows if r['method']==method),key=lambda r:float(r['scale']))
        scale=[float(r['scale']) for r in rr]
        axes[0].plot(scale,[100*float(r['target_top1']) for r in rr],marker='o',label=label)
        axes[1].plot([100*float(r['precision']) for r in rr],[100*float(r['recall']) for r in rr],marker='o',label=label)
        axes[2].plot(scale,[1000*float(r['kid_u_statistic']) for r in rr],marker='o',label=label)
    for r,marker in zip([r for r in rows if r['method']=='native'],['D','X']):
        axes[1].scatter(100*float(r['precision']),100*float(r['recall']),marker=marker,s=65,
                        color='black' if r['vae']=='ema' else 'grey',label='SDE250 '+r['vae'].upper()+' (higher cost)')
    axes[0].set(xlabel='IG scale',ylabel='ConvNeXt target top-1 (%)')
    axes[1].set(xlabel='Inception precision (%)',ylabel='Inception recall (%)')
    axes[2].set(xlabel='IG scale',ylabel='KID U-statistic × 1,000')
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(fontsize=8,loc='lower right')
    handles,labels=axes[1].get_legend_handles_labels()
    axes[1].legend(handles[-2:],labels[-2:],fontsize=8,loc='lower right')
    fig.suptitle('Official SiT-XL/2: reused 1K discovery bank, 10K reference images',fontsize=13)
    paths=[out/'official_sit_xl_multimetrics.png',out/'official_sit_xl_multimetrics.pdf',out/'official_sit_xl_multimetrics_plot.json']
    assert not any(p.exists() for p in paths)
    for p in paths[:2]: fig.savefig(p,dpi=180)
    paths[2].write_text(json.dumps(dict(source_sha256=file_sha256(source),audit_sha256=file_sha256(audit),
        plot_source_sha256=file_sha256(Path(__file__)),outputs={str(p):file_sha256(p) for p in paths[:2]},
        limitation='No uncertainty estimate; sparse unequal-size PR and class-stratified KID. SDE has higher cost.'),indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__': main()
