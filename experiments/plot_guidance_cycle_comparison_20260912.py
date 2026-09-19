"""Separate exploratory 1K and independent 5K comparisons in a shareable figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path('/home/zhoushunyu/eqvae')
BASE=Path('/home/zhoushunyu/data/eqvae/experiments')


def load(root,stage):
    p=BASE/root/stage
    audit=json.loads((p/'analysis_audit.json').read_text());assert audit['passed']
    rows=json.loads((p/'results.json').read_text());assert all(r['complete'] for r in rows)
    return {r['arm']:r for r in rows}


def run():
    screen=load('sit_prefix_coarse_20260912','prefix_screen_1k')
    confirm=load('sit_prefix_confirmation_20260912','prefix_confirm_5k')
    confirm.update(load('sit_prefix_incumbent_20260912','prefix_incumbent_5k'))
    names=['ig_prefix_original_00','ig_prefix_local_00','ig_local_residual_11','ig_prefix_native_00','ig_prefix_clt_00']
    labels=['Original IG','Local IG','Existing fused IG','Native weak prefix','Coarsened weak prefix']
    palette=['#768495','#495a70','#33485e','#167c70','#b65a45']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,pool,title in zip(axes,(screen,confirm),('Exploratory paired 1K','Independent paired 5K')):
        use=[i for i,n in enumerate(names) if n in pool]
        values=[pool[names[i]]['fid'] for i in use]
        bars=ax.barh(range(len(use)),values,color=[palette[i] for i in use],height=.6)
        ax.set_yticks(range(len(use)),[labels[i] for i in use]);ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.16);ax.set_xlabel('FID (lower is better)');ax.set_title(title,loc='left',weight='bold')
        for bar,value in zip(bars,values):
            ax.text(value+.01*max(values),bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Fixed weak-prefix experiments: screening and independent confirmation',fontsize=13,weight='bold')
    fig.supxlabel('Compare methods within each panel. Sample counts and noise banks differ.\nFixed weights and guidance settings; no error bars from a single confirmation bank.',fontsize=9)
    folder=WORK/'docs/figures/guidance_research_cycles_20260912';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(folder/f'prefix_screen_and_confirmation.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':run()
