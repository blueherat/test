"""Portable results and a static research figure from audited experiments."""
from pathlib import Path
import csv
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA=Path('/home/zhoushunyu/data/eqvae/experiments')
OUT=Path('docs/data/guidance_shared_budget_20260912')
SPECS=(('objective','sit_guided_objective_20260912','objective_screen_1k'),
       ('data_source','sit_sampler_reference_20260912','sampler_reference_screen_1k'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    combined=[];by_cycle={};inputs={}
    for cycle,folder,stage in SPECS:
        root=DATA/folder;base=root/stage
        status=json.loads((root/'status.json').read_text())
        assert status['phase']=='complete' and status['final_audit_passed']
        rows=json.loads((base/'results.json').read_text());by_cycle[cycle]={r['arm']:r for r in rows}
        for path in (base/'request.json',base/'results.json',root/'screen_review.json',root/'status.json'):
            inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in rows:
            assert r['complete']
            source=r['source'];control=source+('_native_00' if cycle=='objective' else '_real_00')
            combined.append(dict(cycle=cycle,source=source,arm=r['arm'],samples=1000,
                fid=r['fid'],sfid=r['metrics']['sfid'],inception_score=r['metrics']['inception_score'],
                gpu_seconds=r['sum_batch_gpu_seconds'],full_calls=r['full_calls_per_image'],
                prefix_calls=r['prefix_calls_per_image'],matched_control=control,
                delta_fid_vs_matched_control=r['fid']-by_cycle[cycle][control]['fid']))
    with (OUT/'all_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(11.5,3.6),gridspec_kw={'width_ratios':[1,1.8]})
    display=[('objective',[('ig_direct_00','IG: direct objective'),('cfg_direct_00','CFG: direct objective')]),
        ('data_source',[('ig_ig_00','IG: IG-generated data'),('ig_cfg_00','IG: CFG-generated data'),
                        ('cfg_ig_00','CFG: IG-generated data'),('cfg_cfg_00','CFG: CFG-generated data')])]
    for ax,(cycle,arms) in zip(axs,display):
        records=[next(r for r in combined if r['cycle']==cycle and r['arm']==arm) for arm,_ in arms]
        values=[r['delta_fid_vs_matched_control'] for r in records]
        ax.barh(range(len(arms)),values,color=['#38708f' if v<0 else '#ad675c' for v in values],height=.53)
        ax.set_yticks(range(len(arms)),[label for _,label in arms]);ax.invert_yaxis()
        ax.axvline(0,color='#555555',linewidth=.8)
        ax.axvline(-.5,color='#777777',linewidth=.7,linestyle='--')
        ax.grid(axis='x',color='#e6e6e6',linewidth=.5);ax.set_axisbelow(True)
        span=max(1.,max(values)-min(values));low=min(-.8,min(values)-.25*span);high=max(.5,max(values)+.25*span)
        ax.set_xlim(low,high)
        for i,value in enumerate(values):
            ax.text(value+(.025*span if value>=0 else -.025*span),i,f'{value:+.3f}',
                ha='left' if value>=0 else 'right',va='center',fontsize=9)
        ax.set_xlabel('FID change vs matched real-data continuation (lower is better)')
    axs[0].set_title('Cycle 1: change the training objective',loc='left',fontsize=11,pad=15)
    axs[1].set_title('Cycle 2: change the training data source',loc='left',fontsize=11,pad=15)
    fig.suptitle('Fixed-budget guidance experiments',x=.02,ha='left',fontsize=15)
    fig.text(.02,.015,'Paired exploratory 1K; no confidence intervals. The -0.5 line is a progression margin, not a significance threshold.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.08,1,.89),w_pad=3.)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'comparison.{ext}',dpi=180,bbox_inches='tight')
    (OUT/'artifact_manifest.json').write_text(json.dumps(dict(inputs=inputs,rows=len(combined),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Audited paired exploratory 1K, with matched controls; no statistical significance claim.'),indent=2)+'\n')
    # Correct a copied navigation link in mutable reports; experiment source
    # snapshots and protocols remain frozen.
    for path in (Path('docs/SIT_SAMPLER_REFERENCE_RESULTS_20260912_ZH.md'),
        DATA/'sit_sampler_reference_20260912/sampler_reference_screen_1k/report.md'):
        text=path.read_text().replace('SIT_GUIDED_OBJECTIVE_PROTOCOL_20260912_ZH.md',
            'SIT_SAMPLER_REFERENCE_PROTOCOL_20260912_ZH.md')
        path.write_text(text)
    print(json.dumps({'rows':len(combined),'csv':str(OUT/'all_results.csv'),'figure':str(OUT/'comparison.png')}))


if __name__=='__main__':main()
