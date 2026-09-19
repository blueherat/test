"""Display predetermined class-spaced examples without selecting favorable images."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/home/zhoushunyu/data/eqvae/experiments')
OUT=ROOT/'docs/data/guidance_goal_20260909'
INDICES=(0,125,250,375,500,625,750,875)
SOURCES={
    'IG 1.35':RAW/'official_sit_pfr_20260908/ordinary115/quality/samples.npz',
    'IG 1.75':RAW/'official_sit_baseline_control_20260909/ig175_all/samples.npz',
    'PFR time 1.75':RAW/'official_sit_pfr_symmetric_20260909/time175/samples.npz',
    'PFR projected 1.75':RAW/'official_sit_pfr_symmetric_20260909/projected175/samples.npz',
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((RAW/'official_sit_pfr_symmetric_20260909/status.json').read_text())['phase']=='complete'
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'official_sit_pfr_fixed_examples.png'
    manifest_path=OUT/'official_sit_pfr_fixed_examples.json'
    assert not output.exists() and not manifest_path.exists()
    fig,axes=plt.subplots(len(INDICES),len(SOURCES),figsize=(11,21.3))
    manifest=dict(indices=INDICES,selection='Every 125th class, fixed before image inspection; no image filtering.',sources={})
    for col,(name,path) in enumerate(SOURCES.items()):
        metrics=json.loads((path.parent/'fid.json').read_text())[0]
        digest=sha(path);assert digest==metrics['sample_sha256']
        with np.load(path) as f:
            x=f['arr_0'];assert x.shape==(1000,256,256,3) and x.dtype==np.uint8
            selected=x[list(INDICES)]
        manifest['sources'][name]=dict(path=str(path),sha256=digest,
            selected_pixel_sha256=hashlib.sha256(selected.tobytes()).hexdigest())
        for row,idx in enumerate(INDICES):
            axes[row,col].imshow(selected[row],interpolation='nearest')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(name,fontsize=10)
            if col==0: axes[row,col].set_ylabel(f'Class/index {idx}',fontsize=8)
            for spine in axes[row,col].spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.09,right=.998,bottom=.002,top=.98,hspace=.025,wspace=.025)
    fig.savefig(output,dpi=105)
    plt.close(fig)
    manifest.update(figure=str(output),figure_sha256=sha(output),source_sha256=sha(Path(__file__)),
        limitation='Eight illustrative paired examples; no population-level visual-quality or mechanism claim.')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
