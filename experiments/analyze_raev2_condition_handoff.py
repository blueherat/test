"""Independent image semantics and paired continuation audit, no FID claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
from experiments.raev2_training_core import file_sha256

def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).resolve().parents[1]
    run=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_condition_handoff_20260908')
    result=json.loads((run/'result.json').read_text())
    assert result['complete'] and len(result['arms'])==6
    for source,value in result['sources'].items():assert file_sha256(Path(source))==value
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    classifier=convnext_tiny(weights=weights).cuda().eval()
    transform=weights.transforms()
    labels=np.array(result['labels']);donor=(labels+1)%1000
    arrays={};summary=[];probabilities={}
    for arm in result['arms']:
        name=arm['arm'];data=np.load(run/f'{name}.npz')
        arrays[name]={k:data[k] for k in data.files}
        for key in ['pixels','endpoint','written']:
            assert np.isfinite(arrays[name][key]).all()
            field='pixel_sha256' if key=='pixels' else key+'_sha256'
            assert sha(arrays[name][key])==arm[field]
        p=[]
        for start in range(0,64,16):
            x=torch.from_numpy(arrays[name]['pixels'][start:start+16]).permute(0,3,1,2)
            p.append(classifier(transform(x).cuda()).softmax(-1).cpu().numpy())
        p=np.concatenate(p);probabilities[name]=p
        rank=np.argsort(-p,axis=1)
        summary.append(dict(arm=name,target_top1=float(np.mean(rank[:,0]==labels)),
                            target_top5=float(np.mean(np.any(rank[:,:5]==labels[:,None],axis=1))),
                            donor_top1=float(np.mean(rank[:,0]==donor)),
                            target_probability=float(np.mean(p[np.arange(64),labels])),
                            donor_probability=float(np.mean(p[np.arange(64),donor]))))
    assert np.array_equal(arrays['calibrate_drop']['written'],arrays['calibrate_keep']['written'])
    assert np.array_equal(arrays['conditional']['written'],arrays['unconditional']['written'])
    def mse(a,b):return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    pairs={}
    for before,after in [('unconditional','conditional'),('calibrate_drop','calibrate_keep'),
                         ('calibrate_drop','conditional'),('unconditional_roundtrip','conditional')]:
        pairs[before+'__'+after]=mse(arrays[before]['endpoint'],arrays[after]['endpoint'])
    result.update(classification=summary,endpoint_pair_mse=pairs,
                  classifier='torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1; official resize/crop/normalize; uint8 decoded pixels',
                  classifier_sha256=file_sha256(Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')),
                  evaluator_source_sha256=file_sha256(Path(__file__)))
    result['endpoint_across_64_class_variance_per_coordinate']={
        name:float(np.var(a['endpoint'].astype(np.float64),axis=0).mean()) for name,a in arrays.items()}
    result['decoder_assets_audited_after_sampling']={str(p):file_sha256(p) for p in [
        Path('/home/zhoushunyu/data/eqvae/models/RAEv2/stage1/imagenet/dinov3l-k7/decoder.pt'),
        Path('/home/zhoushunyu/data/eqvae/models/RAEv2/stage1/imagenet/dinov3l-k7/stats.pt')]}
    np.savez(run/'classifier_probabilities.npz',**probabilities,labels=labels)
    result['probabilities_file_sha256']=file_sha256(run/'classifier_probabilities.npz')
    out=root/'experiments/results/terminal_defect_20260908/raev2_condition_handoff.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arrays['memory']=dict(np.load(run.parent/'fsg_memory_handoff_20260908/memory.npz'))
    arrays['swapped_memory']=dict(np.load(run.parent/'fsg_memory_handoff_swap_20260908/swapped_memory.npz'))
    fig,axes=plt.subplots(8,8,figsize=(16,16))
    for row,(name,a) in enumerate(arrays.items()):
        for col in range(8):
            axes[row,col].imshow(a['pixels'][col]);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f'{labels[col]}: {weights.meta["categories"][labels[col]][:18]}',fontsize=8)
            if col==0:axes[row,col].set_ylabel(name,fontsize=7)
    fig.tight_layout();fig.savefig(run/'contact_sheet.png',dpi=120);plt.close(fig)
    print(json.dumps(dict(classification=summary,endpoint_pair_mse=pairs),indent=2))

if __name__=='__main__':main()
