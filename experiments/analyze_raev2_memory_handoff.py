"""Audit whole-trajectory null parity and semantics of one-write memory."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
from experiments.raev2_training_core import file_sha256
from experiments.probe_raev2_memory_handoff import sha

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).resolve().parents[1]
    run=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_memory_handoff_20260908')
    other=run.parent/'fsg_condition_handoff_20260908'
    result=json.loads((run/'result.json').read_text());assert result['complete']
    for source,expected in result['sources'].items():assert file_sha256(Path(source))==expected
    memory=np.load(run/'memory.npz');zero=np.load(run/'zero_memory.npz')
    uncond=np.load(other/'unconditional.npz');conditional=np.load(other/'conditional.npz')
    for key in ['pixels','endpoint']:
        np.testing.assert_array_equal(zero[key],uncond[key][:8])
    assert np.count_nonzero(zero['memory'])==0
    for meta in result['arms']:
        data=memory if meta['arm']=='memory' else zero
        for key,field in [('pixels','pixel_sha256'),('endpoint','endpoint_sha256'),('memory','memory_sha256')]:
            assert sha(data[key])==meta[field]
    assert sha(uncond['written'])==result['noise_sha256']
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model=convnext_tiny(weights=weights).cuda().eval();preprocess=weights.transforms()
    labels=np.array(result['labels']);summary=[];probs={}
    for name,data in [('memory',memory),('unconditional',uncond),('conditional',conditional)]:
        values=[]
        for start in range(0,64,16):
            x=torch.from_numpy(data['pixels'][start:start+16]).permute(0,3,1,2)
            values.append(model(preprocess(x).cuda()).softmax(-1).cpu().numpy())
        p=np.concatenate(values);probs[name]=p;rank=np.argsort(-p,axis=1)
        summary.append(dict(arm=name,top1=float(np.mean(rank[:,0]==labels)),
                            top5=float(np.mean(np.any(rank[:,:5]==labels[:,None],axis=1))),
                            target_probability=float(np.mean(p[np.arange(64),labels]))))
    np.savez(run/'classifier_probabilities.npz',**probs,labels=labels)
    result.update(classification=summary,zero_memory_parity='8 entire latent endpoints and decoded uint8 images bitwise',
                  classifier='torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1 with official preprocessing',
                  classifier_sha256=file_sha256(Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')),
                  probability_file_sha256=file_sha256(run/'classifier_probabilities.npz'),
                  evaluator_source_sha256=file_sha256(Path(__file__)))
    (root/'experiments/results/terminal_defect_20260908/raev2_memory_handoff.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(classification=summary,parity=result['zero_memory_parity']),indent=2))

if __name__=='__main__':main()
