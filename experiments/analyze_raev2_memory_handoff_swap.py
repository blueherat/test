"""Does generated class follow the donor memory after external-label withdrawal?"""
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
    run=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_memory_handoff_swap_20260908')
    result=json.loads((run/'result.json').read_text());assert result['complete']
    for source,value in result['sources'].items():assert file_sha256(Path(source))==value
    assert file_sha256(Path(result['donor_file']))==result['donor_file_sha256']
    data=np.load(run/'swapped_memory.npz');original=np.load(result['donor_file'])
    np.testing.assert_array_equal(data['memory'],np.roll(original['memory'],-1,axis=0))
    for key,field in [('pixels','pixel_sha256'),('endpoint','endpoint_sha256'),('memory','memory_sha256')]:
        assert sha(data[key])==result['arms'][0][field]
    labels=np.array(result['labels']);donors=np.array(result['donor_labels'])
    np.testing.assert_array_equal(donors,np.roll(labels,-1))
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model=convnext_tiny(weights=weights).cuda().eval();preprocess=weights.transforms();values=[]
    for start in range(0,64,16):
        x=torch.from_numpy(data['pixels'][start:start+16]).permute(0,3,1,2)
        values.append(model(preprocess(x).cuda()).softmax(-1).cpu().numpy())
    p=np.concatenate(values);rank=np.argsort(-p,axis=1);scores={}
    for name,target in [('original',labels),('donor',donors)]:
        scores[name]=dict(top1=float(np.mean(rank[:,0]==target)),
                          top5=float(np.mean(np.any(rank[:,:5]==target[:,None],axis=1))),
                          target_probability=float(np.mean(p[np.arange(64),target])))
    np.savez(run/'classifier_probabilities.npz',probabilities=p,labels=labels,donors=donors)
    result.update(classification=scores,classifier='ConvNeXt_Tiny_Weights.IMAGENET1K_V1, official preprocessing',
                  evaluator_sha256=file_sha256(Path(__file__)),probabilities_sha256=file_sha256(run/'classifier_probabilities.npz'))
    (root/'experiments/results/terminal_defect_20260908/raev2_memory_handoff_swap.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(scores,indent=2))

if __name__=='__main__':main()
