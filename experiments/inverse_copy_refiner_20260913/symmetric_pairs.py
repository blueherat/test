"""Add inverse transforms C_tau^{-1} to the same frozen real-image pair bank.

Mixing both directions removes a leading one-sided transformation bias in a
small-transform Bayes analysis; it does not guarantee quality or stationarity.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.inverse_copy_refiner_20260913.pairs import ROOT, Fields, flow, atomic, sha


@torch.inference_mode()
def main():
    output=ROOT/'symmetric_pairs'
    output.mkdir(exist_ok=False)
    original=np.load(ROOT/'pairs.npz')
    parent=json.loads((ROOT/'pairs_summary.json').read_text())
    assert parent['complete'] and parent['pairs_sha256']==sha(ROOT/'pairs.npz')
    atomic(output/'request.json',dict(operation='C plus inverse C, equal pair weighting',
        corruption_order=['C+.75','C+.5','C-.75','C-.5'],
        real_bank_unchanged=True, original_pairs_sha256=parent['pairs_sha256'],
        script_sha256=sha(__file__), pairs_source_sha256=sha(Path(__file__).with_name('pairs.py')),
        steps_grid=64, new_noise=False, additional_branch_calls_per_source=160,
        total_construction_calls_per_source_including_reused_pairs=320,
        source_targets='same original real images',quality_claim=False))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    start=time.monotonic()
    fields=Fields('cuda:0')
    atomic(output/'models.json',fields.metadata)
    values=[]
    for begin in range(0,len(original['clean']),16):
        clean=torch.from_numpy(original['clean'][begin:begin+16]).cuda()
        labels=torch.from_numpy(original['labels'][begin:begin+16]).cuda()
        before=fields.calls
        # This reverses each cross-model copy map: weak backward, strong forward.
        at75=flow(fields,clean,labels,'weak',1.,.75,64)
        at50=flow(fields,at75,labels,'weak',.75,.5,64)
        out75=flow(fields,at75,labels,'strong',.75,1.,64)
        out50=flow(fields,at50,labels,'strong',.5,1.,64)
        assert fields.calls-before==160
        values.append(torch.stack((out75,out50),dim=1).cpu().numpy())
        atomic(output/'status.json',dict(status='running',completed=begin+len(clean),total=500))
    reverse=np.concatenate(values)
    merged=np.concatenate((original['corrupted'],reverse),axis=1)
    np.savez(output/'pairs.npz',clean=original['clean'],corrupted=merged,
        labels=original['labels'],split=original['split'],source_ids=original['source_ids'],
        indices=original['indices'],taus=np.array([.75,.5,.75,.5]),directions=np.array([1,1,-1,-1],dtype=np.int8))
    summary=dict(complete=True,real_sources=500,corrupted_pairs=2000,
        pairs_sha256=sha(output/'pairs.npz'),original_pairs_bitwise_preserved=bool(np.array_equal(merged[:,:2],original['corrupted'])),
        mean_corruption_mse=((merged.astype(np.float64)-original['clean'][:,None])**2).mean(axis=(0,2,3,4)).tolist(),
        extra_equivalent_branch_image_evaluations=500*160,seconds=time.monotonic()-start)
    atomic(output/'summary.json',summary)
    atomic(output/'status.json',dict(status='complete',completed=500,total=500))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
