"""Finite latent-vs-pixel extrapolation discrepancy on frozen native readouts."""
import json,time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

@torch.inference_mode()
def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908')
    out=root/'commutator';out.mkdir(exist_ok=False)
    data=json.loads((root/'results.json').read_text());assert data['complete']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG);native.install_raev2_decoder_config_compat()
    assets=Path('/home/zhoushunyu/data/eqvae/models/RAEv2/stage1/imagenet/dinov3l-k7')
    identity={str(assets/n):native.file_sha256(assets/n) for n in ['decoder.pt','stats.pt']}
    decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False);del decoder.encoder
    rows=[];files={};started=time.perf_counter()
    for i in data['request']['ids']:
        for k in data['request']['steps']:
            p=root/f'id{i:04d}_step{k:03d}.npz';assert native.file_sha256(p)==data['files'][p.name]
            d=np.load(p);b=d['base'];f=d['full'];g=b+np.float32(1.78)*(f-b)
            raw=decoder.decode(torch.from_numpy(np.stack([b,f,g])).cuda()).cpu().numpy()
            dest=out/p.name;np.savez_compressed(dest,raw_rgb=raw);files[dest.name]=native.file_sha256(dest)
            v=raw.astype(float).reshape(3,-1)
            actual=v[2]-v[1];linear=.78*(v[1]-v[0]);residual=actual-linear
            mat=np.stack([actual,linear,residual]);gram=mat@mat.T/mat.shape[1]
            rows.append(dict(id=i,step=k,gram=gram.tolist()))
    result=dict(complete=True,rows=rows,files=files,seconds=time.perf_counter()-started,decoder_batch_calls=32,
                decoder_assets_sha256_before_loading=identity,source_sha256=native.file_sha256(Path(__file__)),
                source_prediction_result_sha256=native.file_sha256(root/'results.json'))
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(complete=True,seconds=result['seconds'])))
if __name__=='__main__':main()
