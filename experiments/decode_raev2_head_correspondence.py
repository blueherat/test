"""Render frozen head predictions, not completed generations or FID samples."""
import json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments import sample_raev2_pfr_retiming as native

@torch.inference_mode()
def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908')
    out=root/'decoded';out.mkdir(exist_ok=False)
    data=json.loads((root/'results.json').read_text());assert data['complete']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG);native.install_raev2_decoder_config_compat()
    decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False);del decoder.encoder
    rows=[];started=time.perf_counter();files={}
    for i in data['request']['ids']:
        canvas=Image.new('RGB',(768,4*280),(245,245,245));draw=ImageDraw.Draw(canvas)
        for j,k in enumerate(data['request']['steps']):
            p=root/f'id{i:04d}_step{k:03d}.npz';assert native.file_sha256(p)==data['files'][p.name]
            v=np.load(p);b=v['base'];f=v['full'];g=b+np.float32(1.78)*(f-b)
            z=torch.from_numpy(np.stack([b,f,g])).cuda()
            raw=decoder.decode(z)
            pix=raw.clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
            for n,name in enumerate(['Base','Full','IG clean']):
                im=Image.fromarray(pix[n]);dest=out/f'id{i:04d}_step{k:03d}_{name.replace(" ","_")}.png';im.save(dest)
                files[dest.name]=native.file_sha256(dest)
                canvas.paste(im,(n*256,j*280+24));draw.text((n*256+5,j*280+5),f'id {i} step {k} {name}',fill=(0,0,0))
                rows.append(dict(id=i,step=k,head=name,clipped_fraction=float(((raw[n]<0)|(raw[n]>1)).float().mean())))
        dest=out/f'id{i:04d}_sheet.png';canvas.save(dest);files[dest.name]=native.file_sha256(dest)
    result=dict(complete=True,images=96,decoder_batch_calls=32,seconds=time.perf_counter()-started,rows=rows,files=files,
                source_sha256=native.file_sha256(Path(__file__)),source_results_sha256=native.file_sha256(root/'results.json'),
                warning='Frozen clean readouts at intermediate states; not sampled endpoints, not a quality/FID benchmark.')
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(complete=True,seconds=result['seconds'],images=96)))
if __name__=='__main__':main()
