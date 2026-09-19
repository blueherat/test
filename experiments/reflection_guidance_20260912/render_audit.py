"""Same existing trajectories: compare latent reflection with pixel reflection."""
from pathlib import Path
import gc
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c,evaluate
from . import core as m

PROTOCOL=c.WORK/'docs/REFLECTION_RENDER_AUDIT_20260912_ZH.md'
STAGE='render_controls'


@torch.inference_mode()
def run(model):
    m.configure();rt=c.runtime(model);before=rt.counts.copy();results=[]
    for track in ('cfg','ig'):
        source=m.ROOT/model/m.STAGE/(track+'_fixed_flip')
        while not (source/'summary.json').exists():
            state=c.read(m.ROOT/'status.json')
            if state['phase']!='running':raise RuntimeError('Missing source after controller completed')
            if not Path('/proc',str(state['pid'])).exists():raise RuntimeError('Source sampler controller missing')
            time.sleep(5)
        origin=c.read(source/'summary.json');root=m.ROOT/model/STAGE/(track+'_physical_flip')
        root.mkdir(parents=True,exist_ok=True)
        if not (root/'summary.json').exists():
            images=[];labels=[];seconds=0.;mae=[]
            for record in origin['records']:
                p=Path(record['file']);assert c.sha(p)==record['sha256']
                with np.load(p) as d:
                    z=c.cuda(d['latents']);native_pixels=d['arr_0'];labels.append(d['labels'])
                torch.cuda.synchronize();start=time.perf_counter()
                pix=rt.decode(torch.flip(z,(-1,)))[:,:,::-1,:].copy()
                torch.cuda.synchronize();seconds+=time.perf_counter()-start
                images.append(pix);mae.append(float(np.abs(pix.astype('float64')-native_pixels.astype('float64')).mean()/255))
            pixels=np.concatenate(images);assert len(pixels)==400
            np.savez(root/'samples.npz',arr_0=pixels);np.save(root/'labels.npy',np.concatenate(labels))
            c.atomic(root/'summary.json',dict(complete=True,model=model,stage=STAGE,arm=track+'_physical_flip',
                primary_samples=400,generated_paths=0,reused_source_paths=400,extra_rendered_views=400,
                incremental_full_calls=0,incremental_prefix_calls=0,full_calls_per_output=origin['full_calls_per_output'],
                prefix_calls_at_inference=0,seconds=seconds,mean_absolute_render_difference=float(np.mean(mae)),
                samples_sha256=c.sha(root/'samples.npz'),source_summary_sha256=c.sha(source/'summary.json'),
                source_summary=str(source/'summary.json'),protocol_sha256=c.sha(PROTOCOL),
                source_sha256=c.sha(Path(__file__)),source_records=origin['records'],diagnostic_only=True))
        results.append(evaluate.evaluate(model,STAGE,track+'_physical_flip'))
        c.atomic(m.ROOT/model/STAGE/'results.json',results)
    assert rt.counts==before
    c.atomic(m.ROOT/model/STAGE/'complete.json',dict(complete=True,new_full_calls=0,new_prefix_calls=0,rendered_views=800))
    del rt;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    for model in ('sit_small','raev2'):run(model)
