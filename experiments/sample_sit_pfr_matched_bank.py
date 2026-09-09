"""Original PFR and ordinary DOPRI5 on the clock study's exact RNG bank."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from torchdiffeq import odeint
from imagenet100_sit_multiscale_models import evaluate_internal_head_only
from internal_guidance_path_extrapolation import (
    split_internal_guidance, project_to_forward_ray, affine_counterfactual_ratio_velocity)


def main():
    import sample_imagenet100_sit_foresight_fixed_point as sampler
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--reference-method',choices=['ordinary','pfr'],required=True)
    p.add_argument('--reference-solver',choices=['dopri5','euler'],required=True)
    own,remaining=p.parse_known_args()
    args=sampler.build_parser().parse_args(remaining)
    assert args.method=='closed' and args.foresight_schedule==''
    assert args.precision=='fp32' and args.diagnostic_samples==0
    if own.reference_method=='pfr':
        assert args.family=='ig' and args.ig_depths==(4,)
        assert args.ig_gamma_segments==((.25,.6),(.5,.7),(1.,0.))
        original_builder=sampler.build_ig_fields
        def builder(model,labels,**kwargs):
            fields=original_builder(model,labels,**kwargs)
            heads=kwargs['heads'];name,spec=next(iter(heads.items()))
            assert len(heads)==1 and spec.depth==4
            fields.counters['pfr_depth4_prefix_forwards']=0
            def guided(t,z):
                times=t.expand(len(z)).contiguous()
                full,values,_=sampler.evaluate_source_with_heads(model,z,times,labels[:len(z)],
                    heads=heads,source_semantics=kwargs['strong_semantics'])
                fields.counters['guided_shared_backbone_forwards']+=1
                weak=values[name];tv=float(t)
                gamma=.6 if tv<.25 else (.7 if tv<.5 else 0.)
                ordinary=full+gamma*(full-weak)
                if gamma==0:return ordinary
                h=min(1/32,.5-tv)
                if h<=0:return ordinary
                base,calibration=split_internal_guidance(full,weak,gamma=gamma)
                query=z+h*project_to_forward_ray(calibration,ordinary).parallel
                future_time=(t+t.new_tensor(h)).expand(len(z)).contiguous()
                future=evaluate_internal_head_only(model,query,future_time,labels[:len(z)],spec=spec)
                fields.counters['pfr_depth4_prefix_forwards']+=1
                return affine_counterfactual_ratio_velocity(full,base,(future,),(1.,),gamma=gamma)
            fields.guided=guided
            fields.metadata['reference_method']='historical projected future reference h=1/32'
            return fields
        sampler.build_ig_fields=builder
    if own.reference_solver=='dopri5':
        def integrate(noise,*,fields,**kwargs):
            result=odeint(fields.guided,noise,noise.new_tensor([0.,1.]),
                           method='dopri5',atol=1e-6,rtol=1e-3)
            return result[-1],[]
        sampler.integrate_condition=integrate
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    record={'method':own.reference_method,'solver':own.reference_solver,
            'atol':1e-6 if own.reference_solver=='dopri5' else None,
            'rtol':1e-3 if own.reference_solver=='dopri5' else None,
            'sources':{},'complete':False}
    for source in [Path(__file__),Path(sampler.__file__),
                   Path(__file__).with_name('internal_guidance_path_extrapolation.py'),
                   Path(__file__).with_name('imagenet100_sit_multiscale_models.py')]:
        record['sources'][str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source,out/source.name)
    (out/'reference_manifest.json').write_text(json.dumps(record,indent=2))
    sampler.main(args)
    record['complete']=True
    path=out/'sampling_manifest.json';meta=json.loads(path.read_text())
    meta['base_integrator']=own.reference_solver
    if own.reference_solver=='dopri5':meta['num_steps']=None
    meta['scope']='Same-bank original PFR / stronger solver comparison'
    meta['reference_intervention']=record
    path.write_text(json.dumps(meta,indent=2))
    (out/'reference_manifest.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
