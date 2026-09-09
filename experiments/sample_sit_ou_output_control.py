"""Fixed constant-gamma PFR on existing velocity/clean SiT source models."""
import argparse,hashlib,json
from pathlib import Path
import torch
from imagenet100_sit_multiscale_models import evaluate_internal_head_only
from internal_guidance_path_extrapolation import project_to_forward_ray,affine_counterfactual_ratio_velocity,split_internal_guidance

def main():
 import sample_imagenet100_sit_foresight_fixed_point as s
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--pfr-arm',choices=['ordinary','pfr','time_only','ou'],required=True);own,remaining=p.parse_known_args();args=s.build_parser().parse_args(remaining)
 assert args.family=='ig' and args.method=='closed' and args.foresight_schedule=='' and args.ig_gamma_segments is None and args.ag_gamma==.35 and args.precision=='fp32'
 extra={'prefix_parity':None,'parity_full_calls':0};builder=s.build_ig_fields
 if own.pfr_arm!='ordinary':
  def wrapped(model,labels,**kw):
   fields=builder(model,labels,**kw);heads=kw['heads'];name,spec=next(iter(heads.items()));assert len(heads)==1 and spec.depth==4
   fields.counters['pfr_prefix_calls']=0
   fields.counters['ou_full_calls']=0
   def guided(t,z):
    full,vals,_=s.evaluate_source_with_heads(model,z,t.expand(len(z)),labels[:len(z)],heads=heads,source_semantics=kw['strong_semantics']);fields.counters['guided_shared_backbone_forwards']+=1
    weak=vals[name];ordinary=full+.35*(full-weak)
    if float(t)>=.5:return ordinary
    h=min(1/32,.5-float(t));base,cal=split_internal_guidance(full,weak,gamma=.35)
    q=z if own.pfr_arm in ['time_only','ou'] else z+h*project_to_forward_ray(cal,ordinary).parallel;tf=(t+t.new_tensor(h)).expand(len(z))
    future=evaluate_internal_head_only(model,q,tf,labels[:len(z)],spec=spec);fields.counters['pfr_prefix_calls']+=1
    if args.num_samples==8 and extra['prefix_parity'] is None:
     _,check,_=s.evaluate_source_with_heads(model,q,tf,labels[:len(z)],heads=heads,source_semantics=kw['strong_semantics']);extra['parity_full_calls']+=1;extra['prefix_parity']=torch.equal(future,check[name]);assert extra['prefix_parity']
    if own.pfr_arm in ['time_only','ou']:
     revision=weak-future
     if float(t)<.25:
      from experiments.pfr_ou_semigroup_spectrum import transport_state_at_fixed_ou_coordinate,ou_degree1_retiming_velocity_defect
      from experiments.pfr_ou_semigroup_controls import split_raw_revision_against_ou_degree1
      from experiments.pfr_retiming_controls import rms_match_per_sample
      qou=transport_state_at_fixed_ou_coordinate(z,t,t+h)
      sf,_,_=s.evaluate_source_with_heads(model,qou,tf,labels[:len(z)],heads=heads,source_semantics=kw['strong_semantics']);fields.counters['ou_full_calls']+=1
      if own.pfr_arm=='ou':
       axis=ou_degree1_retiming_velocity_defect(full,sf,z,t,t+h)
       revision=rms_match_per_sample(split_raw_revision_against_ou_degree1(revision,axis).common,revision)
     return ordinary+1.35*revision
    return affine_counterfactual_ratio_velocity(full,base,(future,),(1.,),gamma=.35)
   fields.guided=guided;return fields
  s.build_ig_fields=wrapped
 out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
 (out/'wrapper_source.py').write_bytes(Path(__file__).read_bytes())
 s.main(args)
 record=dict(complete=True,arm=own.pfr_arm,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**extra)
 (out/'pfr_output_manifest.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
