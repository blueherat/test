"""Hold IG reference gamma at the calibration event's current time."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    import sample_imagenet100_sit_foresight_fixed_point as sampler
    import sample_sit_fsg_clock_ablation as clocks
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--family',choices=['ig'],required=True)
    p.add_argument('--method',choices=['foresight'],required=True)
    p.add_argument('--clock-mode',choices=['short','asynchronous'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    own,_=p.parse_known_args()
    event_time=[None]
    original_operator=clocks.clock_operator
    original_builder=sampler.build_ig_fields

    def operator(mode,num_steps):
        base=original_operator(mode,num_steps)
        def update(state,**kwargs):
            event_time[0]=float(kwargs['time_value'])
            return base(state,**kwargs)
        return update

    def builder(*args,**kwargs):
        assert kwargs['gamma_segments']==((.25,.6),(.5,.7),(1.,0.))
        fields=original_builder(*args,**kwargs)
        # These factories only create closures. Each reference call still makes
        # exactly one shared full/head backbone pass, with no extra model query.
        fixed={g:original_builder(*args,**dict(kwargs,gamma_segments=((1.,g),)))
               for g in [.6,.7,0.]}
        def reference(query_time,state):
            t=event_time[0]
            if t is None:raise RuntimeError('reference query outside calibration')
            g=.6 if t<.25 else (.7 if t<.5 else 0.)
            family=fixed[g]
            before=family.counters.copy()
            value=family.reference(query_time,state)
            for k,v in family.counters.items():fields.counters[k]+=v-before[k]
            return value
        fields.reference=reference
        fields.metadata['reference_gamma_clock']='current calibration event time, held for inverse query'
        return fields

    clocks.clock_operator=operator
    sampler.build_ig_fields=builder
    source=Path(__file__).resolve()
    record={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'reference':'S(z,q)-gamma(t_event)*(S(z,q)-W_depth(z,q))',
            'clock_mode':own.clock_mode,'complete':False}
    own.output_dir.parent.mkdir(parents=True,exist_ok=True)
    preflight=own.output_dir.parent/(own.output_dir.name+'_frozen_reference.json')
    with preflight.open('x') as f:json.dump(record,f,indent=2)
    clocks.main()
    shutil.copy2(source,own.output_dir/source.name)
    record['complete']=True
    (own.output_dir/'frozen_reference_manifest.json').write_text(json.dumps(record,indent=2))
    path=own.output_dir/'sampling_manifest.json'
    meta=json.loads(path.read_text());meta['frozen_reference_intervention']=record
    path.write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
