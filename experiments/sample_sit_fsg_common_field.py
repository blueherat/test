"""Use the same guided field on both calibration legs; preserve local guidance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    import sample_imagenet100_sit_foresight_fixed_point as sampler
    import sample_sit_fsg_clock_ablation as clocks
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--clock-mode',choices=['short','asynchronous'],required=True)
    p.add_argument('--method',choices=['foresight'],required=True)
    own,_=p.parse_known_args()
    for name in ['build_cfg_fields','build_ig_fields']:
        original=getattr(sampler,name)
        def builder(*args,_original=original,**kwargs):
            fields=_original(*args,**kwargs)
            fields.reference=fields.foresight_forward
            fields.metadata['calibration_reference']='same guided field as forward leg'
            return fields
        setattr(sampler,name,builder)
    source=Path(__file__).resolve()
    record={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'intervention':'R=G on calibration; ordinary local guided field unchanged',
            'clock_mode':own.clock_mode,'complete':False}
    own.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with (own.output_dir.parent/(own.output_dir.name+'_common_field.json')).open('x') as f:
        json.dump(record,f,indent=2)
    clocks.main()
    shutil.copy2(source,own.output_dir/source.name)
    record['complete']=True
    (own.output_dir/'common_field_manifest.json').write_text(json.dumps(record,indent=2))
    p=own.output_dir/'sampling_manifest.json';meta=json.loads(p.read_text())
    meta['common_field_intervention']=record;p.write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
