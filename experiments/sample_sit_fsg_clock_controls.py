"""Fixed controls separating extra current guidance and future-only queries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def control_operator(mode, num_steps):
    if mode not in {'gap', 'time_only'}:
        raise ValueError(mode)

    def update(state, *, time_value, future_time, forward_field, inverse_field,
               relaxation=1.):
        h = time_value.new_tensor(1. / num_steps)
        if not bool(future_time > time_value) or not 0 < relaxation <= 1:
            raise ValueError('invalid horizon or relaxation')
        forward = forward_field(time_value, state)
        query_time = time_value if mode == 'gap' else future_time
        reference = inverse_field(query_time, state)
        result = state + h * (forward - reference)
        return result if relaxation == 1. else state + relaxation * (result - state)

    return update


def main():
    import sample_imagenet100_sit_foresight_fixed_point as sampler
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--control-mode', choices=['gap', 'time_only'], required=True)
    own, remaining = p.parse_known_args()
    args = sampler.build_parser().parse_args(remaining)
    if args.method != 'foresight':
        raise ValueError('controls require foresight event schedule')
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'control_mode': own.control_mode, 'h': 1./args.num_steps,
                  'query_state': 'unmoved input state',
                  'query_time': 'current' if own.control_mode == 'gap' else 'event future time',
                  'operator': 'z+h*(G(z,t)-W(z,q))', 'sources': {}, 'complete': False}
    for source in [Path(__file__), Path(sampler.__file__),
                   Path(__file__).with_name('foresight_fixed_point_flow.py')]:
        provenance['sources'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, out/source.name)
    (out/'control_manifest.json').write_text(json.dumps(provenance, indent=2))
    sampler.foresight_round_trip = control_operator(own.control_mode, args.num_steps)
    sampler.main(args)
    provenance['complete'] = True
    manifest_path = out/'sampling_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(scope='Fixed current-guidance and time-only mechanism controls',
                    foresight_operator=provenance['operator'], control_intervention=provenance)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out/'control_manifest.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
