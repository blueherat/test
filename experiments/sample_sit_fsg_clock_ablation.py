"""Controlled Euler state/query clocks; not a reproduction of SDXL DDIM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from foresight_fixed_point_flow import foresight_round_trip


def clock_operator(mode, num_steps):
    if mode not in {'long', 'short', 'asynchronous'}:
        raise ValueError(mode)
    if mode == 'long':
        return foresight_round_trip

    def update(state, *, time_value, future_time, forward_field, inverse_field,
               relaxation=1.):
        if not bool(future_time > time_value) or not 0 < relaxation <= 1:
            raise ValueError('invalid horizon or relaxation')
        h = time_value.new_tensor(1. / num_steps)
        query_time = time_value + h if mode == 'short' else future_time
        future = state + h * forward_field(time_value, state)
        result = future - h * inverse_field(query_time, future)
        return result if relaxation == 1. else state + relaxation * (result - state)

    return update


def main():
    import sample_imagenet100_sit_foresight_fixed_point as sampler
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--clock-mode', choices=['long', 'short', 'asynchronous'], required=True)
    own, remaining = parser.parse_known_args()
    args = sampler.build_parser().parse_args(remaining)
    if args.method not in {'closed', 'foresight'}:
        raise ValueError('This controlled study supports closed/foresight only')
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'clock_mode': own.clock_mode, 'num_steps': args.num_steps,
                  'operator': 'z1=z+h*G(z,t); z2=z1-h*W(z1,t+H)',
                  'state_h': 'event_horizon' if own.clock_mode == 'long' else 1./args.num_steps,
                  'query_H': 1./args.num_steps if own.clock_mode == 'short' else 'event_horizon',
                  'scope': 'Euler clock intervention, not SDXL DDIM replication',
                  'sources': {}, 'complete': False}
    for source in [Path(__file__), Path(sampler.__file__),
                   Path(__file__).with_name('foresight_fixed_point_flow.py')]:
        provenance['sources'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, out / source.name)
    (out/'clock_manifest.json').write_text(json.dumps(provenance, indent=2))
    sampler.foresight_round_trip = clock_operator(own.clock_mode, args.num_steps)
    sampler.main(args)
    manifest_path = out/'sampling_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['scope'] = provenance['scope']
    manifest['clock_intervention'] = provenance
    if args.method == 'foresight':
        manifest['foresight_operator'] = provenance['operator']
    manifest_path.write_text(json.dumps(manifest, indent=2))
    provenance['complete'] = True
    (out/'clock_manifest.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
