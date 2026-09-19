"""Repeat the frozen copy protocol using the independent released SiT-XL IG model.

Only its full conditional field is used. Native XL time runs in the opposite
direction; v_aligned(z,t) = -v_XL(z,1-t). The XL is a reference, NOT an oracle.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from experiments.fm_common_inverse_copy_20260913 import run as pilot
from experiments.lifting_scale_sweep_20260909 import XL_REPO, XL_CKPT

PARENT = Path('/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_xl')
MAPPING = pilot.BASE/'imagenet100_cmc/manifest.json'


class XLFields(pilot.Fields):
    def __init__(self, reference, device):
        from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
        from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO
        from experiments.run_internal_guidance_sit_audit import load_model
        assert reference == 'ref_xl'
        module, source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
        self.models, self.metadata, self.calls = {}, {}, 0
        for key in ('strong', 'weak'):
            model, sem, meta = load_sit_field_model(checkpoint_path=pilot.CHECKPOINTS[key],
                weights='ema', sit_module=module, source_metadata=source, device=torch.device(device))
            self.models[key] = model, sem
            self.metadata[key] = meta
        model, metadata = load_model(repo=XL_REPO, checkpoint_path=XL_CKPT,
            model_name='SiT-XL/2', encoder_depth=8, state_key='ema', device=torch.device(device))
        self.models['ref_xl'] = model, None
        self.metadata['ref_xl'] = dict(**metadata, checkpoint_sha256=pilot.sha(XL_CKPT),
            aligned_field='-native_full(z, 1-t, imagenet1000_label)',
            reference_guidance=0, auxiliary_ig_head_used=False,
            caveat='separate architecture/training; learned reference, not true data inverse')
        entries = json.loads(MAPPING.read_text())['classes']
        assert sorted(e['label'] for e in entries) == list(range(100))
        mapping = [next(e['original_imagenet_label'] for e in entries if e['label'] == i) for i in range(100)]
        assert len(set(mapping)) == 100
        self.label_map = torch.tensor(mapping, device=device, dtype=torch.long)

    def call(self, key, z, t, labels):
        if key != 'ref_xl':
            return super().call(key, z, t, labels)
        self.calls += 1
        full, _, _ = self.models[key][0](z.float(), z.new_full((len(z),), 1-t), self.label_map[labels])
        return -full.float()


def prepare():
    ROOT.mkdir(exist_ok=False)
    shutil.copyfile(PARENT/'inputs.npz', ROOT/'inputs.npz')
    request = json.loads((PARENT/'request.json').read_text())
    request.update(references=['ref_xl'],
        reference_caveats={'ref_xl': 'released independent SiT-XL/2 IG main field; not oracle; full ImageNet conditioning'},
        xl_time_conversion='v_aligned(z,t) = -v_native(z,1-t)',
        label_mapping=str(MAPPING), label_mapping_sha256=pilot.sha(MAPPING),
        wrapper_sha256=pilot.sha(__file__), parent_request_sha256=pilot.sha(PARENT/'request.json'),
        checkpoints={k: str(pilot.CHECKPOINTS[k]) for k in ('strong', 'weak')} | {'ref_xl': str(XL_CKPT)},
        cost_note='XL inverse queries and small-model forward queries have different cost; total calls is not equal compute')
    pilot.atomic(ROOT/'request.json', request)
    print(json.dumps(request), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'run'])
    args = p.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        request = json.loads((ROOT/'request.json').read_text())
        assert request['wrapper_sha256'] == pilot.sha(__file__)
        assert request['label_mapping_sha256'] == pilot.sha(MAPPING)
        pilot.Fields = XLFields  # Process-local dependency injection; no shared source edits.
        pilot.run(argparse.Namespace(out=ROOT, reference='ref_xl', device='cuda:0'))


if __name__ == '__main__':
    main()
