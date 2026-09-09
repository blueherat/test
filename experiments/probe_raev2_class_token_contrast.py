"""Follow-up: donor conditional-minus-null token transport; no sampling."""
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from experiments import sample_raev2_pfr_retiming as native

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/home/zhoushunyu/data/eqvae/experiments/fsg_class_token_contrast_20260908')
CACHE = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/collect')
IDS = [0, 142, 285, 428, 570, 713, 856, 999]
STEPS = [0, 47, 73, 87]


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = native.load_config(native.DEFAULT_CONFIG)
    model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state = torch.load(native.DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['ema'], strict=True)
    del state
    assert model.num_enc_blocks == 28 and model.num_cond_tokens == 12
    n = model.s_embedder.num_patches
    assert n == 256 and cfg.conditioning.arch.num_c_tokens == 8
    sources = [Path(__file__), native.DEFAULT_CONFIG, Path(native.__file__), Path(inspect.getfile(type(model)))]
    source_hashes = {str(p): native.file_sha256(p) for p in sources}
    ckhash = native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    counts = dict(full=0, prefix14=0, suffix14_and_decoder=0)

    def prefix(z, t, y):
        kw = dict(context=y, attn_mask=None)
        h, tb = model._build_sequence(z, t, kw)
        mask = model._build_attn_mask(h, kw)
        assert torch.count_nonzero(mask) == 0
        for block in model.blocks[:14]:
            h = block(h, model.enc_rope, mask)
        counts['prefix14'] += 1
        return h, tb, mask

    def suffix(h, z, tb, mask):
        for block in model.blocks[14:model.num_enc_blocks]:
            h = block(h, model.enc_rope, mask)
        h = model.s_projector(F.silu(tb + h[:, :n]))
        x = model.x_embedder(z)
        for block in model.blocks[model.num_enc_blocks:]:
            x = block(x, h, model.dec_rope)
        counts['suffix14_and_decoder'] += 1
        return model.unpatchify(model.final_layer(x, h), model.x_patch_size)

    banks = {i: np.load(CACHE / f'id{i:04d}/states.npy', mmap_mode='r') for i in IDS}
    manifests = {i: json.loads((CACHE / f'id{i:04d}/summary.json').read_text()) for i in IDS}
    grid = native.shifted_time_grid(100, 8., torch.device('cuda'))
    names = ['conditional_gap', 'class_only', 'image_only', 'time_only',
             'other_class_gap', 'other_class_tokens', 'same_class_other_state_tokens',
             'same_class_other_state_token_contrast', 'other_class_token_contrast']
    rows = []
    started = time.perf_counter()
    for j, i in enumerate(IDS):
        donor = IDS[(j+1) % len(IDS)]
        for step in STEPS:
            for index in [i, donor]:
                assert sha(banks[index][step]) == manifests[index]['state_sha256_by_step'][step]
            z = torch.from_numpy(np.array(banks[i][step], copy=True)).reshape(1, 1024, 16, 16).cuda()
            zd = torch.from_numpy(np.array(banks[donor][step], copy=True)).reshape_as(z).cuda()
            t = grid[step].expand(1).float()
            ys = [torch.tensor([v], device='cuda') for v in [i, 1000, (i+1) % 1000]]
            native_outputs = []
            sequences = []
            for y in ys:
                f, _ = model(z, t, context=y, attn_mask=None)
                counts['full'] += 1
                h, tb, mask = prefix(z, t, y)
                split = suffix(h, z, tb, mask)
                assert torch.equal(f, split), (i, step, 'native split parity')
                native_outputs.append(f)
                sequences.append(h)
            hc, hu, ho = sequences
            fc, fu, fo = native_outputs
            hd, _, _ = prefix(zd, t, ys[0])
            hdu, _, _ = prefix(zd, t, ys[1])

            def swap(donor_seq, sl):
                h = hu.clone()
                h[:, sl] = donor_seq[:, sl]
                return suffix(h, z, tb, mask)

            vectors = [fc-fu, swap(hc, slice(n+4, n+12))-fu,
                       swap(hc, slice(0, n))-fu, swap(hc, slice(n, n+4))-fu,
                       fo-fu, swap(ho, slice(n+4, n+12))-fu,
                       swap(hd, slice(n+4, n+12))-fu,
                       swap(hu + (hd-hdu), slice(n+4, n+12))-fu,
                       swap(hu + (ho-hu), slice(n+4, n+12))-fu]
            v = torch.stack([x.flatten().double() for x in vectors])
            gram = (v @ v.T / v.shape[1]).cpu().numpy()
            # Save actual output differences for a separate CPU recomputation.
            np.savez_compressed(OUT / f'id{i:04d}_step{step:03d}.npz', vectors=v.float().cpu().numpy())
            row = dict(id=i, donor_id=donor, other_class=(i+1)%1000, step=step, time=float(t[0]),
                       state_sha256=sha(banks[i][step]), donor_state_sha256=sha(banks[donor][step]),
                       names=names, gram=gram.tolist())
            rows.append(row)
        print(json.dumps(dict(done_id=i, seconds=time.perf_counter()-started)), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    assert counts == dict(full=96, prefix14=160, suffix14_and_decoder=320)
    result = dict(complete=True, rows=rows, counts=counts, compute_and_save_seconds=elapsed,
                  sources=source_hashes, checkpoint_sha256=ckhash, split_parity='bitwise all 96',
                  dtype='fp32 no TF32', depth=14, ids=IDS, steps=STEPS,
                  cache_note='existing native IG trajectories; no new rollout; diagnostic uses Full only')
    (OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, counts=counts, seconds=elapsed)), flush=True)


if __name__ == '__main__':
    main()
