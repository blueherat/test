"""Real-data risk of conditional information in token subsets; no rollout."""
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
OUT = Path('/home/zhoushunyu/data/eqvae/experiments/fsg_carrier_real_risk_20260908')
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

    from experiments.train_raev2_observable_potential import load_banks
    bankdir = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1')
    banks, bank_record = load_banks(bankdir)
    data, metadata = banks['validation']
    chosen_labels = np.arange(32)*999//31
    indices = [int(np.flatnonzero(metadata['labels']==c)[0]) for c in chosen_labels]
    gen = torch.Generator(device='cuda').manual_seed(202609441)
    rows = []
    started = time.perf_counter()
    for index, label in zip(indices, chosen_labels):
        clean = torch.from_numpy(np.array(data[index], copy=True)).float().unsqueeze(0).cuda()
        eps = torch.randn(clean.shape, generator=gen, device='cuda')
        y = torch.tensor([int(label)], device='cuda')
        yn = torch.tensor([1000], device='cuda')
        for event, time_value in enumerate([1., .9, .75, .55]):
            t = torch.tensor([time_value], device='cuda')
            z = (1-t[:,None,None,None])*clean+t[:,None,None,None]*eps
            fc, _ = model(z, t, context=y, attn_mask=None)
            fu, _ = model(z, t, context=yn, attn_mask=None)
            counts['full'] += 2
            hc, tb, mask = prefix(z, t, y)
            hu, _, _ = prefix(z, t, yn)
            assert torch.equal(fc, suffix(hc, z, tb, mask))
            assert torch.equal(fu, suffix(hu, z, tb, mask))
            def replace(sl):
                h = hu.clone()
                h[:,sl] = hc[:,sl]
                return suffix(h,z,tb,mask)
            preds = [fu,fc,replace(slice(n+4,n+12)),replace(slice(0,n)),replace(slice(n,n+4))]
            v = torch.stack([(p-clean).flatten().double() for p in preds])
            g = (v@v.T/v.shape[1]).cpu().numpy()
            np.savez_compressed(OUT/f'id{index:04d}_event{event}.npz', vectors=v.float().cpu().numpy())
            rows.append(dict(index=index,label=int(label),event=event,time=float(t[0]),
                             names=['null','conditional','class_only','image_only','time_only'],gram=g.tolist(),
                             noise_sha256=sha(eps.cpu().numpy()),clean_sha256=sha(clean.cpu().numpy()),
                             state_sha256=sha(z.cpu().numpy())))
        print(json.dumps(dict(done_label=int(label), seconds=time.perf_counter()-started)), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    assert counts == dict(full=256, prefix14=256, suffix14_and_decoder=640)
    source_hashes[str(Path(inspect.getfile(load_banks)))] = native.file_sha256(Path(inspect.getfile(load_banks)))
    result = dict(complete=True, rows=rows, counts=counts, seconds=elapsed, sources=source_hashes,
                  checkpoint_sha256=ckhash, bank=bank_record, seed=202609441,
                  depth=14, parity='256 native split outputs bitwise',
                  note='32 held-out real images x4 noisy times; true conditional Full only; no FID')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=elapsed)), flush=True)


if __name__ == '__main__':
    main()
