"""Fit one fixed affine shallow-to-deep condition adapter; no FID selection."""
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import sample_raev2_pfr_retiming as native
from experiments.raev2_condition_carrier import representations, readout, guided_outputs
from experiments.train_raev2_observable_potential import load_banks

OUT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_condition_carrier_20260908')
TIMES = [1., .9, .75, .55, .3, .1]


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bankdir = OUT.parent/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
    banks, record = load_banks(bankdir)
    ckhash = native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    cfg = native.load_config(native.DEFAULT_CONFIG)
    model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state = torch.load(native.DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['ema'], strict=True)
    del state
    paths = [Path(__file__), Path(inspect.getfile(representations)), Path(inspect.getfile(type(model))), native.DEFAULT_CONFIG]
    request = dict(checkpoint_sha256=ckhash, bank=record, times=TIMES, train_images=1024, validation_images=128,
                   tokens_per_train_image=8, ridge_relative=.001, seed=202609481,
                   sources={str(p):native.file_sha256(p) for p in paths},
                   scope='Frozen backbone; affine feature distillation only. Validation checks readability, not generation quality. No parameter selection.')
    (OUT/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    start = time.perf_counter()
    xs, ys = [], []
    selected = {}
    gen_tokens = torch.Generator().manual_seed(202609481)
    for split, count in [('train',1024), ('validation',128)]:
        data, meta = banks[split]
        idx = np.linspace(0, len(data)-1, count, dtype=int)
        assert len(np.unique(idx)) == count
        selected[split] = idx.tolist()
    (OUT/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')

    def batches(split):
        data, meta = banks[split]
        ids = selected[split]
        generator = torch.Generator(device='cuda').manual_seed(202609481+(split=='validation'))
        for offset in range(0, len(ids), 8):
            inds = ids[offset:offset+8]
            clean = torch.from_numpy(np.array(data[inds],copy=True)).float().cuda()
            t = torch.tensor([TIMES[i%len(TIMES)] for i in range(offset,offset+len(inds))],device='cuda')
            eps = torch.randn(clean.shape, generator=generator, device='cuda')
            z = (1-t[:,None,None,None])*clean+t[:,None,None,None]*eps
            labels = torch.as_tensor(np.array(meta['labels'][inds],copy=True),device='cuda').long()
            yield offset, z, t, labels

    for offset,z,t,labels in batches('train'):
        shallow, condition = representations(model,z,t,labels)
        if offset == 0:
            full, base = model(z,t,context=labels,attn_mask=None)
            assert torch.equal(full,readout(model,z,condition))
        positions = torch.stack([torch.randperm(256,generator=gen_tokens)[:8] for _ in range(len(z))]).cuda()
        row = torch.arange(len(z),device='cuda')[:,None]
        xs.append(shallow[row,positions].reshape(-1,shallow.shape[-1]).cpu().numpy())
        ys.append(condition[row,positions].reshape(-1,condition.shape[-1]).cpu().numpy())
        if (offset+8)%256 == 0:
            print(json.dumps(dict(train_images=offset+8,seconds=time.perf_counter()-start)),flush=True)
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    del xs,ys
    xm,ym = x.mean(0),y.mean(0)
    x -= xm
    y -= ym
    gram = x.T@x/len(x)
    cross = x.T@y/len(x)
    ridge = .001*np.trace(gram)/len(gram)
    weight = np.linalg.solve(gram+ridge*np.eye(len(gram)),cross)
    normal_error = np.linalg.norm((gram+ridge*np.eye(len(gram)))@weight-cross)/np.linalg.norm(cross)
    assert normal_error < 1e-10
    train_error = float(np.mean((x@weight-y)**2)/np.mean(y**2))
    cpu_adapter = dict(weight=torch.from_numpy(weight.astype(np.float32)),input_mean=torch.from_numpy(xm.astype(np.float32)),
                       output_mean=torch.from_numpy(ym.astype(np.float32)))
    torch.save(cpu_adapter,OUT/'adapter.pt')
    adapter = {k:v.cuda() for k,v in cpu_adapter.items()}
    del x,y,gram,cross
    rows = []
    for offset,z,t,labels in batches('validation'):
        shallow,condition = representations(model,z,t,labels)
        full,weak,output,carrier,estimated = guided_outputs(model,z,shallow,condition,adapter)
        native_full,native_base = model(z,t,context=labels,attn_mask=None)
        assert torch.equal(full,native_full)
        assert torch.equal(readout(model,z,condition+0*(condition-estimated)),full)
        for j in range(len(z)):
            f,w,o,c,b = [v[j].double().flatten() for v in [full,weak,output,carrier,native_base]]
            gap = f-w
            original_gap = f-b
            feature_error = (estimated[j]-condition[j]).double()
            feature_reference = (condition[j]-adapter['output_mean']).double()
            row = dict(index=selected['validation'][offset+j],time=float(t[j]),label=int(labels[j]),
                       condition_nmse=float(feature_error.square().mean()/feature_reference.square().mean()),
                       weak_over_full_norm=float(w.norm()/f.norm()),
                       adapted_gap_over_native_gap_norm=float(gap.norm()/original_gap.norm()),
                       adapted_gap_native_cosine=float(gap.dot(original_gap)/(gap.norm()*original_gap.norm())),
                       carrier_output_difference_over_output_guidance=float((c-o).norm()/(o-f).norm()),
                       carrier_over_full_norm=float(c.norm()/f.norm()))
            assert all(np.isfinite(v) for v in row.values())
            rows.append(row)
        print(json.dumps(dict(validation_images=offset+len(z),seconds=time.perf_counter()-start)),flush=True)
    result = dict(complete=True,rows=rows,train_centered_feature_nmse=train_error,normal_equation_relative_residual=float(normal_error),
                  ridge=float(ridge),adapter_parameters=sum(v.numel() for v in cpu_adapter.values()),
                  seconds=time.perf_counter()-start,adapter_sha256=native.file_sha256(OUT/'adapter.pt'),
                  request_sha256=native.file_sha256(OUT/'request.json'),native_validation_parity='128 full outputs bitwise; zero carrier gain bitwise',
                  quality_evaluated=False)
    (OUT/'fit_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    main()
