"""Check the pretrained weak-prefix entry and gradient isolation before training."""
from types import SimpleNamespace
import torch
from . import catalog,models,train
from experiments import sample_imagenet100_sit_foresight_fixed_point as s
from experiments.imagenet100_sit_multiscale_models import evaluate_source_with_heads
from experiments.lifting_scale_sweep_20260909 import SMALL_CKPT,SMALL_HEAD,atomic,sha


def run():
    torch.set_num_threads(4)
    device=torch.device('cpu')
    module,source=s.load_official_sit_module(s.DEFAULT_OFFICIAL_SIT_REPO,verify_source=True)
    model,semantics,_,payload=s._load_field_model(checkpoint_path=SMALL_CKPT,
        requested_field='auto',weights='ema',sit_module=module,source_metadata=source,device=device)
    del payload
    spec=s.load_internal_head_for_source(checkpoint_path=SMALL_HEAD,name='depth4_v',
        head_weights='ema',model=model,sit_module=module,source_checkpoint_path=SMALL_CKPT,
        source_metadata=source,device=device)
    model.eval().requires_grad_(False)
    weak=models.WeakPrefix(SimpleNamespace(model=model,head=spec)).eval().requires_grad_(True)
    torch.manual_seed(2026120913)
    noise=torch.randn(2,4,32,32);times=torch.full((2,),.375);labels=torch.tensor([0,51])
    with torch.no_grad():
        reference=evaluate_source_with_heads(model,noise,times,labels,
            heads={'depth4_v':spec},source_semantics=semantics)[1]['depth4_v']
        assert torch.equal(reference,weak(noise,times,labels))
    optimizer=torch.optim.AdamW(weak.parameters(),lr=catalog.LR,weight_decay=0.)
    before=next(weak.blocks[0].parameters()).detach().clone()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=weak(noise,times,labels)
        loss=(output.float()-noise.sin()).square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in weak.parameters())
    torch.nn.utils.clip_grad_norm_(weak.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before,next(weak.blocks[0].parameters()))
    result=dict(passed=True,device='cpu',pretrained_weak_prefix_exact=True,
        strong_gradients_absent=True,all_weak_gradients_finite=True,weak_prefix_updated=True,
        loss=float(loss.detach()),trainable_parameters=sum(p.numel() for p in weak.parameters()),
        source_hashes={str(p):sha(p) for p in train.training_sources()},
        strong_sha256=sha(SMALL_CKPT),initial_head_sha256=sha(SMALL_HEAD),no_generated_images=True)
    atomic(catalog.ROOT/'cpu_model_check.json',result)
    print(result,flush=True)


if __name__=='__main__':run()
