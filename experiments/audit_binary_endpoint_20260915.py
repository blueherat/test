"""CPU-only independent checks of the live binary endpoint GAN implementation."""
import copy
import hashlib
import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from experiments.adversarial_weak_training_20260915.binary_critic import (
    BinaryCritic, discriminator_loss, weak_loss,
)
from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import (
    ordinary_rollout, recomputed_rollout,
)
from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps

WORK = Path(__file__).resolve().parents[1]
RUN = Path('/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_v1')
OUT = WORK / 'docs/data/binary_endpoint_audit_20260915.json'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flat(values):
    return torch.cat([v.reshape(-1) for v in values])


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 8).double()

    def forward(self, context, condition):
        return self.linear(torch.cat((context, condition), -1))


class TinyAdapter:
    """Nonlinear frozen field, exercising production GuidanceSteps/field on CPU."""
    name = 'sit_small'

    def __init__(self):
        self.values, self.head_calls = {}, 0
        self.matrix = torch.randn(8, 8, dtype=torch.float64) / 20

    def full(self, z, t, labels, native=False):
        condition = torch.nn.functional.one_hot(labels % 4, 4).to(z)
        context = torch.tanh(z @ self.matrix)
        self.values = dict(context=context, condition=condition)
        return .1 * z + context + .03 * t, None

    def autocast(self):
        return nullcontext()

    def unpatch(self, x):
        return x

    def native_to_velocity(self, raw, z, t):
        return raw


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2026091598)
    q = read(RUN / 'request.json')
    source_checks = {p: sha(p) == h for p, h in q['sources'].items()}
    assert all(source_checks.values())
    assert sha(q['initial_head']) == q['initial_head_sha256']
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  run=str(RUN), request_sha256=sha(RUN/'request.json'),
                  source_checks=source_checks, initial_head_sha_verified=True,
                  initial_baseline_parity=read(RUN/'initial_baseline_parity.json'),
                  scope='New CPU algebra/gradient tests; no new real-model GPU backward test or image generation')
    data_root=RUN.parents[2]/'imagenet_sit_flow/imagenet100_cmc_sdvae'
    index_root=RUN.parents[1]/'guidance_dynamic_50k_20260915/sit_small/data'
    ids=np.load(index_root/'train_ids.npy'); labels=np.load(index_root/'train_labels.npy')
    source_labels=np.load(data_root/'train_labels.npy')
    assert np.array_equal(labels,source_labels[ids])
    train_ids=np.load(data_root/'train_source_indices.npy')
    val_ids=np.load(data_root/'validation_source_indices.npy')
    overlap,ti,vi=np.intersect1d(train_ids,val_ids,return_indices=True)
    tm=np.load(data_root/'train_moments.npy',mmap_mode='r')
    vm=np.load(data_root/'validation_moments.npy',mmap_mode='r')
    identical=sum(np.array_equal(tm[i],vm[j]) for i,j in zip(ti,vi))
    result['data_check']=dict(train_samples=len(ids),classes=len(np.unique(labels)),
        labels_match_source_indices=True,numeric_index_overlap_across_distinct_split_namespaces=len(overlap),
        overlapping_numeric_ids_identical_moments=int(identical),
        source_index_namespace='train and validation are independently indexed source splits; numeric intersection does not identify shared images')

    # Verify the real/fake and weak-head signs, including the minus sign in IG.
    real = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    fake = torch.tensor([-.3, .2], dtype=torch.float64, requires_grad=True)
    dr, df = torch.autograd.grad(discriminator_loss(real, fake), (real, fake))
    dw, = torch.autograd.grad(weak_loss(fake), fake)
    assert (dr < 0).all() and (df > 0).all() and (dw < 0).all()
    w = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    endpoint = .2 + 1.05 * (.2 - w)
    gradient, = torch.autograd.grad(weak_loss(endpoint[None]), w)
    after = .2 + 1.05 * (.2 - (w.detach() - .01 * gradient))
    assert gradient > 0 and after > endpoint
    result['loss_signs'] = dict(discriminator_real_up=True, discriminator_fake_down=True,
                                generator_fake_logit_up=True, extrapolation_minus_sign_correct=True)

    critic = BinaryCritic().double()
    checkpoints = sorted(RUN.glob('checkpoint_*.pt'))
    if checkpoints:
        # Pin the first completed checkpoint so a concurrent training save does
        # not change this numerical experiment between repeated invocations.
        path = checkpoints[0]
        state = torch.load(path, map_location='cpu', weights_only=False)
        critic.load_state_dict(state['critic'])
        initial = torch.load(q['initial_head'], map_location='cpu', weights_only=False)['ema']
        keys = [k for k in initial if k.endswith(('.weight', '.bias'))]
        base = flat([initial[k] for k in keys])
        change = {kind: float(flat([state[kind][k]-initial[k] for k in keys]).norm()/base.norm())
                  for kind in ('head','ema')}
        result['checkpoint'] = dict(path=str(path),sha256=sha(path),step=state['step'],
            optimizer_d_steps=sorted({int(v['step']) for v in state['optimizer_d']['state'].values()}),
            optimizer_w_steps=sorted({int(v['step']) for v in state['optimizer_w']['state'].values()}),
            head_relative_parameter_changes=change)
    critic.eval().requires_grad_(False)

    # Complete 64-step Heun, including the frozen suffix after step 32. The
    # production step/field and binary loss are used, with a small CPU backbone.
    adapter, head = TinyAdapter(), TinyHead()
    labels = torch.tensor([0, 17, 42, 99])
    z0 = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    projection = torch.randn(8, 2048, dtype=torch.float64) / 10
    steps = GuidanceSteps(adapter, head, labels, 1.05, method='real')
    parameters = tuple(head.parameters())

    def terminal(z):
        return weak_loss(critic(torch.sin(z @ projection), labels))

    ref = ordinary_rollout(steps, len(steps), z0)
    gr = torch.autograd.grad(terminal(ref), (z0, *parameters))
    actual = recomputed_rollout(steps, len(steps), z0, parameters)
    ga = torch.autograd.grad(terminal(actual), (z0, *parameters))
    rel = float((flat(ga)-flat(gr)).norm()/flat(gr).norm())
    assert torch.equal(ref, actual) and rel < 1e-10
    direction = [torch.randn_like(p) for p in parameters]
    norm = flat(direction).norm()
    direction = [d/norm for d in direction]
    analytic = float(sum((g*d).sum() for g,d in zip(ga[1:], direction)))
    original = [p.detach().clone() for p in parameters]
    finite_checks=[]
    # sampling.field intentionally casts the weak prediction to float32 even
    # for this double-precision surrogate. Very small finite differences see
    # rounding noise, while large steps can cross critic LeakyReLU boundaries.
    # Require agreement at two adjacent step sizes, not an arbitrarily fixed h.
    with torch.no_grad():
        for epsilon in (1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5):
            values=[]
            for sign in (1,-1):
                for p, b, d in zip(parameters,original,direction): p.copy_(b+sign*epsilon*d)
                values.append(float(terminal(ordinary_rollout(steps,len(steps),z0.detach()))))
            finite=(values[0]-values[1])/(2*epsilon)
            finite_checks.append(dict(epsilon=epsilon,finite_difference=finite,abs_error=abs(analytic-finite)))
        for p,b in zip(parameters,original):p.copy_(b)
    print('finite_difference_step_size_study',analytic,finite_checks,flush=True)
    tolerance=max(1e-7,2e-4*abs(analytic))
    agrees=[v['abs_error'] < tolerance for v in finite_checks]
    assert any(a and b for a,b in zip(agrees,agrees[1:])), (analytic,finite_checks)
    result['production_solver_cpu_test'] = dict(steps=64,coefficient=1.05,forward_exact=True,
        full_chain_gradient_relative_error=rel,directional_derivative=analytic,
        finite_difference_checks=finite_checks,
        finite_difference_tolerance=tolerance,
        two_adjacent_finite_difference_steps_agree=True,
        float32_prediction_cast_requires_finite_difference_step_size_check=True,
        uses_real_binary_critic_checkpoint=bool(checkpoints),backbone='nonlinear CPU surrogate')

    # Same scalar definitions as training, including feature-input R1. The
    # mean of four equal local gradients must equal the global-batch gradient.
    real_features=torch.randn(24,2048,dtype=torch.float64)
    fake_features=torch.randn(24,2048,dtype=torch.float64)
    y=torch.arange(24)%100
    frozen=copy.deepcopy(critic).train().requires_grad_(True)

    def d_gradient(ids):
        d=copy.deepcopy(frozen)
        x=torch.cat((real_features[ids],fake_features[ids])).detach().requires_grad_(True)
        m=len(ids); logits=d(x,y[ids].repeat(2))
        gx=torch.autograd.grad(logits[:m].sum(),x,create_graph=True)[0][:m]
        loss=discriminator_loss(logits[:m],logits[m:]) + .5*gx.square().sum(1).mean()
        grads=flat(torch.autograd.grad(loss,tuple(d.parameters())))
        return grads

    full=d_gradient(torch.arange(24))
    averaged=torch.stack([d_gradient(torch.arange(i*6,(i+1)*6)) for i in range(4)]).mean(0)
    error=float((averaged-full).norm()/full.norm())
    assert error < 1e-10
    result['four_rank_discriminator_mean_test']=dict(global_batch=24,local_batch=6,
        includes_r1=True,relative_gradient_error=error)

    # Explicitly test D/W isolation across an optimizer step on the same batch.
    generator=nn.Linear(8,2048).double()
    d=copy.deepcopy(frozen)
    g_output=generator(z0.detach())
    opt=torch.optim.Adam(d.parameters(),lr=1e-4)
    logits=d(torch.cat((real_features[:4],g_output.detach())),labels.repeat(2))
    discriminator_loss(logits[:4],logits[4:]).backward()
    assert all(p.grad is None for p in generator.parameters())
    opt.step();opt.zero_grad(set_to_none=True);d.eval().requires_grad_(False)
    weak_loss(d(g_output,labels)).backward()
    assert all(p.grad is None for p in d.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters())
    result['optimizer_isolation']=dict(d_step_does_not_update_weak=True,
        weak_step_does_not_update_discriminator=True,weak_input_gradient_preserved=True)

    rows=[json.loads(s) for s in (RUN/'train.jsonl').read_text().splitlines() if s.strip()]
    active=[r for r in rows if r['generator_updated']]
    norms=np.array([r['head_gradient_norm'] for r in active])
    finite_rows=all(np.isfinite(float(v)) for r in rows for v in r.values() if isinstance(v,(int,float)))
    assert finite_rows
    result['live_log_snapshot']=dict(step=rows[-1]['step'],weak_updates=len(active),
        nonfinite=False,gradient_clip_count=int((norms>1).sum()),
        gradient_min=float(norms.min()),gradient_median=float(np.median(norms)),gradient_max=float(norms.max()),
        last20={k:float(np.mean([r[k] for r in rows[-20:]])) for k in
                ('d_ce','g_loss','r1','real_accuracy','fake_accuracy','endpoint_rms','seconds')},
        progress=read(RUN/'progress.json'),training_status=read(RUN/'training/status.json'))
    latest=RUN/'latest.json'
    if latest.exists():result['replica_receipt']=read(latest)
    prior=read(WORK/'docs/data/adversarial_weak_audit_20260915/cpu.json')
    result['past_real_model_gradient_audit_dependencies_unchanged']={
        p:sha(p)==h for p,h in prior['dependency_hashes_at_audit'].items()}
    result['current_sources_still_match']=all(sha(p)==h for p,h in q['sources'].items())
    assert result['current_sources_still_match']
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_checks','replica_receipt')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
