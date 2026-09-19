"""CPU validation before launching; real-model validation is the first queued job."""
import argparse
import ast
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch

from . import config as k
from . import components as x
from . import planning
from .training import State


def recovery_check():
    """Compare continuous and restored optimization, including EMA and data RNG."""
    saved_root=k.ROOT
    with tempfile.TemporaryDirectory(prefix='guidance50k_recovery_') as directory:
        k.ROOT=Path(directory)
        k.atomic(k.ROOT/'request.json',dict(test_only=True))
        try:
            with patch('torch.cuda.synchronize',lambda:None), \
                 patch('torch.cuda.get_rng_state',lambda:torch.zeros(1,dtype=torch.uint8)), \
                 patch('torch.cuda.set_rng_state',lambda value:None):
                torch.manual_seed(19)
                model=torch.nn.Linear(3,2)
                ema=copy.deepcopy(model).requires_grad_(False)
                opt=x.optimizer(model)
                g=torch.Generator().manual_seed(29)
                root=k.ROOT/'state'
                state=State(root,dict(head=model,ema=ema),dict(head=opt),g,{})
                def advance(state,until):
                    model,ema=state.modules['head'],state.modules['ema']
                    opt=state.optimizers['head']
                    for step in range(state.step+1,until+1):
                        batch=torch.randn((7,3),generator=state.generator)
                        desired=torch.randn((7,2),generator=state.generator)
                        loss=(model(batch)-desired).square().mean()
                        opt.zero_grad(set_to_none=True);loss.backward();opt.step();x.ema_update(ema,model)
                        state.step=step
                advance(state,3);state.checkpoint()
                # An uncommitted inactive recovery slot must not replace the valid one.
                inactive=1-k.read(root/'latest_pointer.json')['slot']
                (root/f'recovery_{inactive}.pt').write_bytes(b'interrupted uncommitted write')
                advance(state,7)
                expected={key:{n:v.clone() for n,v in module.state_dict().items()} for key,module in state.modules.items()}
                next_expected=torch.randn(11,generator=state.generator)
                other=torch.nn.Linear(3,2)
                other_ema=copy.deepcopy(other).requires_grad_(False)
                restored=State(root,dict(head=other,ema=other_ema),dict(head=x.optimizer(other)),torch.Generator().manual_seed(999),{})
                assert restored.step==3
                advance(restored,7)
                for key,module in restored.modules.items():
                    for name,value in module.state_dict().items():
                        assert torch.equal(value,expected[key][name]),(key,name)
                assert torch.equal(torch.randn(11,generator=restored.generator),next_expected)
        finally:k.ROOT=saved_root


def main(freeze):
    assert not torch.cuda.is_initialized()
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    plan=planning.jobs()
    seen=set()
    for job in plan:
        assert job['id'] not in seen
        assert set(job['depends'])<=seen,job
        seen.add(job['id'])
    assert {job['arm'] for job in plan if job['action']=='train'}==set(k.ARMS)
    assert {job['arm'] for job in plan if job['action']=='sample'}==set(k.SCREEN_ARMS)
    assert len([job for job in plan if job['action']=='nuisance'])==8
    assert k.STEPS==50000
    bank=x.SourceBank('strong',device='cpu')
    g=torch.Generator().manual_seed(47)
    for fold in (0,1):
        batch=bank.draw(g,n=100,train_excluding_fold=fold)
        assert torch.all(batch['pos_id']<18) and torch.all(batch['neg_id']<18)
        assert torch.all(batch['fold']!=fold)
    heldout=bank.draw(g,n=100,validation=True)
    assert torch.all(heldout['pos_id']>=18) and torch.all(heldout['neg_id']>=18)
    bank.negative=torch.cat((bank.negative,bank.negative),1)
    mixed=bank.draw(g,n=100,train_excluding_fold=0)
    assert torch.all(mixed['neg_id']%20<18) and torch.all(mixed['fold']==1)

    template=x.local.Head(4,8,2)
    eta=x.SourceHead(template)
    probabilities=torch.sigmoid(eta(torch.randn(4,4,4),torch.randn(4,4),torch.linspace(.1,.9,4)))
    assert torch.equal(probabilities,torch.full((4,),.5))
    for kind in ('fm','residual','guided_weak','contrast','contrast_weak','covariance','null'):
        pred=torch.randn(4,4,8,requires_grad=True)
        target=torch.randn(4,4,8,requires_grad=True)
        baseline=torch.randn(4,4,8,requires_grad=True)
        probability=torch.tensor([.1,.3,.6,.9],requires_grad=True)
        loss=x.objective(pred,target,torch.tensor([0.,1.,0.,1.]),probability,baseline,kind)
        loss.backward()
        assert torch.isfinite(pred.grad).all()
        assert probability.grad is baseline.grad is target.grad is None
    recovery_check()
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,syntax=True,all_candidates_have_executable_jobs=True,jobs=len(plan),
        trained_main_heads=len(k.ARMS),screen_arms=len(k.SCREEN_ARMS),nuisance_folds=8,steps=k.STEPS,
        clean_endpoint_folds_disjoint=True,validation_clean_endpoints_excluded=True,
        source_probabilities_initialize_half=True,all_loss_gradients_finite=True,
        only_prediction_receives_gradients=True,optimizer_ema_rng_recovery_bitwise=True,
        uncommitted_checkpoint_slot_ignored=True,
        cuda_initialized=False,gpu_checks_pending=True)
    if freeze:
        request=k.prepare()
        assert not torch.cuda.is_initialized()
        result['request_sha256']=k.sha(k.ROOT/'request.json')
        result['tau']=request['tau']
        k.atomic(k.ROOT/'cpu_preflight.json',result)
        k.atomic(k.ROOT/'plan.json',plan)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    main(parser.parse_args().freeze)
