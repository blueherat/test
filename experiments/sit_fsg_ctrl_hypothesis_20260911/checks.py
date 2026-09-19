"""Limiting cases, frozen trajectory replay, and deterministic suffix checks."""
from __future__ import annotations
import time
import numpy as np
import torch
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core,pipeline as p,study
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha


def cpu_checks():
    # Nonzero k distinguishes memory feedback from instantaneous suppression.
    g=torch.tensor([-.3,-.1,.1,.3]).reshape(1,1,1,4)
    history=-g
    m,s=core.correction(g,history,'smc',.2,5.)
    instant,_=core.correction(g,history,'instant',.2,5.)
    assert not torch.equal(m,instant)
    assert torch.equal(s,(g-history)+5*history)
    soft,_=core.correction(g,history,'soft',.2,5.)
    assert (soft*g>=0).all() and (instant*g<0).any()
    normal,_=core.correction(g,history,'norm',.2,5.)
    torch.testing.assert_close(core.norm(normal),core.norm(instant))
    # A nonisotropic future map rotates the inverse response; a scalar CFG
    # gain cannot generally reproduce it even when displacement length agrees.
    matrix=torch.tensor([[2.,1.],[0.,.5]],dtype=torch.float64)
    wanted=torch.tensor([1.,1.],dtype=torch.float64)
    delta=torch.linalg.solve(matrix,wanted)
    assert torch.linalg.vector_norm(matrix@delta-wanted)<1e-12
    aligned=wanted/wanted.norm()*delta.norm()
    assert torch.linalg.vector_norm(matrix@aligned-wanted)>.1
    return dict(memory_disambiguated=True,soft_no_flip=True,norm_control_equal=True,
                nonisotropic_inverse_response=True)


@torch.inference_mode()
def run():
    p.ROOT.mkdir(parents=True,exist_ok=True)
    loaded_sources={str(x.resolve()):sha(x) for x in p.sources()}
    cpu=cpu_checks();parent=read(p.SOURCE/'request.json')
    rt=core.old.make_runtime()
    noise=torch.from_numpy(np.load(p.SOURCE/'inputs/noise.npy')[:8].copy()).cuda()
    labels=torch.from_numpy(np.load(p.SOURCE/'inputs/labels.npy')[:8].copy()).cuda()
    limits=core.limiting_checks(rt,noise,labels);hooks=p.engine.parent.hook_counts(rt)
    golden=[];checks=[];before=time.perf_counter()
    for method,arm in p.REUSED.items():
        z,stats,states,_=core.trajectory(rt,noise,labels,method,capture=(8,24,40))
        path=p.SOURCE/arm/'rank0/batch0000.npz'
        with np.load(path) as old:
            error=float(np.abs(z.cpu().numpy()-old['latents']).max())
            np.testing.assert_array_equal(z.cpu().numpy(),old['latents'])
            assert stats['full_calls']==int(old['full_calls'])
            assert stats['prefix_calls']==int(old['prefix_calls'])
        assert rt.labels is labels and p.engine.parent.hook_counts(rt)==hooks
        golden.append(dict(method=method,arm=arm,path=str(path),sha256=sha(path),exact=True,max_abs=error))
        print(dict(golden_exact=method,full=stats['full_calls'],prefix=stats['prefix_calls']),flush=True)
        if method=='cfg_tuned':
            resumed,_,_,_=core.trajectory(rt,noise,labels,method,start=24,state=states[24]['state'])
            assert torch.equal(resumed,z)
            suffix=core.future(rt,states[24]['state'],24,labels,'guided',amount=1.25)
            assert torch.equal(suffix,z)
            null_path=core.future(rt,states[24]['state'],24,labels,'null',return_path=True)
            null_later=core.future(rt,null_path[16],40,labels,'null')
            assert torch.equal(null_later,null_path[-1])
            shared=states[24]['state'][:2]
    for name in ('fsg_length_high','fsg_debiased_high','cycle_high','instant_high','soft_high','norm_high','refined_high'):
        z,stats,_,_=core.trajectory(rt,noise,labels,name)
        assert torch.isfinite(z).all()
        checks.append(dict(method=name,full=stats['full_calls'],prefix=stats['prefix_calls']))
        print(checks[-1],flush=True)
    candidates,setup,records=core.proposals(rt,shared,24,labels[:2],1.25)
    with core.old.exact_matmul():
        gram=setup['q'].transpose(1,2)@setup['q']
    torch.testing.assert_close(gram,torch.eye(4,device=gram.device)[None].expand(2,-1,-1),rtol=1e-5,atol=1e-5)
    for key in candidates:
        assert torch.isfinite(candidates[key]).all()
        if key.endswith('_radius'):
            torch.testing.assert_close(core.norm(candidates[key]-shared),setup['radius'],rtol=2e-5,atol=2e-5)
    for record in records.values():
        assert (record['after']<=record['before']+1e-10).all()
    for method,extra in [('cfg_tuned',0),('fsg_high',12)]:
        _,stats,_,_=core.trajectory(rt,noise,labels,method,handoff=16)
        assert stats['full_calls']==160+extra
        assert stats['auxiliary_full_calls']==extra
    reader=study.Readouts(rt)
    readouts=reader.latents(z,labels)
    assert len(readouts)==6 and all(np.asarray(v).shape==(8,) for v in readouts.values())
    result=dict(passed=True,cpu=cpu,limits=limits,golden=golden,trajectories=checks,
        shared_basis_orthogonal=True,exact_radius_controls=True,objective_acceptance_monotone=True,
        guided_replay_exact=True,null_suffix_semigroup_exact=True,classifier_pixel_input_checked=True,
        full_calls_refinement_matches_fsg=True,handoff_cost_accounting_checked=True,source_hashes=loaded_sources,
        elapsed_seconds=time.perf_counter()-before,no_new_fid_used=True)
    assert next(row['full'] for row in checks if row['method']=='refined_high')==260
    for path,digest in loaded_sources.items():
        assert sha(path)==digest,'Source changed during preflight: '+path
    atomic(p.ROOT/'development_check.json',result)
    print(dict(development_check_passed=True,seconds=result['elapsed_seconds']),flush=True)
