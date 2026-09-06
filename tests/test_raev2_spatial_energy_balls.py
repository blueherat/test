from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import sample_raev2_spatial_energy_balls as sampler


def test_spatial_projectors_are_orthogonal_complete_and_idempotent():
    g = torch.Generator().manual_seed(19)
    x = torch.randn((5, 3, 4, 4), generator=g, dtype=torch.float64)
    dc, ac = sampler.split_spatial(x)
    assert torch.allclose(dc+ac, x, atol=2e-16, rtol=1e-15)
    assert abs(float((dc*ac).sum())) < 1e-14
    assert torch.allclose(sampler.split_spatial(dc)[0], dc, atol=1e-16, rtol=1e-15)
    assert torch.allclose(ac.mean((-2,-1)), torch.zeros((5,3), dtype=x.dtype), atol=1e-16)
    energies = sampler.cohort_energies(x, microbatch=2)
    assert sum(energies) == pytest.approx(float(x.square().mean()), abs=1e-15)


def test_white_noise_rank_units_are_one_over_256_and_255_over_256():
    # Orthogonal deterministic ensemble with identity second moment, no Monte Carlo tolerance.
    ensemble = torch.eye(256, dtype=torch.float64).reshape(256,1,16,16)*16
    assert sampler.cohort_energies(ensemble, microbatch=7) == [1/256, 255/256]
    assert sampler.bridge_budgets(1., [.3,.8]) == [1/256,255/256]
    assert sampler.bridge_budgets(0., [.3,.8]) == [.3,.8]
    t=.37
    actual=sampler.bridge_budgets(t,[.3,.8])
    assert actual==pytest.approx([(1-t)**2*.3+t*t/256,(1-t)**2*.8+t*t*255/256])


def test_spatial_ball_detects_excess_hidden_by_global_total_cancellation():
    original=torch.tensor([[[[2.,0.]]]],dtype=torch.float32)
    global_state,spatial_state=original.clone(),original.clone()
    # DC=1 exceeds .25, AC=1 lies below2, total2 remains below2.25.
    global_record=sampler.project_cohort(global_state,0.,[.25,2.],mode='global_ball')
    spatial_record=sampler.project_cohort(spatial_state,0.,[.25,2.],mode='spatial_balls')
    assert torch.equal(global_state,original)
    assert global_record['lambda_fp64']==[1.,1.]
    assert spatial_record['lambda_fp64']==[.5,1.]
    assert torch.equal(spatial_state,torch.tensor([[[[1.5,-.5]]]]))
    assert spatial_record['energy_after_dc_ac']==[.25,1.]


def test_coefficients_are_shared_across_whole_cohort_and_all_channels():
    x=torch.zeros((16,2,1,2),dtype=torch.float32)
    x[:8,0]=2.;x[:8,1]=4.
    original=x.clone()
    result=sampler.project_cohort(x,0.,[1.,0.],mode='spatial_balls',microbatch=8)
    # Cohort DC mean=(4+16)/4=5, so one lambda=sqrt(1/5), not separate B8/channel fits.
    assert result['energy_before_dc_ac']==[5.,0.]
    assert result['lambda_fp64']==pytest.approx([1/np.sqrt(5),1])
    assert torch.equal(x,original*float(np.float32(1/np.sqrt(5))))
    assert result['energy_after_dc_ac'][0]==pytest.approx(1.,abs=1e-7)
    separate=original.clone()
    for start in (0,8):
        sampler.project_cohort(separate[start:start+8],0.,[1.,0.],mode='spatial_balls')
    assert not torch.equal(separate,x)


def test_zero_budgets_zero_components_and_exact_identity_bypass():
    assert sampler.contraction(0.,0.)==1.
    assert sampler.contraction(1.,0.)==0.
    z=torch.zeros((8,2,2,2))
    result=sampler.project_cohort(z,0.,[0.,0.],mode='spatial_balls')
    assert result['lambda_fp64']==[1.,1.] and result['exact_all_one_bypass']
    y=torch.full_like(z,3.)
    result=sampler.project_cohort(y,0.,[0.,0.],mode='spatial_balls')
    assert result['lambda_fp64']==[0.,1.] and torch.count_nonzero(y)==0
    for pair in ((-1.,1.),(1.,-1.),(float('nan'),1.)):
        with pytest.raises(ValueError):sampler.contraction(*pair)
    with pytest.raises(FloatingPointError):sampler.cohort_energies(torch.full_like(z,float('inf')))


def test_applied_fp32_all_one_bypasses_recomposition_even_if_ideal_active():
    x=torch.tensor([[[[2.35,-.45,1.234567,.78]]]],dtype=torch.float32)
    original=x.clone();before=sampler.cohort_energies(x)
    budgets=[a*(1-1e-9)**2 for a in before]
    result=sampler.project_cohort(x,0.,budgets,mode='spatial_balls')
    assert result['ideal_active'] and not result['actual_applied_active']
    assert result['lambda_applied_fp32']==[1.,1.] and result['exact_all_one_bypass']
    assert torch.equal(x,original)
    assert result['float_state_residual_rms']>0


def test_projection_reports_actual_fp32_rounding_and_uses_next_time():
    x=torch.tensor([[[[1.234567,-.731,2.15,-1.32]]]],dtype=torch.float32)
    before=x.clone();following=.2;moments=[.1,.3]
    report=sampler.project_cohort(x,following,moments,mode='spatial_balls')
    assert report['bridge_budget_dc_ac']==sampler.bridge_budgets(following,moments,spatial_size=4)
    d,a=sampler.split_spatial(before.double())
    ideal=report['lambda_fp64'][0]*d+report['lambda_fp64'][1]*a
    err=x.double()-ideal
    assert report['float_state_residual_rms']==pytest.approx(float(err.square().mean().sqrt()),abs=1e-16)
    assert report['energy_after_dc_ac']==sampler.cohort_energies(x)
    for actual,budget in zip(report['energy_after_dc_ac'],report['bridge_budget_dc_ac']):
        assert actual<=budget+2e-7


def test_official_native_bf16_mixing_then_fp32_euler_and_floor(monkeypatch):
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:nullcontext())
    class Model:
        def __call__(self,z,t,**kwargs):
            assert kwargs['context'].tolist()==[0,1]
            return torch.full_like(z,1.03125,dtype=torch.bfloat16),torch.full_like(z,.3125,dtype=torch.bfloat16)
    z=torch.tensor([[[[.75,-1.5]]],[[[.3,-.9]]]],dtype=torch.float32);labels=torch.arange(2)
    full,base=Model()(z,None,context=labels)
    for current,following in ((.5,.4),(.074,0.),(.03,.02)):
        native=(base+1.78*(full-base)).float() if current>=.1 else full.float()
        expected=z-(current-following)*((z-native)/max(current,.05))
        assert torch.equal(sampler.successor(Model(),z,labels,current,following),expected)
    wrong=base.float()+1.78*(full.float()-base.float())
    assert not torch.equal((base+1.78*(full-base)).float(),wrong)


def test_noise_one_full_cuda_draw_and_cpu_pairing_without_cuda(monkeypatch):
    noise,rng=sampler.paired_noise(sampler.SEED,torch.device('cpu'),count=16,latent_shape=(2,1,2))
    again,rng2=sampler.paired_noise(sampler.SEED,torch.device('cpu'),count=16,latent_shape=(2,1,2))
    assert torch.equal(noise,again) and torch.equal(rng,rng2)
    calls=[]
    class Generator:
        def __init__(self,*,device):calls.append(('device',str(device)))
        def manual_seed(self,seed):calls.append(('seed',seed));return self
        def get_state(self):return torch.arange(4,dtype=torch.uint8)
    def randn(shape,*,generator,device,dtype):
        calls.append(('draw',shape,str(device),dtype));return torch.empty(0)
    monkeypatch.setattr(torch,'Generator',Generator);monkeypatch.setattr(torch,'randn',randn)
    sampler.paired_noise(sampler.SEED,torch.device('cuda:0'),count=1000)
    assert calls==[('device','cuda:0'),('seed',sampler.SEED),
                   ('draw',(1000,1024,16,16),'cuda:0',torch.float32)]


def test_time_major_official_matches_batch_major_without_energy_work(monkeypatch):
    def clean(model,state,times,labels):return state*.125+labels.reshape(-1,1,1,1).float()/31
    monkeypatch.setattr(sampler,'clean_forward',clean)
    monkeypatch.setattr(sampler.production,'clean_forward',clean)
    monkeypatch.setattr(sampler,'cohort_energies',lambda *a,**k:pytest.fail('official must not calculate energies'))
    noise,_=sampler.paired_noise(7,torch.device('cpu'),count=16,latent_shape=(2,2,2))
    labels=torch.arange(16);grid=sampler.shifted_time_grid(100,8,torch.device('cpu')).tolist()
    time_state,records,_=sampler.time_major(None,noise.clone(),labels,grid,mode='official',clean_moments=None)
    direct=torch.cat([x.cpu() for _,x,_ in sampler.batch_major(None,noise,labels,grid,torch.device('cpu'))])
    assert torch.equal(time_state,direct)
    assert all(x['projection'] is None and x['projection_and_diagnostics_wall_seconds']==0 for x in records)
    untouched=noise.clone();assert sampler.project_cohort(untouched,0.,None,mode='official') is None
    assert torch.equal(untouched,noise)


def test_time_major_completes_all_successors_before_one_cohort_control(monkeypatch):
    calls=[]
    def step(model,state,labels,current,following):
        calls.append(('model',labels.tolist()));return labels[:,None,None,None].float().expand_as(state)
    def project(state,following,moments,*,mode):
        calls.append(('projection',len(state)))
        assert torch.equal(state[:,0,0,0],torch.arange(16,dtype=torch.float32))
        return {'cohort_count':16}
    monkeypatch.setattr(sampler,'successor',step);monkeypatch.setattr(sampler,'project_cohort',project)
    sampler.time_major(None,torch.zeros(16,1,1,2),torch.arange(16),[1.,0.],mode='spatial_balls',clean_moments=[1.,1.])
    assert calls==[('model',list(range(8))),('model',list(range(8,16))),('projection',16)]


def test_fixed_cli_allows_only_official_cost_steps(tmp_path):
    common=['--output-dir',str(tmp_path),'--seed',str(sampler.SEED)]
    assert sampler.parse_args([*common,'--mode','parity']).num_samples==16
    baseline=sampler.parse_args([*common,'--mode','official','--parity-dir','parity','--num-steps','154'])
    assert baseline.num_samples==1000 and baseline.num_steps==154
    for mode in ('global_ball','spatial_balls'):
        assert sampler.parse_args([*common,'--mode',mode,'--parity-dir','parity']).num_steps==100
        with pytest.raises(SystemExit):sampler.parse_args([*common,'--mode',mode,'--parity-dir','parity','--num-steps','154'])
    for extra in (['--mode','parity','--num-samples','1000'],['--mode','official'],
                  ['--mode','parity','--seed','999'],['--mode','parity','--num-steps','99']):
        with pytest.raises(SystemExit):sampler.parse_args([*common,*extra])


def test_calibration_takes_only_historical_real_all5000_and_divides_by_channels(tmp_path):
    energy=np.full((2,3,2,1024),999.,dtype=np.float64)
    energy[0,0,0]=.25;energy[0,0,1]=.75
    p=tmp_path/'energies.npz'
    np.savez(p,global_equal_class_energy=energy,channels=np.arange(1024),
             views=['all_5000','exclude_21_shared_sources'],arms=['real','historical_scale1','ig_1p78'],
             components=['spatial_DC','spatial_AC'])
    moments,_=sampler.load_calibration(p,expected_sha=sampler.sha256_file(p))
    assert moments==[.25,.75]
    with pytest.raises(ValueError,match='frozen historical'):
        sampler.load_calibration(p,expected_sha='wrong')


def test_official_archive_and_count_accounting_with_no_extra_energy(tmp_path,monkeypatch):
    monkeypatch.setattr(sampler,'COUNT',16);monkeypatch.setattr(sampler,'LATENT_SHAPE',(1,1,2))
    monkeypatch.setattr(sampler.production,'sampling_step',lambda m,p,z,y,t,s,**k:z*.9)
    monkeypatch.setattr(sampler,'project_cohort',lambda *a,**k:pytest.fail('official must bypass projection'))
    monkeypatch.setattr(sampler,'decode',lambda decoder,state:np.full((len(state),256,256,3),127,dtype=np.uint8))
    noise=torch.ones(16,1,1,2);labels=torch.arange(16)
    result=sampler.sample(None,None,noise,labels,[1.,.5,0.],[.2,.8],SimpleNamespace(mode='official'),tmp_path,torch.device('cpu'))
    assert result['stage2_forward_calls']==4 and result['stage2_sample_forwards']==32
    assert result['decoder_forward_calls']==2 and result['decoder_sample_forwards']==16
    assert result['energy_reductions_in_official']==0 and result['diagnostic_steps']==0
    assert not (tmp_path/'step_diagnostics.jsonl').exists()
    with np.load(tmp_path/'samples.npz',allow_pickle=False) as z:
        assert z['arr_0'].shape==(16,256,256,3) and z['arr_0'].dtype==np.uint8
        assert np.array_equal(z['ids'],np.arange(16)) and np.array_equal(z['labels'],np.arange(16))
    assert len(json.loads((tmp_path/'batch_manifest.json').read_text())['batches'])==2
