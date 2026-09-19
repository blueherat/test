"""Exercise the actual amended planner in memory without changing live jobs."""
from . import install,k,NAME,REQUEST
from experiments.guidance_dynamic_recovery_20260915 import pipeline as recovery_pipeline
from experiments.guidance_dynamic_50k_20260915 import pipeline as original


def probe():
    expected_original = k.read(k.ROOT/'request.json')['methods']
    assert [m for m in k.METHODS if m!=NAME]==expected_original
    assert k.METHODS.index(NAME)==k.METHODS.index('guided_weak')+1
    assert k.STEPS==50000 and k.GLOBAL_BATCH==256 and k.WORLD==2
    assert k.SEARCH_STEPS==(16,8,4,1)
    cases = [
        ('sit_candidate',{'sit_small/real','sit_small/guided_weak'},('sit_small',NAME)),
        ('jit_real',{f'sit_small/{m}' for m in k.METHODS},('jit','real')),
        ('jit_candidate',{f'sit_small/{m}' for m in k.METHODS}|{'jit/real','jit/guided_weak'},('jit',NAME))]
    results=[]
    for label,complete,expected in cases:
        planner=original.Supervisor.__new__(original.Supervisor)
        planner.jobs={};planner.ledger={};planner.active={};planner.gpus=[]
        planner.searches={key:dict(phase='complete') for key in complete}
        planner.current=None
        allowed=planner.advance()
        assert planner.current==expected
        assert {planner.jobs[name]['model'] for name in allowed}=={expected[0]}
        train=planner.jobs['train__'+expected[0]+'__'+expected[1]]
        assert train['gpus']==2
        assert all(j['gpus']==2 for j in planner.jobs.values() if j['action'] in ('normalize','check_runtime2','train'))
        results.append(dict(case=label,current=list(planner.current),training_gpus=train['gpus'],allowed_jobs=sorted(allowed)))
    result=dict(passed=True,cpu_only=True,live_ledger_untouched=True,
        candidate_request_sha256=k.sha(REQUEST),cases=results)
    k.atomic(k.ROOT/'deployed_handoff/planning_checks.json',result)
    print(result)


if __name__=='__main__':
    original.main=probe
    recovery_pipeline.install=install
    recovery_pipeline.main()
