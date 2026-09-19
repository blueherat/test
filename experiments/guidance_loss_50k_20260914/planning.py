"""An explicit DAG: every candidate is executable, with bounded follow-up rounds."""
from . import config as k


def jobs():
    result=[]
    def add(action,depends=(),**kwargs):
        name='_'.join([action,*[str(v) for v in kwargs.values()]])
        result.append(dict(id=name,action=action,depends=list(depends),gpu=action not in ('weights','evaluate'),**kwargs))
        return name
    preflight=add('gpu_preflight')
    train_jobs={}
    def train(arm,deps=()):
        train_jobs[arm]=add('train',[preflight,*deps],arm=arm)
    def quality(arm):
        deps=[] if arm in ('native','cfg_native','context') else [train_jobs[arm.removesuffix('_half')]]
        sample=add('sample',[preflight,*deps],arm=arm,stage='screen1000')
        add('evaluate',[sample],arm=arm,stage='screen1000')
    for arm in ('real','self','gaussian','mixture'):train(arm)
    features=add('features',[preflight])
    weights=add('weights',[features])
    for arm in ('excess','shuffled'):train(arm,[weights])
    for arm in ('real','self','gaussian','gaussian_half','mixture','excess','shuffled','context','native','cfg_native'):
        quality(arm)
    def nuisance(source,deps=()):
        return [add('nuisance',[preflight,*deps],source=source,fold=fold) for fold in (0,1)]
    nuisance_g=nuisance('strong')
    for arm in ('real_residual','guided_weak','contrast_s','contrast_m','contrast_weak','covariance','contrast_null'):
        deps=nuisance_g if k.ARMS[arm]['loss'] in ('contrast','contrast_weak','covariance') else []
        train(arm,deps)
        quality(arm)
    weak=add('endpoints',[preflight,train_jobs['self']],source='weak')
    nuisance_w=nuisance('weakmix',[weak])
    train('weakmix_fm',[weak]);quality('weakmix_fm')
    train('weakmix_contrast',nuisance_w);quality('weakmix_contrast')
    for source in ('ig','cfg'):
        endpoint=add('endpoints',[preflight],source=source)
        nuis=nuisance(source,[endpoint])
        train(source+'_residual');quality(source+'_residual')
        train(source+'_contrast',nuis);quality(source+'_contrast')
    return result


def output(job):
    action=job['action']
    if action=='gpu_preflight':return k.ROOT/'gpu_preflight.json'
    if action=='train':return k.ROOT/'training'/job['arm']/'complete.json'
    if action=='nuisance':return k.ROOT/'nuisance'/job['source']/f"fold{job['fold']}"/'complete.json'
    if action=='endpoints':return k.ROOT/'endpoints'/job['source']/'complete.json'
    if action in ('features','weights'):return k.ROOT/'weights'/('features_complete.json' if action=='features' else 'complete.json')
    return k.ROOT/k.MODEL/job['stage']/job['arm']/('summary.json' if action=='sample' else 'metrics.json')


def verify_output(job):
    path=output(job)
    if not path.exists():return False
    row=k.read(path)
    assert row['request_sha256']==k.sha(k.ROOT/'request.json')
    if job['action']=='gpu_preflight':
        return bool(row.get('passed') and row.get('new_objectives_and_cfg_features_checked'))
    assert row['complete']
    for group in ('files','dependencies','head_provenance'):
        for name,digest in row.get(group,{}).items():assert k.sha(name)==digest
    if job['action'] in ('train','nuisance'):
        assert row['steps']==50000 and row['head_sha256']==k.sha(path.parent/'head.pt')
    if job['action'] in ('sample','evaluate'):
        assert row['samples_sha256']==k.sha(path.parent/'samples.npz')
        assert row['full_calls_per_output']==k.inference_counts(job['arm'])['full']
    return True
