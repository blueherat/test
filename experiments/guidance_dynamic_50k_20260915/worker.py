import argparse
from . import config as k


def main():
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--model',choices=k.MODELS,required=True)
    for key in ('arm','source','split','point'):p.add_argument('--'+key)
    for key in ('rank','fold','n'):p.add_argument('--'+key,type=int)
    a=p.parse_args();k.verify()
    if a.action in ('normalize','check','train','nuisance'):
        from . import training
        context=training.distributed()
        try:
            if a.action=='normalize':training.normalize(a.model,context)
            elif a.action=='check':
                from .checks import gpu
                gpu(a.model,context)
            else:training.train(a.model,a.arm,context,a.source,a.fold)
        finally:training.base.cleanup_distributed(context)
    elif a.action=='inputs':
        from .sampling import quality_inputs
        quality_inputs(a.model)
    elif a.action in ('sample','collect','evaluate'):
        from . import sampling
        if a.action=='sample':sampling.sample(a.model,a.point,a.n,a.rank)
        else:getattr(sampling,a.action)(a.model,a.point,a.n)
    elif a.action in ('generate','endpoints_collect'):
        from . import endpoints
        if a.action=='generate':endpoints.generate(a.model,a.source,a.split,a.rank)
        else:endpoints.collect(a.model,a.source,a.split)
    elif a.action in ('features','weights'):
        from . import weights
        if a.action=='features':weights.features(a.model,a.rank)
        else:weights.fit(a.model)
    else:raise ValueError(a.action)


if __name__=='__main__':
    try:main()
    except k.RequestedStop as e:
        print(str(e),flush=True);raise SystemExit(75)
