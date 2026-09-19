"""Unoptimized historical discrete-adjoint reference for paired validation."""
from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import recomputed_rollout
from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps


def sample(adapter, head, noise, labels, coefficient):
    if adapter.name != 'raev2':
        steps = GuidanceSteps(adapter, head, labels, coefficient, method='real')
    else:
        from experiments.raev2_shallow_ig_20260914.core import predict
        class Steps:
            def __len__(self): return 100
            def __call__(self, i, x):
                rt=adapter.rt;rt.labels=labels
                t,u=rt.grid[i],rt.grid[i+1]
                with rt.context():
                    full,_=rt.pair(x,t)
                    if .1<=float(t)<=1.:
                        clean=adapter.unpatch(predict('mlp',head,adapter.values)).float()
                        weak=rt.native.clean_to_velocity(clean,x,rt.times(x,t),denominator_floor=float(rt.cfg.transport.t_eps))
                        full=full+coefficient*(full-weak)
                    return x+(u-t)*full
        steps=Steps()
    return recomputed_rollout(steps,len(steps),noise,head.parameters())
