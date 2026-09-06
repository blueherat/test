"""Least-squares removal of guidance parallel to the weak transport estimate."""
import torch


def projected_guidance(state, full, base, native_guided, t, *, coordinate):
    """Preserve native IG plus an FP32 geometric correction, per complete image.

    velocity: projection on B-z, as in the optimized scale of CFG-Zero*.
    noise: projection on z-(1-t)B, i.e. the physical Gaussian noise estimate.
    The clean-space correction is -gamma Proj_reference(F-B) in either case.
    No zero-init window, clipping, restoration, or new gain is included.
    """
    if coordinate not in ('velocity','noise'):
        raise ValueError('unknown transport coordinate')
    if t<.1:
        return native_guided
    f,b=full.float(),base.float()
    gap=f-b
    ref=b-state if coordinate=='velocity' else state-(1-t)*b
    dims=tuple(range(1,state.ndim))
    energy=ref.square().sum(dims,keepdim=True)
    denom=torch.where(energy>0,energy,torch.ones_like(energy))
    parallel=(gap*ref).sum(dims,keepdim=True)/denom*ref
    return native_guided-.78*parallel
