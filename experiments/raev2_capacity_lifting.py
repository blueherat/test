"""Finite flow composition supplies guidance; no extra native IG drift."""

def lift_strength(t, alpha, schedule):
    if schedule == 'constant': return float(alpha)
    if schedule == 'fade': return float(alpha)*max(0.,min(1.,(float(t)-.2)/.3))
    raise ValueError(schedule)


def blocks(grid, alpha, schedule):
    k=0
    while k<len(grid)-1:
        t=float(grid[k]);a=lift_strength(t,alpha,schedule) if .1<=t<=1. else 0.
        if a==0.:
            yield k,k+1,0.; k+=1; continue
        stop=min(k+4,len(grid)-1)
        # Do not straddle either the native active-window end or the zero-lift tail.
        while stop>k+1 and (float(grid[stop-1])<.1 or lift_strength(float(grid[stop-1]),alpha,schedule)==0.):stop-=1
        yield k,stop,a
        k=stop


def heun_flow(z, grid, kind, field):
    for t,s in zip(grid[:-1],grid[1:]):
        v=field(z,t,kind)
        z=z+(s-t)*.5*(v+field(z+(s-t)*v,s,kind))
    return z


def advance_block(z, grid, alpha, field):
    if alpha:
        target=heun_flow(z,grid,'full',field)
        lifted=heun_flow(target,grid.flip(0),'base',field)
        z=z+alpha*(lifted-z)
    # The guidance has been supplied by the finite lift. Main fine steps remain Euler.
    for t,s in zip(grid[:-1],grid[1:]):z=z+(s-t)*field(z,t,'full')
    return z
