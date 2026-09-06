"""Difference of two Gaussian posterior means, in fixed spatial eigenspaces."""


def posterior_difference(t, real_variance, model_variance):
    a=1-t
    vp=t*t+a*a*real_variance
    vq=t*t+a*a*model_variance
    return a*t*t*(real_variance-model_variance)/(vp*vq)


def two_mode_correction(state, t, calibration):
    p,q=calibration['real'],calibration['generated']
    dc=state.mean((-2,-1),keepdim=True)
    ac=state-dc
    return (posterior_difference(t,p['dc_variance'],q['dc_variance'])*dc+
            posterior_difference(t,p['ac_variance'],q['ac_variance'])*ac)
