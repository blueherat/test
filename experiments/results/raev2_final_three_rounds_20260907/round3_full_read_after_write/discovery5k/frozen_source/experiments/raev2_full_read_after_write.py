"""Read the exact prewritten native IG increment with the Full predictor.

This is finite feedback through the real model, not a Taylor expansion in IG
strength. The cached-reader identity is mathematical; native arithmetic is
preserved by adding the finite response to the original guided clean head.
"""

MODES = ('official', 'full_read_after_write', 'write_null')
PLAN = {
    'protocol': 'raev2_full_read_after_write_v1',
    'message': 'M = native BF16 B+1.78*(F-B), then FP32, minus F.float()',
    'prewritten_state': 'z_hat = z + ((t-s)/s)*M, with s>0',
    'cached_identity': '(s/t)*z_hat+(1-s/t)*F(z,t) = native IG Euler in real arithmetic',
    'reader': 'Full(z_hat,t), same current time, class and native B8',
    'finite_response': 'Full(z_hat,t)-Full(z,t)',
    'effective_clean': 'G_native + finite_response, in FP32',
    'strength': 'Original IG1.78, no new strength, horizon, window or training.',
    'scope': 'Original 100-step shift8 grid; only active source queries t>=.1.',
    'control': 'write_null reads the unchanged input and must reproduce original pixels.',
    'theory': 'Exact cached prewrite identity plus measured finite Full feedback; no quality, contraction or local-linearity guarantee.',
}


def euler(state, clean, current, following):
    return state-(current-following)*((state-clean)/max(current, .05))


def prewritten_state(state, full, native_guided, current, following):
    if not 0 < following < current:
        raise ValueError('prewriting requires 0 < following < current')
    return state+((current-following)/following)*(native_guided-full)


def reread_clean(native_guided, full_written, full_original):
    # Parentheses preserve the exact zero-response control in floating point.
    return native_guided+(full_written-full_original)
