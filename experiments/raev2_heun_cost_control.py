"""Native-IG Heun arithmetic for an independently fixed inference budget."""


def predictor(state, clean, t, s, eps=.05):
    return state - (t - s) * ((state - clean) / max(t, eps))


def corrected(state, predicted, clean, next_clean, t, s, eps=.05):
    return state - .5 * (t - s) * (
        (state - clean) / max(t, eps) + (predicted - next_clean) / max(s, eps))
