"""Finite Base response to the extra IG write, with a paired Full control."""
import torch

MODES = ('official', 'causal_reference', 'causal_null')
PLAN = {
    'protocol': 'raev2_causal_reference_v1',
    'full_clean': 'F(z,t)', 'native_guided_clean': 'native BF16 B+1.78*(F-B), then FP32',
    'query_without_extra_ig': 'Euler(z,F,t,s)',
    'query_with_extra_ig': 'Euler(z,G_native,t,s)',
    'finite_response': 'B(query_with_extra_ig,s)-B(query_without_extra_ig,s)',
    'new_reference': 'B(z,t)+finite_response',
    'effective_clean': 'G_native-1.78*finite_response, in FP32',
    'anchor': 'Original current weak transport anchor is preserved.',
    'strength': 'Unchanged original 1.78; no new gain, horizon, time window or trained parameter.',
    'scope': 'Original active source queries t>=.1; unchanged 100-step shift8 grid.',
    'theory': 'Exact finite counterfactual reference algebra, not a small-strength expansion, accuracy guarantee or fixed-point solution.',
}


def euler(state, clean, current, following):
    return state-(current-following)*((state-clean)/max(current, .05))


def calibrated_clean(native_guided, base_written, base_unwritten, beta=1.78):
    return native_guided-beta*(base_written-base_unwritten)

