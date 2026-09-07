"""One shared native-feature log-variance head with a convex Gaussian risk."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

PLAN = {
    'name': 'conditional_variance64k',
    'train': {'count': 64000, 'time_seed': 202609109},
    'validation': {'count': 8000, 'time_seed': 202609110},
    'batch': 8, 'steps': 100, 'feature_dimension': 2880,
    'parameters_including_bias': 2881, 'prefix_depth': 8,
    'features': 'native BF16-autocast block7 patch tokens, FP32 RMS pooled first and second moments',
    'real_noise_pairing': 'one real row global_id with the same global_id existing native initial epsilon',
    'normalization': 'training feature mean and one global RMS; append1',
    'target': 'R=mean_coordinates((real_clean-native_guided_clean)^2)',
    'variance': 'old_mse0(t) * exp(theta dot normalized_features)',
    'risk': '.5 mean(h + (R/mse0) expm1(-h)) + .5 ridge ||theta||^2',
    'ridge': 2881/64000, 'ridge_rule': 'unit-predictive-logvariance Gaussian prior, dim/N independent real-noise pairs',
    'optimizer': {'method': 'L-BFGS-B', 'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-13, 'maxls': 30},
    'entry': 'gradient maxabs<1e-5 and validation mean NLL change+2 class SE<0',
    'validation_uncertainty': 'descriptive1000-class cluster SE, not a formal guarantee or untouched historical holdout',
    'sampling': {'samples': [1000, 5000], 'seeds': [202609071, 202609072],
                 'all100_times': True, 'no_strength_or_window': True, 'no_extra_main_calls_or_input_backward': True},
    'no_parameter_grid_or_best_checkpoint_selection': True,
}


def query_indices(split):
    count = PLAN[split]['count']//8
    assert count % 100 == 0
    indices = np.tile(np.arange(100), count//100)
    np.random.default_rng(PLAN[split]['time_seed']).shuffle(indices)
    return indices


def native_token_statistics(tokens):
    value = tokens.float()
    normalized = value/value.square().mean(-1, keepdim=True).add(1e-8).sqrt()
    return torch.cat((normalized.mean(1), normalized.square().mean(1)), dim=-1)


class NativePrefixTap:
    def __init__(self, model):
        assert model.base_model_depth == PLAN['prefix_depth']
        self.patches = model.s_embedder.num_patches
        self.value = None
        self.calls = 0
        self.token_dtype = None
        self.handle = model.blocks[model.base_model_depth-1].register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.token_dtype = str(output.dtype)
        self.value = native_token_statistics(output[:, :self.patches].detach())
        self.calls += 1

    def take(self):
        assert self.value is not None, 'native features missing for this query'
        result, self.value = self.value, None
        return result

    def close(self):
        self.handle.remove()
        self.value = None


class ConditionalVarianceRisk:
    def __init__(self, features, ratio, ridge):
        self.x = np.asarray(features, dtype=np.float64)
        self.y = np.asarray(ratio, dtype=np.float64)
        assert self.x.ndim == 2 and self.y.shape == (len(self.x),)
        assert np.isfinite(self.x).all() and np.isfinite(self.y).all() and (self.y > 0).all()
        self.ridge = ridge

    def losses(self, theta):
        h = self.x@theta
        with np.errstate(over='raise', invalid='raise'):
            return .5*(h+self.y*np.expm1(-h))

    def __call__(self, theta):
        h = self.x@theta
        with np.errstate(over='raise', invalid='raise'):
            exp = np.exp(-h)
            value = .5*np.mean(h+self.y*np.expm1(-h))+.5*self.ridge*np.dot(theta, theta)
            gradient = self.x.T@(.5*(1-self.y*exp))/len(self.x)+self.ridge*theta
        assert np.isfinite(value) and np.isfinite(gradient).all()
        return float(value), gradient


class ConditionalVarianceHead(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        fitted = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        assert fitted['validation']['entry_condition_passed']
        assert fitted['plan'] == PLAN
        self.plan = fitted['plan']
        self.validation = fitted['validation']
        self.calibration_sha256 = fitted['calibration_sha256']
        self.register_buffer('weight', fitted['weight'].double())
        self.register_buffer('mean', fitted['feature_mean'].double())
        self.register_buffer('scale', torch.tensor(fitted['feature_scale'], dtype=torch.float64))
        self.register_buffer('bias', torch.tensor(fitted['bias'], dtype=torch.float64))

    def noise_coefficient(self, features, mse0, q, zero=False):
        weight = self.weight*0 if zero else self.weight
        bias = self.bias*0 if zero else self.bias
        h = ((features.double()-self.mean)/self.scale)@weight+bias
        # Form the same scalar q*sqrt(mse0) before the float32 noise multiply
        # when theta=0; do not change the baseline stochastic arithmetic.
        coefficient = (float(q)*(float(mse0)*h.exp()).sqrt()).float()
        assert torch.isfinite(coefficient).all() and (coefficient > 0).all()
        return coefficient, h.detach()
