"""Input gradient of the one fitted frozen-prefix density-ratio potential."""
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from experiments.raev2_prefix_ratio_features import prefix_features


@contextmanager
def prefix_fp32(device_type):
    """Restore native global precision flags and any enclosing autocast mode."""
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


class PrefixRatioHead(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        fitted = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        assert fitted['validation']['entry_condition_passed'], 'unvalidated prefix head cannot enter sampling'
        self.plan = fitted['plan']
        self.validation = fitted['validation']
        self.register_buffer('weight', fitted['weight'].double())
        self.register_buffer('mean', fitted['feature_mean'].double())
        self.register_buffer('scale', torch.tensor(fitted['feature_scale'], dtype=torch.float64))
        self.register_buffer('bias', torch.tensor(fitted['bias'], dtype=torch.float64))
        assert self.weight.shape == self.mean.shape == (2880,)
        assert self.scale.item() > 0

    def from_features(self, features):
        # Keep the FP64 fitted head and normalization; only pretrained features
        # are FP32. Backpropagation casts their feature gradient to FP32.
        return ((features.double()-self.mean)/self.scale)@self.weight+self.bias

    def potential(self, model, state, times, labels):
        with prefix_fp32(state.device.type):
            features = prefix_features(model, state.float(), times.float(), labels)
            return self.from_features(features)

    def clean_correction(self, model, state, times, labels):
        with torch.enable_grad():
            variable = state.detach().float().requires_grad_(True)
            potential = self.potential(model, variable, times, labels)
            gradient, = torch.autograd.grad(potential.sum(), variable)
        # d=(1-t)f cancels the signal denominator analytically; valid at t=1.
        correction = times.float()[:, None, None, None].square()*gradient
        assert torch.isfinite(correction).all(), 'non-finite prefix input gradient'
        return correction.detach()
