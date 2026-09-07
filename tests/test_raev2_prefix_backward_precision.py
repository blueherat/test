"""The guidance gradient must retain FP32 protection through backward."""
import torch
from experiments.raev2_prefix_ratio_guidance import PrefixRatioHead, prefix_fp32


class PrecisionWitness(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        assert not torch.is_autocast_enabled(value.device.type)
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        ctx.save_for_backward(value)
        return value.square().flatten(1).sum(1)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        assert not torch.is_autocast_enabled(value.device.type)
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        return 2*value*gradient[:, None, None, None]


class WitnessHead(PrefixRatioHead):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def potential(self, model, state, times, labels):
        with prefix_fp32(state.device.type):
            return PrecisionWitness.apply(state)


def test_precision_context_covers_backward_and_restores_enclosing_state():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        state = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
        times = torch.tensor([1., .3])
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            result = WitnessHead().clean_correction(None, state, times, torch.arange(2))
            assert torch.is_autocast_enabled('cpu')
        torch.testing.assert_close(result, 2*state*times[:, None, None, None].square())
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert not result.requires_grad
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
