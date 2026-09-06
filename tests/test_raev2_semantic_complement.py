import torch
from experiments.raev2_semantic_complement import semantic_correction


def test_constrained_optimum_and_degenerate_depth():
    full = torch.tensor([[4., 3.], [4., 3.]])
    base = torch.tensor([[2., 3.], [4., 3.]])
    null = torch.tensor([[1., -1.], [1., -1.]])
    delta = semantic_correction(full, base, null, orthogonal=True)
    torch.testing.assert_close(delta, torch.tensor([[0., .6], [.45, .6]]))
    assert torch.equal((delta * (full - base)).sum(1), torch.zeros(2))


def test_zero_class_difference_preserves_native_bf16_anchor():
    full = torch.tensor([[1., .00390625, -2.]], dtype=torch.bfloat16)
    base = torch.tensor([[.9921875, .0078125, -1.]], dtype=torch.bfloat16)
    native = (base + 1.78 * (full - base)).float()
    for orthogonal in (False, True):
        corrected = native + semantic_correction(full, base, full, orthogonal=orthogonal)
        assert torch.equal(corrected, native)
