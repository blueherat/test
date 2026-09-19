import pytest
import torch
import os
from classifier_guidance.sampler import Sampler
from classifier_guidance.features import chunked_features


@pytest.mark.parametrize('heun', [False, True])
def test_full_discrete_gradient_including_frozen_suffix(heun):
    torch.manual_seed(7)
    head = torch.nn.Linear(3, 3).double()
    noise = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    labels = torch.tensor([1, 2])
    grid = torch.tensor([0., .1, .35, .6, 1.], dtype=torch.double)
    amounts = torch.tensor([.7, 1.1, 0., 0.], dtype=torch.double)
    active = [True, True, False, False]
    frozen = torch.randn(3, 3, dtype=torch.double)

    def field(x, t, y, amount, enabled):
        strong = torch.tanh(x @ frozen + t + y[:, None]*.1)
        return strong + amount*(strong-head(x)) if enabled else strong

    def reference(x):
        for i, enabled in enumerate(active):
            h = grid[i+1]-grid[i]
            v = field(x, grid[i], labels, amounts[i], enabled)
            predicted = x+h*v
            x = x+(h/2)*(v+field(predicted,grid[i+1],labels,amounts[i],enabled)) if heun else predicted
        return x

    engine = Sampler(field,head,noise,labels,grid,amounts,active,heun=heun,graphs=False)
    expected = reference(noise)
    actual = engine(noise, labels)
    params = (noise, *head.parameters())
    expected_g = torch.autograd.grad(expected.sin().sum(), params)
    actual_g = torch.autograd.grad(actual.sin().sum(), params)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(actual_g, expected_g):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    # Two outstanding trajectories must not overwrite each other's saved states.
    first, second = engine(noise, labels), engine(noise*.5, labels)
    torch.autograd.grad(first.sum()+second.sum(), params)
    value = engine(noise, labels)
    with torch.no_grad(): head.weight.add_(.01)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        value.sum().backward()


def test_chunked_feedback_chain_rule_with_uneven_last_chunk():
    torch.manual_seed(8)
    decode = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh()).double().eval().requires_grad_(False)
    feature = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Sigmoid()).double().eval().requires_grad_(False)
    x = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    expected = feature(decode(x))
    actual = chunked_features(decode, feature, x, chunk=2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    g1, = torch.autograd.grad(expected.square().sum(), x)
    g2, = torch.autograd.grad(actual.square().sum(), x)
    torch.testing.assert_close(g1, g2, rtol=1e-12, atol=1e-12)
    assert all(p.grad is None for p in (*decode.parameters(), *feature.parameters()))


@pytest.mark.skipif(os.environ.get('CLASSIFIER_TEST_CUDA') != '1', reason='requires explicitly leased CUDA GPU')
def test_cuda_replay_refreshes_labels_weights_and_saved_trajectories():
    torch.manual_seed(14)
    head = torch.nn.Linear(3, 3).cuda()
    x = torch.randn(2, 3, device='cuda')
    labels = torch.tensor([1, 2], device='cuda')
    grid = torch.linspace(0, 1, 5, device='cuda')
    amount = torch.tensor([.4, .7, 0, 0], device='cuda')
    active = [True, True, False, False]

    def field(x, t, y, a, enabled):
        strong = torch.sin(x*.7+t+y[:, None]*.1)
        return strong+a*(strong-head(x)) if enabled else strong

    graph = Sampler(field, head, x, labels, grid, amount, active, heun=True, graphs=True)
    eager = Sampler(field, head, x, labels, grid, amount, active, heun=True, graphs=False)
    for i in range(2):
        # Interleave outstanding trajectories: static capture buffers get reused.
        y = labels+i
        ga, gb = graph(x, y), graph(x*.5, y.flip(0))
        ea, eb = eager(x, y), eager(x*.5, y.flip(0))
        torch.testing.assert_close(ga, ea, rtol=0, atol=0)
        torch.testing.assert_close(gb, eb, rtol=0, atol=0)
        gg = torch.autograd.grad(ga.square().sum()+gb.sin().sum(), tuple(head.parameters()))
        eg = torch.autograd.grad(ea.square().sum()+eb.sin().sum(), tuple(head.parameters()))
        for a,b in zip(gg,eg): torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        with torch.no_grad():
            for p,g in zip(head.parameters(),gg):p.add_(g,alpha=-.01)
    with torch.no_grad(): head.weight.data = head.weight.detach().clone()
    with pytest.raises(RuntimeError,match='storage changed'):
        graph(x, labels)
