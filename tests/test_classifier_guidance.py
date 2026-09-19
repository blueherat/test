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
    # Independent finite differences, not just another autograd expression.
    assert torch.autograd.gradcheck(lambda x, *parameters: engine(x, labels), params,
                                    eps=1e-6, atol=1e-6, rtol=1e-4)
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
@pytest.mark.parametrize('heun', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_replay_refreshes_labels_weights_and_saved_trajectories(heun, dtype):
    torch.manual_seed(14)
    head = torch.nn.Linear(3, 3).cuda().to(dtype=dtype)
    x = torch.randn(2, 3, device='cuda', dtype=dtype, requires_grad=True)
    labels = torch.tensor([1, 2], device='cuda')
    grid = torch.tensor([0., .137, .413, .731, 1.], device='cuda', dtype=dtype)
    amount = torch.tensor([.4, .7, 0, 0], device='cuda', dtype=dtype)
    active = [True, True, False, False]

    def field(x, t, y, a, enabled):
        strong = torch.sin(x*.7+t+y[:, None]*.1)
        return strong+a*(strong-head(x)) if enabled else strong

    graph = Sampler(field, head, x, labels, grid, amount, active, heun=heun, graphs=True)
    eager = Sampler(field, head, x, labels, grid, amount, active, heun=heun, graphs=False)
    for i in range(2):
        # Interleave outstanding trajectories: static capture buffers get reused.
        y = labels+i
        ga, gb = graph(x, y), graph(x*.5, y.flip(0))
        ea, eb = eager(x, y), eager(x*.5, y.flip(0))
        torch.testing.assert_close(ga, ea, rtol=0, atol=0)
        torch.testing.assert_close(gb, eb, rtol=0, atol=0)
        gg = torch.autograd.grad(ga.square().sum()+gb.sin().sum(), (x, *head.parameters()))
        eg = torch.autograd.grad(ea.square().sum()+eb.sin().sum(), (x, *head.parameters()))
        tolerance = 1e-12 if dtype == torch.float64 else 1e-5
        for a,b in zip(gg,eg): torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
        with torch.no_grad():
            for p,g in zip(head.parameters(),gg[1:]):p.add_(g,alpha=-.01)
    with pytest.raises(ValueError, match='captured shape'):
        graph(x[:1], labels)
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match='one CUDA stream'):
            graph(x, labels)
    value = graph(x, labels)
    with torch.no_grad(): head.weight.add_(.01)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        value.sum().backward()
    with torch.no_grad(): head.weight.data = head.weight.detach().clone()
    with pytest.raises(RuntimeError,match='storage changed'):
        graph(x, labels)


def test_parameter_replacement_and_capture_contracts_fail_explicitly():
    head = torch.nn.Linear(2, 2).double()
    x = torch.ones(1, 2, dtype=torch.double)
    y = torch.tensor([0])
    grid = torch.tensor([0., .5, 1.], dtype=torch.double)
    field = lambda x, t, y, a, active: head(x).sin()
    engine = Sampler(field, head, x, y, grid, [1., 1.], [True, True], heun=False, graphs=False)
    out = engine(x, y)
    # .data assignment bypasses PyTorch's version counter, so check storage too.
    with torch.no_grad(): head.weight.data = head.weight.detach().clone()
    with pytest.raises(RuntimeError, match='storage changed'):
        out.sum().backward()
    engine = Sampler(field, head, x, y, grid, [1., 1.], [True, True], heun=False, graphs=False)
    # Replacing a Parameter while keeping its storage must also be detected.
    head.weight = torch.nn.Parameter(head.weight.detach())
    with pytest.raises(RuntimeError, match='storage changed'):
        engine(x, y)
    with pytest.raises(ValueError, match='Learned time grids'):
        Sampler(field, head, x, y, grid.requires_grad_(), [1., 1.], [True, True], heun=False, graphs=False)


@pytest.mark.parametrize('heun', [False, True])
def test_binary_gan_update_matches_direct_unroll_with_updated_critic(heun):
    """Check full first-order GAN update, feature R1 and frozen-input derivatives."""
    import copy
    from torch.nn import functional as F
    from classifier_guidance.training import step
    from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic

    torch.manual_seed(21)
    head = torch.nn.Linear(3, 3).double()
    direct_head = copy.deepcopy(head)
    critic = BinaryCritic(dimension=4, hidden=5, classes=2).double()
    direct_critic = copy.deepcopy(critic)
    decode = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh()).double().eval().requires_grad_(False)
    feature = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Sigmoid()).double().eval().requires_grad_(False)
    frozen = torch.randn(3, 3, dtype=torch.double)
    noise = torch.randn(4, 3, dtype=torch.double)
    real = torch.randn(4, 5, dtype=torch.double)
    labels = torch.tensor([0, 1, 0, 1])
    grid = torch.tensor([0., .15, .45, .7, 1.], dtype=torch.double)
    amounts, active = [.7, 1.1, 0., 0.], [True, True, False, False]

    def field(w, x, t, y, amount, enabled):
        s = torch.tanh(x@frozen+t+y[:, None]*.1)
        # Shared frozen features depend on x and must NOT be detached.
        weak = w(torch.sin(x@frozen+t))
        return s+amount*(s-weak) if enabled else s

    engine = Sampler(lambda *args: field(head, *args), head, noise, labels, grid,
                     amounts, active, heun=heun, graphs=False)
    ow = torch.optim.Adam(head.parameters(), lr=1e-3, betas=(.9, .99))
    od = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0., .99))
    rw = torch.optim.Adam(direct_head.parameters(), lr=1e-3, betas=(.9, .99))
    rd = torch.optim.Adam(direct_critic.parameters(), lr=1e-3, betas=(0., .99))
    actual = step(head=head, critic=critic, optimizer_w=ow, optimizer_d=od,
                  sample=engine, decode=decode, feature=feature, real=real, noise=noise,
                  labels=labels, r1=.7, feature_chunk=0)

    x = noise
    for i, enabled in enumerate(active):
        h = grid[i+1]-grid[i]
        v = field(direct_head, x, grid[i], labels, amounts[i], enabled)
        p = x+h*v
        x = x+h/2*(v+field(direct_head, p, grid[i+1], labels, amounts[i], enabled)) if heun else p
    generated_features = feature(decode(x))
    features = torch.cat([feature(real).detach(), generated_features.detach()]).requires_grad_(True)
    logits = direct_critic(features, labels.repeat(2))
    ce = F.softplus(-logits[:4]).mean()+F.softplus(logits[4:]).mean()
    r1 = torch.autograd.grad(logits[:4].sum(), features, create_graph=True)[0][:4].square().sum(1).mean()
    (ce+.35*r1).backward()
    torch.nn.utils.clip_grad_norm_(direct_critic.parameters(), 10.)
    rd.step()
    direct_critic.eval().requires_grad_(False)
    loss = F.softplus(-direct_critic(generated_features, labels)).mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(direct_head.parameters(), 1.)
    rw.step()
    torch.testing.assert_close(actual[:4], torch.stack([ce.detach(), r1.detach(), loss.detach(), norm]),
                               rtol=1e-12, atol=1e-12)
    for left, right in ((head, direct_head), (critic, direct_critic)):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], rtol=1e-12, atol=1e-12)
    assert all(p.grad is None for module in (decode, feature) for p in module.parameters())
