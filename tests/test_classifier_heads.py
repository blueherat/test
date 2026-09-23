import copy

import torch

from classifier_guidance.heads import Head, DeeperMLPHead, deepen, load_weights, restore_head_optimizer


def fixture():
    torch.manual_seed(21)
    head = Head(8, 3, 2).eval()
    # Production readouts are pretrained, unlike the zero-output constructor.
    with torch.no_grad():
        head.output.weight.normal_(std=.2)
        head.output.bias.normal_(std=.1)
    return head, torch.randn(2, 4, 8, requires_grad=True), torch.randn(2, 8, requires_grad=True)


def test_expansion_preserves_function_and_existing_gradients_but_can_learn():
    head, tokens, condition = fixture()
    rng = torch.get_rng_state().clone()
    expanded = deepen(head)
    assert torch.equal(torch.get_rng_state(), rng)
    old, new = head(tokens, condition), expanded(tokens, condition)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    g_old = torch.autograd.grad(old.square().mean(), (tokens, condition, *head.parameters()))
    g_new = torch.autograd.grad(new.square().mean(), (tokens, condition, *expanded.parameters()))
    for a, b in zip(g_old, g_new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(g.abs().sum() > 0 for g in g_new[-2:])
    with torch.no_grad():
        expanded.hidden_residual.weight.add_(g_new[-2], alpha=-.1)
        expanded.hidden_residual.bias.add_(g_new[-1], alpha=-.1)
    assert not torch.equal(expanded(tokens, condition), old)
    restored = DeeperMLPHead(8, 3, 2).eval()
    load_weights(restored, expanded.state_dict())
    torch.testing.assert_close(restored(tokens, condition), expanded(tokens, condition), rtol=0, atol=0)


def test_optimizer_migration_preserves_old_adam_and_initializes_only_new_parameters():
    head, tokens, condition = fixture()
    old = torch.optim.Adam(head.parameters(), lr=1e-3)
    head(tokens, condition).square().mean().backward()
    old.step()
    state = copy.deepcopy(old.state_dict())
    saved_weights = copy.deepcopy(head.state_dict())
    expanded = deepen(head)
    new = torch.optim.Adam(expanded.parameters(), lr=1e-3)
    added = restore_head_optimizer(new, state, expanded, saved_weights)
    assert added == ['hidden_residual.weight', 'hidden_residual.bias']
    for name, parameter in head.named_parameters():
        target = dict(expanded.named_parameters())[name]
        for key, value in old.state[parameter].items():
            torch.testing.assert_close(new.state[target][key], value, rtol=0, atol=0)
    assert expanded.hidden_residual.weight not in new.state
    assert expanded.hidden_residual.bias not in new.state
    new.zero_grad(set_to_none=True)
    expanded(tokens, condition).square().mean().backward()
    new.step()
    assert new.state[expanded.hidden_residual.weight]['step'].item() == 1
    assert new.state[expanded.token.weight]['step'].item() == 2
    # A later deeper checkpoint restores by name without another expansion.
    restored = deepen(head)
    load_weights(restored, expanded.state_dict())
    optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    assert restore_head_optimizer(optimizer, new.state_dict(), restored, expanded.state_dict(),
                                  [name for name, _ in expanded.named_parameters()]) == []
    torch.testing.assert_close(optimizer.state[restored.hidden_residual.weight]['exp_avg'],
                               new.state[expanded.hidden_residual.weight]['exp_avg'], rtol=0, atol=0)
