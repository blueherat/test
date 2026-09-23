"""Function-preserving expansion of the existing intermediate MLP readout."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.guidance_distribution_20260912.local_head import Head


class DeeperMLPHead(Head):
    """One additional hidden Linear+SiLU residual branch before the readout.

    h = SiLU(token(x) + condition(c) + position(p))
    output = readout(h + SiLU(hidden_residual(h))).
    Zero extra weights preserve the trained shallow head, including input VJPs.
    """

    def __init__(self, width, outdim, side):
        super().__init__(width, outdim, side)
        self.hidden_residual = nn.Linear(width, width)
        nn.init.zeros_(self.hidden_residual.weight)
        nn.init.zeros_(self.hidden_residual.bias)

    def forward(self, tokens, condition):
        x = (tokens.float() - self.token_mean) / self.token_std
        y = (condition.float() - self.condition_mean) / self.condition_std
        hidden = F.silu(self.token(x) + self.condition(y)[:, None] + self.position(self.positions))
        hidden = hidden + F.silu(self.hidden_residual(hidden))
        return self.output(hidden)


def deepen(head):
    if isinstance(head, DeeperMLPHead):
        return head
    side = math.isqrt(head.positions.shape[1])
    if side * side != head.positions.shape[1]:
        raise ValueError('The readout requires a square position grid')
    # Do not change the random stream used by the caller's critic/data setup.
    with torch.random.fork_rng(devices=[]):
        expanded = DeeperMLPHead(head.token.in_features, head.output.out_features, side)
    expanded.to(device=head.token.weight.device, dtype=head.token.weight.dtype)
    load_weights(expanded, head.state_dict())
    expanded.train(head.training)
    expanded.requires_grad_(head.token.weight.requires_grad)
    return expanded


def load_weights(head, weights):
    """Strict loading, allowing only the explicit shallow-to-deeper migration."""
    expanded = isinstance(head, DeeperMLPHead)
    saved_expanded = 'hidden_residual.weight' in weights
    if saved_expanded and not expanded:
        raise ValueError('Deeper MLP weights require a DeeperMLPHead')
    if expanded and not saved_expanded:
        weights = dict(weights, **{
            'hidden_residual.weight': torch.zeros_like(head.hidden_residual.weight),
            'hidden_residual.bias': torch.zeros_like(head.hidden_residual.bias),
        })
    head.load_state_dict(weights, strict=True)


def architecture(head):
    return dict(kind='mlp_extra_hidden_residual' if isinstance(head, DeeperMLPHead) else 'mlp',
                hidden_width=head.token.out_features,
                extra_hidden_layers=int(isinstance(head, DeeperMLPHead)),
                parameters=sum(p.numel() for p in head.parameters()))


def restore_head_optimizer(optimizer, saved, head, saved_weights, saved_names=None):
    """Keep Adam moments for existing parameters; new parameters start fresh."""
    if len(saved['param_groups']) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError('Head optimizer migration supports one parameter group')
    names = [name for name, _ in head.named_parameters()]
    old_names = saved_names if saved_names is not None else [name for name in names if name in saved_weights]
    old_ids = saved['param_groups'][0]['params']
    if len(old_names) != len(old_ids) or len(set(old_names)) != len(old_names):
        raise ValueError('Checkpoint optimizer parameters do not match its head')
    if set(old_names) - set(names):
        raise ValueError('Cannot discard trained optimizer parameters')
    new_ids = optimizer.state_dict()['param_groups'][0]['params']
    ids_by_name = dict(zip(names, new_ids))
    parameters = dict(head.named_parameters())
    moments = {}
    for name, old_id in zip(old_names, old_ids):
        if old_id in saved['state']:
            values = copy.deepcopy(saved['state'][old_id])
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in values and values[key].shape != parameters[name].shape:
                    raise ValueError(f'Invalid {key} shape for {name}')
            moments[ids_by_name[name]] = values
    group = copy.deepcopy(saved['param_groups'][0])
    group['params'] = new_ids
    optimizer.load_state_dict(dict(state=moments, param_groups=[group]))
    return [name for name in names if name not in old_names]
