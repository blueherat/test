"""Exact first derivatives of a deterministic discrete sampler, by recomputation.

The forward stores states, not activation tapes. The backward recomputes one step
at a time and differentiates that *discrete* step. This is not an approximate
continuous adjoint, last-K truncation, or a shortcut through the frozen backbone.
Parameters and all other step-function state must remain unchanged until backward.
"""

import torch
from torch.autograd.function import once_differentiable


class _RecomputedRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, step, steps, initial, *parameters):
        ctx.step = step
        ctx.steps = steps
        states = [initial.detach()]
        for index in range(steps):
            states.append(step(index, states[-1]).detach())
        # Saving parameters also enables PyTorch's in-place version checks.
        ctx.save_for_backward(*states[:-1], *parameters)
        return states[-1]

    @staticmethod
    @once_differentiable
    def backward(ctx, terminal_gradient):
        saved = ctx.saved_tensors
        states, parameters = saved[:ctx.steps], saved[ctx.steps:]
        parameter_gradients = [None] * len(parameters)
        adjoint = terminal_gradient
        for index in reversed(range(ctx.steps)):
            with torch.enable_grad():
                state = states[index].detach().requires_grad_(True)
                output = ctx.step(index, state)
                if output.requires_grad:
                    derivatives = torch.autograd.grad(
                        output, (state, *parameters), grad_outputs=adjoint,
                        allow_unused=True, create_graph=False,
                    )
                else:
                    derivatives = (None,) * (1 + len(parameters))
            adjoint = derivatives[0]
            if adjoint is None:
                adjoint = torch.zeros_like(state)
            for position, gradient in enumerate(derivatives[1:]):
                if gradient is None:
                    continue
                if parameter_gradients[position] is None:
                    parameter_gradients[position] = gradient
                else:
                    parameter_gradients[position].add_(gradient)
        return (None, None,
                adjoint if ctx.needs_input_grad[2] else None,
                *parameter_gradients)


def recomputed_rollout(step, steps, initial, parameters):
    """Apply ``step(index, state)`` repeatedly, retaining exact first gradients.

    ``step`` must be deterministic and may close over ``parameters``. Parameters
    outside this explicit list must be frozen. Memory is O(steps * state size)
    plus one step's activations. Recomputed backward increases training compute.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    parameters = tuple(parameters)
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("Only trainable parameters belong in the explicit list")
    return _RecomputedRollout.apply(step, steps, initial, *parameters)


def ordinary_rollout(step, steps, initial):
    """Untruncated autograd reference, useful for checking recomputation."""
    state = initial
    for index in range(steps):
        state = step(index, state)
    return state
