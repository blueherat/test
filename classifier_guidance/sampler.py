"""Full discrete sampler gradients with graph replay and one field tape at a time.

No timestep truncation, approximate ODE adjoint, or frozen-input stop-gradient.
For Heun, the two vector-field VJPs are evaluated separately by the chain rule.
CUDA graphs keep the original operators/precision, avoiding Python launch work.
"""
import math
import torch
from torch.autograd.function import once_differentiable


class Field:
    """Explicit tensor inputs: state, time, labels, guidance amount."""

    def __init__(self, adapter, head):
        self.adapter, self.head = adapter, head

    def __call__(self, state, time, labels, amount, active):
        a = self.adapter
        with a.autocast():
            full, _ = a.full(state, time, labels)
            if active:
                tokens = a.values['tokens'] if a.name == 'raev2' else a.values['context']
                weak = a.unpatch(self.head(tokens, a.values['condition'])).float()
                if a.name == 'raev2':
                    weak = a.clean_to_velocity(weak, state, time)
                full = full + amount * (full - weak)
            return a.native_to_velocity(full, state, time)


class FieldReplay:
    """Reusable deterministic CUDA field + VJP, bound to parameter storage.

    Inputs/outputs are copied at the boundary; no graph-owned result escapes.
    Shapes are fixed per instance. Callers may update parameter values in place,
    but must rebuild this object after replacing parameter objects/storage.
    """

    def __init__(self, field, example, labels, parameters, *, graphs=True, time_dtype=None):
        self.field, self.parameters = field, tuple(parameters)
        self.graphs = graphs
        self.shape = example.shape
        self.input_signature = (example.shape, example.dtype, example.device, labels.shape, labels.dtype, labels.device)
        self.stream = torch.cuda.current_stream(example.device) if graphs else None
        self.signature = tuple((id(p), p.data_ptr(), p.shape, p.dtype) for p in self.parameters)
        self.x = example.detach().clone().requires_grad_(True)
        self.labels = labels.clone()
        self.time = torch.full((), .25, device=example.device, dtype=time_dtype or example.dtype)
        self.amount = torch.full((), 1., device=example.device, dtype=example.dtype)
        self.cotangent = torch.ones_like(example)
        self.captures = {}
        if graphs:
            # Independent captures, no cross-graph intermediates; every output
            # is cloned before another replay. This is the independent-graph
            # pool-sharing case documented in PyTorch CUDA semantics.
            pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for active in (False, True):
                    for backward in (False, True):
                        for _ in range(3):
                            self._execute(active, backward)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=stream):
                            outputs = self._execute(active, backward)
                        self.captures[active, backward] = graph, outputs
            torch.cuda.current_stream().wait_stream(stream)

    def _execute(self, active, backward):
        with torch.enable_grad() if backward else torch.no_grad():
            output = self.field(self.x, self.time, self.labels, self.amount, active)
            if not backward:
                return (output,)
            return torch.autograd.grad(output, (self.x, *self.parameters),
                                       self.cotangent, allow_unused=True)

    def run(self, state, time, labels, amount, active, gradient=None):
        signature = (state.shape, state.dtype, state.device, labels.shape, labels.dtype, labels.device)
        if signature != self.input_signature:
            raise ValueError('Field inputs must match captured shape, dtype and device')
        if not self.graphs:
            with torch.enable_grad() if gradient is not None else torch.no_grad():
                x = state.detach().requires_grad_(gradient is not None)
                out = self.field(x, time, labels, amount, active)
                if gradient is None:
                    return out.detach()
                return torch.autograd.grad(out, (x, *self.parameters), gradient, allow_unused=True)
        with torch.no_grad():
            if torch.cuda.current_stream(state.device) != self.stream:
                raise RuntimeError('Field replay is bound to one CUDA stream')
            self.x.copy_(state)
            self.time.copy_(time)
            self.labels.copy_(labels)
            self.amount.copy_(amount)
            if gradient is not None:
                self.cotangent.copy_(gradient)
            graph, outputs = self.captures[active, gradient is not None]
            graph.replay()
            copied = tuple(None if value is None else value.clone() for value in outputs)
        return copied if gradient is not None else copied[0]


class _Rollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, noise, labels, *parameters):
        engine.check_parameters(parameters)
        ctx.engine = engine
        states, predictors = [], []
        state = noise
        for i, active in enumerate(engine.active):
            states.append(state)
            t, u, amount = engine.grid[i], engine.grid[i+1], engine.amounts[i]
            h = u-t
            first = engine.replay.run(state, t, labels, amount, active)
            predicted = state + h*first
            if engine.heun:
                predictors.append(predicted)
                second = engine.replay.run(predicted, u, labels, amount, active)
                state = state + (h/2)*(first+second)
            else:
                state = predicted
        ctx.save_for_backward(labels, *states, *predictors, *parameters)
        return state

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        e = ctx.engine
        e.check_parameters(e.parameters)
        saved = ctx.saved_tensors  # Also validates parameter/tensor version counters.
        n = len(e.active)
        labels, states = saved[0], saved[1:1+n]
        predictors = saved[1+n:1+2*n] if e.heun else ()
        totals = [None] * len(e.parameters)

        def accumulate(derivatives):
            for j, value in enumerate(derivatives):
                if value is not None:
                    if totals[j] is None:
                        totals[j] = value
                    else:
                        totals[j].add_(value)

        for i in reversed(range(n)):
            t, u, a = e.grid[i], e.grid[i+1], e.amounts[i]
            h = u-t
            if e.heun:
                second = e.replay.run(predictors[i], u, labels, a, e.active[i], gradient*(h/2))
                accumulate(second[1:])
                first = e.replay.run(states[i], t, labels, a, e.active[i], gradient*(h/2)+h*second[0])
                accumulate(first[1:])
                gradient = gradient + second[0] + first[0]
            else:
                first = e.replay.run(states[i], t, labels, a, e.active[i], gradient*h)
                accumulate(first[1:])
                gradient = gradient + first[0]
        return (None, gradient if ctx.needs_input_grad[1] else None, None, *totals)


class Sampler:
    def __init__(self, field, head, example, labels, grid, amounts, active, *, heun, graphs=True):
        self.head = head
        self.parameters = tuple(head.parameters())
        if any(not p.requires_grad for p in self.parameters):
            raise ValueError('Sampler expects a trainable weak head')
        if grid.requires_grad or (torch.is_tensor(amounts) and amounts.requires_grad):
            raise ValueError('Learned time grids or guidance coefficients require a different autograd contract')
        self.grid = grid.detach().clone()
        self.amounts = torch.as_tensor(amounts, device=example.device, dtype=example.dtype).detach().clone()
        self.active = tuple(active)
        if len(self.grid) != len(self.active)+1 or len(self.amounts) != len(self.active):
            raise ValueError('Inconsistent schedule lengths')
        self.heun = heun
        self.replay = FieldReplay(field, example, labels, self.parameters, graphs=graphs, time_dtype=self.grid.dtype)

    def check_parameters(self, parameters):
        current = tuple((id(p), p.data_ptr(), p.shape, p.dtype) for p in self.head.parameters())
        if current != self.replay.signature:
            raise RuntimeError('Head parameter storage changed; rebuild sampler')

    def __call__(self, noise, labels):
        return _Rollout.apply(self, noise, labels, *self.parameters)


def for_adapter(adapter, head, example, labels, coefficient, *, graphs=True):
    if any(p.requires_grad for p in adapter.model.parameters()):
        raise ValueError('The strong model must be frozen')
    if any(m.training for root in (adapter.model, head) for m in root.modules()):
        raise ValueError('Replay/recomputation requires eval-mode strong model and head')
    if adapter.name == 'jit':
        # Preserve upstream FP32 attention logits and BF16 output multiplication.
        # Its zero bias was allocated on CPU then copied on every attention call.
        def attention(query, key, value, dropout_p=0.):
            bias = torch.zeros(query.size(0), 1, query.size(-2), key.size(-2),
                               dtype=query.dtype, device=query.device)
            with torch.autocast('cuda', enabled=False):
                weight = query.float() @ key.float().transpose(-2, -1) * (1/math.sqrt(query.size(-1)))
            weight += bias
            weight = torch.softmax(weight, dim=-1)
            weight = torch.dropout(weight, dropout_p, train=True)
            return weight @ value
        adapter.jig.model_jit.scaled_dot_product_attention = attention
    if adapter.name in ('sit_small', 'jit'):
        # Upstream builds this constant on CPU and copies it on every evaluation.
        # Compute on CPU once (preserving its rounding), then keep a GPU buffer.
        embed = adapter.model.t_embedder
        dim = embed.frequency_embedding_size
        frequencies = torch.exp(-math.log(10000) * torch.arange(dim//2, dtype=torch.float32)/(dim//2)).to(example.device)

        def timestep_embedding(t, dimension, max_period=10000):
            if dimension != dim or max_period != 10000:
                raise ValueError('Unsupported cached frequency dimensions')
            values = t[:, None].float()*frequencies[None]
            result = torch.cat([torch.cos(values), torch.sin(values)], dim=-1)
            return torch.cat([result, torch.zeros_like(result[:, :1])], dim=-1) if dim % 2 else result

        embed.timestep_embedding = timestep_embedding
    if adapter.name == 'raev2':
        grid = adapter.rt.grid
        left = grid[:-1].cpu().tolist()
        active = [.1 <= t <= 1. and coefficient != 0 for t in left]
        amounts = [coefficient if enabled else 0. for enabled in active]
    else:
        total = 64 if adapter.name == 'sit_small' else 100
        grid = torch.linspace(0, 1, total+1, device=example.device)
        left = grid[:-1].cpu().tolist()
        active = [t < .5 and coefficient != 0 for t in left]
        amounts = [coefficient*(6/7 if adapter.name == 'sit_small' and t < .25 else 1.) for t in left]
    return Sampler(Field(adapter, head), head, example, labels, grid, amounts, active,
                   heun=adapter.name == 'sit_small', graphs=graphs)
