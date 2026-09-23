"""CPU float64 audit of the actual replay backward at current signed scales.

Uses a small nonlinear surrogate, not JiT/SiT weights or CUDA graphs. The
comparison covers the current coefficient range, clean-to-velocity floor,
mixed Heun/Euler and output clamping. A deliberate stop-gradient is a
negative control for the sensitivity of the audit.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import torch

from classifier_guidance.sampler import Sampler
from classifier_guidance.schedules import GuidanceSchedule

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/self_guidance_schedule_shape_20260923"
CSV = Path("/home/zhoushunyu/data/eqvae/projects/classifier_guidance/"
           "jit_block1_gan_schedule_20260922/coefficients.csv")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260923)
    raw = CSV.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    coefficients = torch.tensor([float(r["raw_a"]) for r in rows], dtype=torch.float64)
    assert len(coefficients) == 50
    assert len({r["training_step"] for r in rows}) == 1
    schedule = GuidanceSchedule(50).double().eval()
    with torch.no_grad():
        schedule.coefficients.copy_(coefficients)
    grid = torch.linspace(0, 1, 51, dtype=torch.float64)
    indices = torch.arange(50, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    noise = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(5, 5, dtype=torch.float64) / 3
    weak_matrix = torch.randn(5, 5, dtype=torch.float64) / 3
    conditioning = torch.tensor([.3, -.2, .5, -.6, .9], dtype=torch.float64)
    weights = torch.tensor([.2, -.3, .7, -.5, .4], dtype=torch.float64)

    def field_at(x, t, y, coefficient):
        cond = (y[:, None] + 1) * conditioning
        strong = 1.4 * torch.tanh(.35 * x @ matrix + t * cond + .2)
        weak = 1.25 * torch.tanh(.25 * x @ weak_matrix + .4 * t * cond - .1)
        clean = weak + (1 + coefficient) * (strong - weak)
        return (clean - x) / (1 - t).clamp_min(.05)

    def field(x, t, y, index, active):
        assert active
        return field_at(x, t, y, schedule(index))

    def reference(x, coeff, flags, *, faulty_predictor_detach=False):
        for i, heun in enumerate(flags):
            h = grid[i+1] - grid[i]
            first = field_at(x, grid[i], labels, coeff[i])
            predicted = x + h * first
            if heun:
                if faulty_predictor_detach:
                    predicted = predicted.detach()
                second = field_at(predicted, grid[i+1], labels, coeff[i])
                x = x + h/2 * (first + second)
            else:
                x = predicted
        return x

    def loss(x):
        image = ((x + 1) / 2).clamp(0, 1)
        return (weights * torch.sin(1.7 * image) + .07 * image.square()).mean()

    cases = []
    for name, flags in [("49_heun_final_euler", [True]*49+[False]),
                        ("50_heun", [True]*50)]:
        engine = Sampler(field, schedule, noise, labels, grid, indices,
                         [True]*50, heun=flags, graphs=False)
        expected = reference(noise, schedule.coefficients, flags)
        actual = engine(noise, labels)
        ref_grad = torch.autograd.grad(loss(expected), (noise, schedule.coefficients))
        replay_grad = torch.autograd.grad(loss(actual), (noise, schedule.coefficients))
        rel = [float((x-y).norm() / y.norm()) for x, y in zip(replay_grad, ref_grad)]
        assert torch.equal(actual, expected)
        assert max(rel) < 1e-11
        fd = []
        for index in [0, 25, 38, 44, 45, 46, 47, 48, 49]:
            for epsilon in [1e-4, 1e-5]:
                plus, minus = coefficients.clone(), coefficients.clone()
                plus[index] += epsilon
                minus[index] -= epsilon
                with torch.no_grad():
                    finite = float((loss(reference(noise, plus, flags)) -
                                    loss(reference(noise, minus, flags))) / (2*epsilon))
                derivative = float(replay_grad[1][index])
                error = abs(finite-derivative)
                assert error < 1e-8 + 1e-5*abs(derivative)
                fd.append(dict(index=index, epsilon=epsilon, gradient=derivative,
                               finite_difference=finite, absolute_error=error,
                               relative_error=error/max(abs(derivative), 1e-12)))

        faulty = reference(noise, schedule.coefficients, flags, faulty_predictor_detach=True)
        faulty_grad, = torch.autograd.grad(loss(faulty), (schedule.coefficients,))
        faulty_relative = float((faulty_grad-ref_grad[1]).norm()/ref_grad[1].norm())
        assert torch.equal(faulty, expected)
        assert faulty_relative > 1e-2
        cases.append(dict(
            name=name, endpoint_max_error=float((actual-expected).detach().abs().max()),
            input_gradient_relative_error=rel[0], coefficient_gradient_relative_error=rel[1],
            output_clamped_fraction=float(((expected < -1) | (expected > 1)).double().mean()),
            tail_gradient_actual=replay_grad[1][-10:].tolist(),
            tail_gradient_reference=ref_grad[1][-10:].tolist(),
            finite_differences=fd,
            deliberate_predictor_detach=dict(
                same_forward=True, gradient_relative_error=faulty_relative,
                tail_gradient=faulty_grad[-10:].tolist()),
        ))
    result = dict(
        kind="Actual Sampler graphs=False on a nonlinear CPU float64 surrogate",
        limitation="No real JiT/SiT weights, BF16/TF32, CUDA graph or real GAN critic validation.",
        coefficient_source=str(CSV), coefficient_source_sha256=hashlib.sha256(raw).hexdigest(),
        coefficient_training_step=int(rows[0]["training_step"]),
        coefficients=coefficients.tolist(), cases=cases, passed=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_tail_backward_cpu_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=True, coefficient_training_step=result["coefficient_training_step"],
                         cases=[{k:v for k,v in c.items() if k not in
                                 ("finite_differences", "tail_gradient_actual",
                                  "tail_gradient_reference")} for c in cases]), indent=2))


if __name__ == "__main__":
    main()
