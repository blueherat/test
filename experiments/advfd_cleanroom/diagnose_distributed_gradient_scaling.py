"""Numerically audit AdvFD's gradient reduction after differentiable gather.

Run with, for example::

    torchrun --standalone --nproc-per-node=2 \
        experiments/advfd_cleanroom/diagnose_distributed_gradient_scaling.py \
        --output /tmp/advfd_gradient_scaling.json

Each rank owns a disjoint feature chunk.  The official differentiable gather
exposes the global feature matrix to the FD loss, but its backward returns only
the local chunk's contribution.  This script compares summing and averaging
those parameter-gradient contributions with a centralized full-batch gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


OFFICIAL_ADVFD_ROOT = Path(
    "/data/users/zhoushunyu/research_repos/AdvFD"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    return parser.parse_args()


def _load_official_functions():
    sys.path.insert(0, str(OFFICIAL_ADVFD_ROOT))
    from frechet_distance.losses import (  # noqa: PLC0415
        compute_frechet_distance_loss,
        diff_all_gather,
    )
    from utils.distributed_util import all_reduce_grads  # noqa: PLC0415

    return compute_frechet_distance_loss, diff_all_gather, all_reduce_grads


def _local_inputs(rank: int, *, device: torch.device) -> torch.Tensor:
    chunks = (
        torch.tensor(
            [[-1.2, 0.5], [0.3, 1.7], [1.1, -0.4], [0.8, 0.2]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[-0.7, -1.1], [1.6, 0.9], [0.2, -0.8], [1.3, 1.4]],
            dtype=torch.float64,
        ),
    )
    if rank >= len(chunks):
        raise ValueError("This diagnostic is intentionally defined for two ranks")
    return chunks[rank].to(device)


def _features(parameter: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    # Both mean and covariance terms depend on the shared scalar parameter.
    offset = inputs.new_tensor([0.15, -0.25])
    return parameter * inputs + parameter.square() * offset


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Launch this diagnostic with exactly two processes")

    if args.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    compute_fd, diff_all_gather, all_reduce_grads = _load_official_functions()

    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64, device=device)
    with torch.no_grad():
        model.weight.fill_(1.3)
    parameter = model.weight.reshape(())
    local_inputs = _local_inputs(rank, device=device)
    local_features = _features(parameter, local_inputs)
    global_features = diff_all_gather(local_features)

    reference_mean = torch.tensor([0.2, -0.1], dtype=torch.float64, device=device)
    reference_covariance = torch.tensor(
        [[1.4, 0.25], [0.25, 0.9]], dtype=torch.float64, device=device
    )
    loss = compute_fd(
        reference_mean,
        reference_covariance,
        all_feats=global_features,
    )
    loss.backward()
    local_gradient = model.weight.grad.detach().clone()

    summed_gradient = local_gradient.clone()
    dist.all_reduce(summed_gradient, op=dist.ReduceOp.SUM)

    # Recreate the local gradient because the official helper mutates it.
    model.weight.grad.copy_(local_gradient)
    official_collectives = all_reduce_grads(model)
    official_average = model.weight.grad.detach().clone()

    all_inputs = torch.cat(
        [_local_inputs(i, device=device) for i in range(world_size)], dim=0
    )
    centralized_parameter = torch.tensor(
        1.3, dtype=torch.float64, device=device, requires_grad=True
    )
    centralized_features = _features(centralized_parameter, all_inputs)
    centralized_loss = compute_fd(
        reference_mean,
        reference_covariance,
        all_feats=centralized_features,
    )
    centralized_gradient = torch.autograd.grad(
        centralized_loss, centralized_parameter
    )[0]

    gathered_local = [torch.zeros_like(local_gradient) for _ in range(world_size)]
    dist.all_gather(gathered_local, local_gradient)
    if rank == 0:
        expected_average = centralized_gradient / world_size
        payload = {
            "world_size": world_size,
            "backend": args.backend,
            "loss_per_rank": float(loss.detach()),
            "centralized_loss": float(centralized_loss.detach()),
            "local_gradient_contributions": [
                float(value) for value in gathered_local
            ],
            "summed_gradient": float(summed_gradient),
            "centralized_gradient": float(centralized_gradient),
            "official_average_gradient": float(official_average),
            "expected_centralized_div_world_size": float(expected_average),
            "sum_vs_centralized_abs_error": float(
                (summed_gradient - centralized_gradient).abs()
            ),
            "official_vs_divided_abs_error": float(
                (official_average - expected_average).abs()
            ),
            "official_to_centralized_ratio": float(
                official_average / centralized_gradient
            ),
            "official_collective_calls": int(official_collectives),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
