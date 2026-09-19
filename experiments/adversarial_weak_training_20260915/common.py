import os
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.distributed as dist
from experiments.guidance_dynamic_50k_20260915 import config as original
from experiments.guidance_loss_50k_20260914.config import atomic, read, sha

WORK = Path(__file__).resolve().parents[2]
ROOT = original.EXPS / "adversarial_weak_training_20260915"
PYTHON = original.PYTHON


def now():
    return datetime.now(timezone.utc).isoformat()


def setup():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, world


def barrier():
    if dist.is_initialized():
        dist.barrier()


def average_gradients(module):
    if dist.is_initialized():
        for parameter in module.parameters():
            if parameter.grad is None:
                raise RuntimeError("A trainable parameter did not receive a gradient")
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())


def mean_values(values):
    value = torch.tensor(values, dtype=torch.float64, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(value)
        value /= dist.get_world_size()
    return value.cpu().tolist()


def stopped():
    return (ROOT / "STOP_AFTER_CURRENT").exists()


def source_receipts():
    dependencies = [*Path(__file__).parent.glob("*.py"),
                    *(WORK / "experiments/adversarial_guidance_endpoint_20260915").glob("*.py"),
                    *(WORK / "experiments/guidance_dynamic_50k_20260915").glob("*.py")]
    dependencies += [WORK / path for path in (
        "experiments/advfd_cleanroom/core.py",
        "experiments/advfd_cleanroom/feature_extractors.py",
        "experiments/guidance_distribution_20260912/local_head.py",
        "experiments/guidance_loss_50k_20260914/components.py",
        "experiments/guidance_loss_50k_20260914/config.py",
        "experiments/guidance_pasted_20260912/common.py",
        "experiments/lifting_scale_sweep_20260909.py",
        "experiments/train_imagenet100_sit_flow.py",
        "experiments/sample_imagenet100_sit_fid.py",
        "experiments/raev2_training_core.py",
        "experiments/compute_adm_fid.py",
        "experiments/guidance_strength_sweep_20260915/planning.py",
        "experiments/weak_reference_loss_20260914/idle.py",
    )]
    return {str(path.resolve()): sha(path) for path in sorted(set(dependencies))}
