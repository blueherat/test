
"""
代码改编自 https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../train_eqvae'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


import torch
import torch.distributed as dist
from download import find_model
from models.dit_models import DiT_models
from models.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from evaluator import Evaluator
import tensorflow.compat.v1 as tf
from train_eqvae.ldm.models.autoencoder import AutoencoderKL as LDMAutoencoderKL

from ldm.util import instantiate_from_config
import yaml
from omegaconf import OmegaConf

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_kl(config, ckpt_path=None):

    model = LDMAutoencoderKL(**config.model.params)

    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_kl(x):
    x = 2.*x - 1.
    return x



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    从 .png 样本文件夹构建单个 .npz 文件。
    """
    samples = []
    for i in tqdm(range(num), desc="正在从样本构建 .npz 文件"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f".npz 文件已保存到 {npz_path} [shape={samples.shape}]。")
    return npz_path



def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def calculate_metrics(ref_batch, sample_batch, fid_path):
    config = tf.ConfigProto(
        allow_soft_placement=True  
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    evaluator.warmup()

    ref_acts = evaluator.read_activations(ref_batch)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch, ref_acts)

    sample_acts = evaluator.read_activations(sample_batch)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_batch, sample_acts)

    with open(fid_path, 'w') as fd:

        fd.write("正在计算评估指标...\n")
        fd.write(f"Inception Score:{evaluator.compute_inception_score(sample_acts[0])}\n" )
        fd.write(f"FID:{sample_stats.frechet_distance(ref_stats)}\n")
        fd.write(f"sFID:{sample_stats_spatial.frechet_distance(ref_stats_spatial)}\n")
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        fd.write(f"Precision:{prec}\n")
        fd.write(f"Recall:{recall}\n")



def main(args):
    """
    运行采样。
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True：速度更快，但可能带来少量数值差异
    assert torch.cuda.is_available(), "使用 DDP 采样至少需要一张 GPU。sample.py 支持仅 CPU 使用。"
    torch.set_grad_enabled(False)

    # 设置 DDP：
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"启动 rank={rank}，seed={seed}，world_size={dist.get_world_size()}。")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "只有 DiT-XL/2 模型支持自动下载。"
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # 加载模型：
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels
    ).to(device)
    # 自动下载预训练模型，或加载 train.py 生成的自定义 DiT 检查点：
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # 重要！
    diffusion = create_diffusion(str(args.num_sampling_steps))

    if args.vae_ckpt is not None:
        vae_config = load_config(args.vae_config, display=False)
        vae = load_kl(vae_config, ckpt_path=args.vae_ckpt).to(device)
    else:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.hf_model_name).to(device)


    assert args.cfg_scale >= 1.0, "几乎所有情况下，cfg_scale 都应 >= 1.0。"
    using_cfg = args.cfg_scale > 1.0

    # 创建保存样本的文件夹：
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"

    if args.ddpm:
        folder_name += f"-ddpm"


    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    exp_dir_name = os.path.dirname(args.sample_dir)
    fid_dir =  f"{exp_dir_name}/fids"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(fid_dir, exist_ok=True)

        print(f".png 样本将保存到 {sample_folder_dir}")
        print(f"指标将保存到 {fid_dir}")

    dist.barrier()

    fid_path = f"{fid_dir}/metrics-adm-{folder_name}.txt"

    # 计算每张 GPU 需要生成多少样本，以及需要运行多少次迭代：
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # 为了整除，会略多采样一些，再丢弃多余样本：
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"将采样的图像总数：{total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples 必须能被 world_size 整除。"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu 必须能被单 GPU 批大小整除。"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # 采样输入：
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # 设置 classifier-free guidance：
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # 采样图像：
        if args.ddpm:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        else:
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # 移除空类别样本

        if args.vae_ckpt is not None:

            samples = vae.decode(samples / args.vae_scaling_factor)

        else:

            samples = vae.decode(samples / args.vae_scaling_factor).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # 将样本作为单独的 .png 文件保存到磁盘
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total

            if args.vae in ["ours", "hf"]:
                sample = custom_to_pil(sample)
                sample.save(f"{sample_folder_dir}/{index:06d}.png")
            else:
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # 确保所有进程保存完样本后，再转换为 .npz
    dist.barrier()
    if rank == 0:
        sample_batch = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        calculate_metrics(sample_batch, args.ref_batch, fid_path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae-ckpt",  type=str,  default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--ddpm", type=bool, default=False)
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--ref-batch", type=str, default="/data/imagenet/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--wavelets", type=str, default=False)
    parser.add_argument("--diff-proj", type=str, default=False)
    parser.add_argument("--gaussian-registers", type=str, default=False)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="默认使用 TF32 矩阵乘法，可在 Ampere GPU 上大幅加速采样。")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="可选的 DiT 检查点路径（默认自动下载预训练 DiT-XL/2 模型）。")
    args = parser.parse_args()
    main(args)
