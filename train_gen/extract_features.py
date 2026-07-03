"""
代码改编自 https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../train_eqvae'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from models.dit_models import DiT_models
from models.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm

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



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    让 EMA 模型向当前模型更新一步。
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO：考虑只应用到 require_grad 的参数，避免 pos_embed 产生微小数值变化。
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    为模型中的所有参数设置 requires_grad 标志。
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    结束 DDP 训练。
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    创建同时写入日志文件和标准输出的 logger。
    """
    if dist.get_rank() == 0:  # 真实 logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # 空 logger（不执行任何操作）
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    来自 ADM 的中心裁剪实现。
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                   训练循环                                    #
#################################################################################

def main(args):
    """
    提取用于训练 DiT 模型的潜特征。
    """
    assert torch.cuda.is_available(), "当前特征提取至少需要一张 GPU。"

    # 设置 DDP：
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "批大小必须能被 world size 整除。"
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"启动 rank={rank}，seed={seed}，world_size={dist.get_world_size()}。")

    # 设置特征文件夹：
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_features'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_labels'), exist_ok=True)

    # 创建模型：
    assert args.image_size % 8 == 0, "图像尺寸必须能被 8 整除（供 VAE 编码器使用）。"
    latent_size = args.image_size // 8

    if args.vae_ckpt is not None:
        vae_config = load_config(args.vae_config, display=False)
        vae = load_kl(vae_config, ckpt_path=args.vae_ckpt).to(device)
    else:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.hf_model_name).to(device)


    # 设置数据：
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():

            if args.vae_ckpt is not None:

                x  = vae.encode(x).sample().mul_(args.vae_scaling_factor)

            else:
                x = vae.encode(x).latent_dist.sample().mul_(args.vae_scaling_factor)

        x = x.detach().cpu().numpy()    # (1, 4, 32, 32)
        np.save(f'{args.features_path}/imagenet256_features/{rank}-{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.features_path}/imagenet256_labels/{rank}-{train_steps}.npy', y)
            
        train_steps += 1

if __name__ == "__main__":
    # 这里的默认参数与论文中训练 DiT-XL/2 的超参数保持一致（训练迭代次数除外）。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse", "xl", "sd3","ours"], default="ema") 
    parser.add_argument("--vae-ckpt", type=str, default=None) 
    parser.add_argument("--vae-config", type=str, default=None)  
    parser.add_argument("--hf-model-name", type=str, default="zelaki/eq-vae")  
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)  
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
