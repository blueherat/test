"""
代码改编自 https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
# 测试该脚本时下面第一个标志为 False，但设为 True 可显著加速 A100 训练：
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
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
from accelerate import Accelerator
from models.dit_models import DiT_models
from models.diffusion import create_diffusion

# from models import DiT_models
# from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                                 训练辅助函数                                  #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    让 EMA 模型向当前模型更新一步。
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO：考虑只应用到 require_grad 的参数，避免 pos_embed 产生微小数值变化。
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    为模型中的所有参数设置 requires_grad 标志。
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    创建同时写入日志文件和标准输出的 logger。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
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


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "特征文件和标签文件数量应相同"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                   训练循环                                    #
#################################################################################

def main(args):
    """
    训练新的 DiT 模型。
    """
    assert torch.cuda.is_available(), "当前训练至少需要一张 GPU。"

    # 设置 accelerator：
    accelerator = Accelerator()
    device = accelerator.device

    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # 例如 DiT-XL/2 --> DiT-XL-2（用于文件夹命名）

    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # 创建实验文件夹


    if args.dataset_name:
        experiment_dir += f"-{args.dataset_name}"

    if args.vae_name:
        experiment_dir += f"-{args.vae_name}"


    # 设置实验文件夹：
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # 创建结果文件夹（包含所有实验子文件夹）
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # 保存模型检查点
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"实验目录已创建：{experiment_dir}")

    # 创建模型：
    assert args.image_size % 8 == 0, "图像尺寸必须能被 8 整除（供 VAE 编码器使用）。"
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=args.in_channels
    )
    # 参数初始化在 DiT 构造函数内部完成
    model = model.to(device)
    ema = deepcopy(model).to(device)  # 创建模型的 EMA 副本，用于训练后使用


    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]


    requires_grad(ema, False)


    diffusion = create_diffusion(timestep_respacing="")  # 默认：1000 步，线性噪声调度
    if accelerator.is_main_process:
        logger.info(f"DiT 参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 设置优化器（论文中使用默认 Adam betas=(0.9, 0.999) 和 1e-4 常数学习率）：

    # 设置数据：
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"数据集包含 {len(dataset):,} 张图像（{args.feature_path}）")

    # 准备模型训练：
    update_ema(ema, model, decay=0)  # 确保 EMA 使用同步后的权重初始化
    model.train()  # 重要：启用用于 classifier-free guidance 的 embedding dropout
    ema.eval()  # EMA 模型应始终处于 eval 模式
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # 监控/日志变量：
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"开始训练 {args.epochs} 个 epoch...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"开始 epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # 记录损失值：
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # 测量训练速度：
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # 在所有进程上规约损失历史：
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) 训练损失：{avg_loss:.4f}，训练步/秒：{steps_per_sec:.2f}")
                # 重置监控变量：
                running_loss = 0
                log_steps = 0
                start_time = time()

            # 保存 DiT 检查点：
            if train_steps % args.ckpt_every == 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"检查点已保存到 {checkpoint_path}")
            if train_steps > args.max_train_steps:
                break
    
    if accelerator.is_main_process:
        logger.info("完成！")


if __name__ == "__main__":
    # 这里的默认参数会用论文中的超参数训练 DiT-XL/2（训练迭代次数除外）。
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset-name", type=str, default=False)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--max-train-steps", type=str, default=400_000)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae-name", type=str, default="ema")  # 该选项不影响训练
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    args = parser.parse_args()
    main(args)
