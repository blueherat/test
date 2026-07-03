import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image


def load_image_tensor(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    返回 NCHW 格式、取值范围为 [-1, 1] 的图像张量。
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # 使尺寸能被 8 整除，以适配常见 VAE 配置
    w = (w // 8) * 8
    h = (h // 8) * 8
    if (w, h) != img.size:
        img = img.resize((w, h), resample=Image.BICUBIC)

    x = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(h, w, 3).numpy())
    x = x.to(device=device, dtype=dtype) / 255.0  # [0,1] 范围内的 HWC
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1CHW
    x = x * 2.0 - 1.0  # [-1,1]
    return x


@torch.no_grad()
def encode_latents(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    vae.eval()
    posterior = vae.encode(x).latent_dist
    latents = posterior.sample()
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    return latents * scaling


def pca_to_rgb(latents: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    通过在通道维上做 PCA 投影，将形状为 [B, C, H, W] 的潜变量张量转换为
    [B, 3, H, W] 的 RGB 张量。
    """
    B, C, H, W = latents.shape
    X = latents.permute(0, 2, 3, 1).reshape(-1, C).to(torch.float32)

    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top3_eigenvectors = eigenvectors[:, -3:]  # 形状：[C, 3]

    X_projected = X_centered @ top3_eigenvectors  # 形状：[B * H * W, 3]

    rgb = X_projected.reshape(B, H, W, 3).permute(0, 3, 1, 2)


    rgb_min = rgb.reshape(B, -1).min(dim=1)[0].reshape(B, 1, 1, 1)
    rgb_max = rgb.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1, 1)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + eps)

    return rgb_norm


def save_rgb_tensor_as_image(rgb: torch.Tensor, path: str, size: int = 256):
    """
    将 [0, 1] 范围内的 [B, 3, H, W] 张量保存为图像文件。
    """
    # 转到 CPU、转换为 uint8，并取第一个 batch 样本
    rgb_np = rgb[0].detach().cpu().clamp(0, 1)
    rgb_np = (rgb_np * 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(rgb_np, mode="RGB")
    if size is not None:
        img = img.resize((size, size), resample=Image.BILINEAR)
    img.save(path)

def first3_to_rgb(latents: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    取前 3 个通道，并对每张图像做 min-max 归一化到 [0, 1]，
    将 [B, C, H, W] 潜变量转换为 [B, 3, H, W] 的 RGB 张量。
    """
    if latents.ndim != 4:
        raise ValueError(f"期望潜变量形状为 [B,C,H,W]，实际得到 {tuple(latents.shape)}")
    rgb = latents[:, :3].to(torch.float32)
    B = rgb.shape[0]
    rgb_min = rgb.reshape(B, -1).min(dim=1)[0].reshape(B, 1, 1, 1)
    rgb_max = rgb.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1, 1)
    return (rgb - rgb_min) / (rgb_max - rgb_min + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="pca_vis", help="保存输出图像的目录（默认：与输入图像所在目录相同）")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    eqvae = AutoencoderKL.from_pretrained("zelaki/eq-vae").to(device=device, dtype=dtype)

    x = load_image_tensor(args.image, device=device, dtype=dtype)

    eq_latents = encode_latents(eqvae, x)

    print(f"图像：{args.image}")
    print(f"输入张量：shape={tuple(x.shape)} dtype={x.dtype} device={x.device} range=[{x.min().item():.3f},{x.max().item():.3f}]")
    print(f"eq-vae 潜变量：shape={tuple(eq_latents.shape)} dtype={eq_latents.dtype} device={eq_latents.device}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.image).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 将潜变量转换为 RGB 并保存
    eq_rgb = pca_to_rgb(eq_latents)
    eq_first3 = first3_to_rgb(eq_latents)
    
    input_stem = Path(args.image).stem
    eq_path = output_dir / f"{input_stem}_eq_vae_pca.png"
    eq_first3_path = output_dir / f"{input_stem}_eq_vae_first3.png"
    
    save_rgb_tensor_as_image(eq_rgb, str(eq_path))
    save_rgb_tensor_as_image(eq_first3, str(eq_first3_path))


if __name__ == "__main__":
    main()
