"""计算用于评估 GAN 的 Frechet Inception Distance（FID）。

FID 指标用于计算两个图像分布之间的距离。通常，其中一个分布由汇总统计量
（均值和协方差矩阵）表示，另一个分布由 GAN 给出。

作为独立程序运行时，它会比较指定位置中 PNG/JPEG 图像的分布与汇总统计量
（pickle 格式）给出的分布。

FID 的计算假设 X_1 和 X_2 分别是生成样本与真实样本在 Inception 网络
pool_3 层上的激活。

更多细节请查看 --help。

代码改编自 https://github.com/bioinf-jku/TTUR，改为使用 PyTorch 而非
Tensorflow。

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
    from tqdm import tqdm
except ImportError:
    # 如果 tqdm 不可用，提供一个模拟版本
    def tqdm(x):
        return x


FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501


class InceptionV3(nn.Module):
    """返回特征图的预训练 InceptionV3 网络"""

    # 默认返回的 Inception block 索引，对应最终平均池化的输出
    DEFAULT_BLOCK_INDEX = 3

    # 将特征维度映射到对应的输出 block 索引
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # 第一次最大池化特征
        192: 1,  # 第二次最大池化特征
        768: 2,  # 辅助分类器之前的特征
        2048: 3,  # 最终平均池化特征
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        """构建预训练 InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            要返回特征的 block 索引。可选值为：
                - 0：对应第一次最大池化的输出
                - 1：对应第二次最大池化的输出
                - 2：对应输入辅助分类器的输出
                - 3：对应最终平均池化的输出
        resize_input : bool
            若为 true，在将输入送入模型前，用双线性插值将宽高缩放到 299。
            由于去掉全连接层后的网络是全卷积结构，它应能处理任意尺寸输入，
            因此严格来说不一定需要缩放。
        normalize_input : bool
            若为 true，将输入从 (0, 1) 范围缩放到预训练 Inception 网络期望的
            (-1, 1) 范围。
        requires_grad : bool
            若为 true，模型参数需要梯度。这在微调网络时可能有用。
        use_fid_inception : bool
            若为 true，使用 Tensorflow FID 实现所用的预训练 Inception 模型。
            若为 false，使用 torchvision 中可用的预训练 Inception 模型。
            FID Inception 模型与 torchvision 的 Inception 模型权重不同，
            结构也略有差异。若要计算 FID 分数，强烈建议将该参数设为 true，
            以得到可比较的结果。
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "最大可用输出 block 索引为 3"

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights="DEFAULT")

        # Block 0：输入到 maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1：maxpool1 到 maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2：maxpool2 到辅助分类器
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3：辅助分类器到最终平均池化
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """获取 Inception 特征图

        Parameters
        ----------
        inp : torch.autograd.Variable
            形状为 Bx3xHxW 的输入张量，取值应位于 (0, 1) 范围。

        Returns
        -------
        torch.autograd.Variable 列表，对应所选输出 block，并按索引升序排列。
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # 从 (0, 1) 范围缩放到 (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """包装 `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        # 防范异常版本字符串
        version = (0,)

    # 如果 torchvision 版本支持，则跳过默认权重初始化。
    # 参见 https://github.com/mseitzer/pytorch-fid/issues/28。
    if version >= (0, 6):
        kwargs["init_weights"] = False

    # 向后兼容：0.13 之前用 `pretrained` 参数处理 `weights` 参数。
    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                "torchvision {} 不支持 weights=={}".format(
                    torchvision.__version__, kwargs["weights"]
                )
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """构建用于 FID 计算的预训练 Inception 模型

    用于 FID 计算的 Inception 模型使用另一组权重，结构也与 torchvision 的
    Inception 略有不同。

    本方法会先构造 torchvision 的 Inception，然后修补 FID Inception 模型中
    与之不同的必要部分。
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    # state_dict = torch.load('/path/to/your/pt_inception-2015-12-05-6726825d.pth') # <--- 下载该文件，或使用默认下载函数
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """为 FID 计算修补过的 InceptionA block"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # 修补：Tensorflow 的平均池化在求平均时不会使用 padding 出来的 0
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """为 FID 计算修补过的 InceptionC block"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # 修补：Tensorflow 的平均池化在求平均时不会使用 padding 出来的 0
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """为 FID 计算修补过的第一个 InceptionE block"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # 修补：Tensorflow 的平均池化在求平均时不会使用 padding 出来的 0
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """为 FID 计算修补过的第二个 InceptionE block"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # 修补：FID Inception 模型在这里使用最大池化而不是平均池化。
        # 这很可能是该特定 Inception 实现中的错误，因为其他 Inception 模型
        # 在这里使用平均池化（也与论文描述一致）。
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=50, help="使用的批大小")
parser.add_argument(
    "--num-workers",
    type=int,
    help=(
        "用于数据加载的进程数。默认值为 `min(8, num_cpus)`"
    ),
)
parser.add_argument(
    "--device", type=str, default=None, help="使用的设备，例如 cuda、cuda:0 或 cpu"
)
parser.add_argument(
    "--dims",
    type=int,
    default=2048,
    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    help=(
        "要使用的 Inception 特征维度。"
        "默认使用 pool3 特征"
    ),
)
parser.add_argument(
    "--save-stats",
    action="store_true",
    help=(
        "从样本目录生成 npz 归档。"
        "第一个路径作为输入，第二个路径作为输出。"
    ),
)
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("生成图像路径或 .npz 统计文件路径"),
)

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, sp_len=None
):
    """计算所有图像在 pool_3 层上的激活。

    参数：
    -- files       : 图像文件路径列表
    -- model       : Inception 模型实例
    -- batch_size  : 模型一次处理的图像批大小。请确保样本数是批大小的倍数，
                     否则部分样本会被忽略。保留此行为是为了匹配原始 FID
                     分数实现。
    -- dims        : Inception 返回的特征维度
    -- device      : 执行计算的设备
    -- num_workers : 并行 dataloader worker 数量

    返回：
    -- 形状为 (num images, dims) 的 numpy 数组，包含将查询张量输入
       Inception 后得到的指定张量激活。
    """
    model.eval()

    if batch_size > len(files):
        print(
            (
                "警告：批大小大于数据量。"
                "将批大小设为数据量"
            )
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        if type(model).__name__ == 'InceptionV3':
            with torch.no_grad():
                pred = model(batch)[0]


            # 如果模型输出不是标量，则应用全局空间平均池化。
            # 当选择的维度不等于 2048 时会发生这种情况。
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        else:
            with torch.no_grad():
                pred = model(batch)
            pred = pred.cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet 距离的 Numpy 实现。
    两个多元高斯 X_1 ~ N(mu_1, C_1) 和 X_2 ~ N(mu_2, C_2) 之间的
    Frechet 距离为
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    稳定版本来自 Dougal J. Sutherland。

    参数：
    -- mu1   : 生成样本在 Inception 网络某一层上的激活均值。
    -- mu2   : 在代表性数据集上预先计算得到的激活样本均值。
    -- sigma1: 生成样本激活的协方差矩阵。
    -- sigma2: 在代表性数据集上预先计算得到的激活协方差矩阵。

    返回：
    --   : Frechet 距离。
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "训练集与测试集均值向量长度不同"
    assert (
        sigma1.shape == sigma2.shape
    ), "训练集与测试集协方差矩阵维度不同"

    diff = mu1 - mu2

    # 矩阵乘积可能近似奇异
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "FID 计算产生奇异乘积；"
            "向协方差估计的对角线添加 %s"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 数值误差可能产生很小的虚部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("虚部 {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, sp_len=None
):
    """计算 FID 使用的统计量。
    参数：
    -- files       : 图像文件路径列表
    -- model       : Inception 模型实例
    -- batch_size  : 图像 numpy 数组会按 batch_size 拆分成批次；合理的批大小取决于硬件。
    -- dims        : Inception 返回的特征维度
    -- device      : 执行计算的设备
    -- num_workers : 并行 dataloader worker 数量

    返回：
    -- mu    : Inception 模型 pool_3 层激活在样本上的均值。
    -- sigma : Inception 模型 pool_3 层激活的协方差矩阵。
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers, sp_len)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1, sp_len=None):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )

        if sp_len is not None:
            files = files[:sp_len]
        m, s = calculate_activation_statistics(
            files, model, batch_size, dims, device, num_workers
        )

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1, model_name="inception_v3", sp_len=None):
    """计算两个路径之间的 FID"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("无效路径：%s" % p)
        
    if model_name == "inception_v3":
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
    else:
        raise NotImplementedError(f"模型 {model_name} 尚未实现")

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers, sp_len
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers, sp_len
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # Windows 下没有 os.sched_getaffinity，改用 os.cpu_count
            # （它可能不会返回“可用”的 CPU 数量）。
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    fid_value = calculate_fid_given_paths(
        args.path, args.batch_size, device, args.dims, num_workers
    )
    print("FID: ", fid_value)
