### 1. 环境配置

```bash
conda env create -f environment.yaml
conda activate eqvae_train
pip install packaging==21.3
pip install 'torchmetrics<0.8'
pip install transformers==4.10.2
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install Pillow==9.5.0
```

### 2. 下载 SD-VAE

要从官方 LDM 仓库下载 SD-VAE，请运行：

```bash
bash download_sdvae.sh
```

### 3. 数据集

#### 数据集下载

目前我们提供的是 [OpenImages](https://storage.googleapis.com/openimages/web/index.html) 实验。下载完成后，请在[配置文件](configs/eqvae_config.yaml)中修改 `train_dir`、`val_dir` 和 `dataset_name` 路径/字段。

### 4. 训练

在 8 张 GPU 上运行 EQ-VAE 正则化训练：

```bash
python main.py \
    --base configs/eqvae_config.yaml \
    -t \
    --gpus 0,1,2,3,4,5,6,7 \
    --resume pretrained_models/model.ckpt \
    --logdir logs/eq-vae
```

随后该脚本会自动在 `logs/eq-vae` 下创建文件夹，用于保存日志和检查点。
`configs/eqvae_config.yaml` 中提供的参数就是论文中使用的设置。你可以根据自己的实验调整以下选项：

- `anisotropic`：若为 `True`，执行各向异性缩放。
- `uniform_sample_scale`：若为 `True`，从 `[0.25, 1)` 中均匀采样缩放因子；若为 `False`，则从 `{0.25, 0.5, 0.75}` 中随机选择缩放因子。
- `p_prior`：执行先验保持而非等变性正则化的概率。
- `p_prior_s`：在较低分辨率上执行先验保持而非等变性正则化的概率。
