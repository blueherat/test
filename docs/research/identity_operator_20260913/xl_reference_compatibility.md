# SiT-XL 公共反演参考：路径、类别、VAE 与 wrapper 审计

2026-09-13。结论：**现有官方 SiT-XL/2 + IG 的 full conditional 场可作为 SiT-S copy pilot 的独立训练来源参考；必须同时反转时间与速度符号。** 不是 oracle，也不是数据独立的模型：双方都使用 ImageNet。此次审计只读源码、CPU mmap checkpoint 元信息、CPU 比较 VAE 权重和读取已冻结 request；未采样、未初始化 CUDA、未修改运行源码。

## 1. 资产与训练身份

- XL checkpoint：`/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt`，解析到 `/data/users/zhoushunyu/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt`。文件 **10,842,729,224 bytes**；CPU mmap 读取到 `epoch=799`、`steps=3984208`，EMA state 有 296 个 tensor，28 个 blocks，label embedding 为 `(1001,1152)`。
- checkpoint args 明确 `model='SiT-XL/2'`、`encoder_depth=8`、`num_classes=1000`、`path_type='linear'`、`prediction='v'`、`weighting='uniform'`。它不是 small 同一次训练的后续 checkpoint。
- 原仓库为 `/home/zhoushunyu/eqvae/research_repos/internal_guidance_study/Internal-Guidance`，当前 commit `d048cc3e132001636dd35114720c4754295d1e52`。[官方本地 README](../../../research_repos/internal_guidance_study/Internal-Guidance/README.md)给出 SiT-XL/2、depth 8、linear/v 配置及 800 epoch 发布权重入口；其 FID-50K 是原协议结果，不能挪作此纯 conditional reference 的质量指标。
- [load_model](../../../experiments/run_internal_guidance_sit_audit.py:141) 使用 `state_key='ema'`、`input_size=32`、`num_classes=1000`、`use_cfg=True`、`fused_attn=False`、`qk_norm=False`，严格加载后设 FP32 / eval / 不计算参数梯度。
- 本次 root 运行生成的 `.../fm_common_inverse_copy_20260913_xl/ref_xl/models.json` 记录参数数 **677,786,144**，checkpoint SHA256 **`a7f4eb9f417295a14e5063b70020ab3f55b60a4905ccda7b244343deaf9840cd`**。该 hash 引用此次 runtime 实际记录；审计者未额外重读完整 10.8 GB 文件计算第二次 hash。

## 2. FM 路径及 label 对齐

由 [XL loss](../../../research_repos/internal_guidance_study/Internal-Guidance/SiT/loss.py:57) 与 [small 训练路径](../../../experiments/train_imagenet100_sit_flow.py:274)：

\[
z^{\rm XL}_\tau=(1-\tau)x+\tau\epsilon,\quad v^{\rm XL}_{\rm target}=\epsilon-x;
\qquad
z^{\rm S}_t=(1-t)\epsilon+tx,\quad v^{\rm S}_{\rm target}=x-\epsilon.
\]

所以令 `tau=1-t` 后，路径坐标一致，统一到 small 时间的速度必须为

\[
v_R(z,t,c)=-v^{\rm XL}_{\rm full}(z,1-t,m(c)).
\]

模型输入时间仍在 `[0,1]`，不额外乘 1000。统一坐标下，图像反演为 `t:1→0`，每个阶段重新查询当前状态对应的 reference 场。[Runtime](../../../experiments/lifting_scale_sweep_20260909.py:100) 原生 XL grid 是 `1→0`，small 是 `0→1`；不能直接共用其原生时间而忘记上述转换。

类别 manifest：`/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc/manifest.json`。SHA256 为 `6de2f588d8ff10e7c7ecf1ab818aae29bc5f15a6c85e04968e06a22934c796ad`。按 `classes[].label` 排序取 `original_imagenet_label`；实际检查 0..99 完整、100 个目标唯一且全部在 0..999。例如 `0→15`、`13→151`、`99→994`。XL null 是 **1000**，small null 是 **100**；本次 reference 不查询 null。

最小接口如下，仅作为后续独立 worker 的复用说明，本审计没有执行 GPU 部分：

```python
from pathlib import Path
import json
import torch
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.lifting_scale_sweep_20260909 import XL_REPO, XL_CKPT

device = torch.device('cuda:0')
model, meta = load_model(repo=XL_REPO, checkpoint_path=XL_CKPT,
    model_name='SiT-XL/2', encoder_depth=8, state_key='ema', device=device)
manifest = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc/manifest.json')
rows = sorted(json.loads(manifest.read_text())['classes'], key=lambda r: r['label'])
assert [r['label'] for r in rows] == list(range(100))
lookup = torch.tensor([r['original_imagenet_label'] for r in rows], device=device)

@torch.inference_mode()
def reference_velocity(z, t, labels100):
    native_t = torch.full((len(z),), 1.0-float(t), device=z.device, dtype=torch.float32)
    with torch.autocast('cuda', enabled=False):
        full, _, _ = model(z.float(), native_t, lookup[labels100.long()])
    return -full.to(z.dtype)
```

返回三项中的第二项是 depth-8 **IG weak head，不是 null 分支**。取第一项表示纯 conditional、无额外 CFG/IG；记录中的 `reference_guidance=0` 是附加强度为零，即常见总 CFG `w=1`。full forward 仍计算 auxiliary head，但不使用它改变 reference 场。

## 3. VAE：108 个 encoder tensor 的实际比较

两者均使用 `4×32×32` SD-VAE latent、比例 `0.18215`、零 bias。small 的缓存 manifest 为 `/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae/manifest.json`，明确 VAE 为 MSE、前 4 通道为 posterior mean、后 4 通道为 std。

实际缓存文件：

```text
/home/zhoushunyu/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493/diffusion_pytorch_model.safetensors
/home/zhoushunyu/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a/diffusion_pytorch_model.safetensors
```

每份文件 334,643,276 bytes；相邻 `config.json` 均存在。用 `safetensors.safe_open(..., framework='pt', device='cpu')` 打开二者，先验证 `encoder.*` 与 `quant_conv.*` key 集合相同，再逐 key `torch.equal(a.get_tensor(k), b.get_tensor(k))`：**108 个 tensor，0 个不同**，比较后 `torch.cuda.is_initialized()==False`。这验证的是相同输入预处理下编码分布的参数一致，不是对完整 VAE 输出作相同声明。

[官方 preprocessing README](../../../research_repos/internal_guidance_study/Internal-Guidance/SiT/preprocessing/README.md:23)也说明 EMA/MSE 的区别在 decoder；[预处理 encode 默认值](../../../research_repos/internal_guidance_study/Internal-Guidance/SiT/preprocessing/dataset_tools.py:380)是 MSE。XL 训练脚本中加载 EMA 不意味着需要换 latent 坐标。因此本 pilot 统一使用 MSE decoder 合理，避免把 decoder 差异混入不同生成场的 copy 对照。

本次输入已经冻结为 **R0 只抽一次 VAE posterior noise**，随后五轮均无新随机变量；不应把它改写成 posterior mode 实验。每轮传递 latent，decode 仅用于展示和读数；pixel 误差相对 R0 解码图，不能称完整 RGB 输入→RGB 输出的 VAE 多轮循环误差。

## 4. xl_reference.py 只读审查

审查对象：[xl_reference.py](../../../experiments/fm_common_inverse_copy_20260913/xl_reference.py:21) 与其复用的 [run.py](../../../experiments/fm_common_inverse_copy_20260913/run.py:110)。**当前 FP32 pilot 无阻塞问题。**

- `call()` 同时执行 `1-t` 与 `-full`，label 映射正确；reference 分支不进入 CFG/AG 外推。模型加载后为 eval，运行主体包在 inference mode，TF32 关闭且没有 autocast。
- `run.py` 每轮先反演 `previous=current`，再以待测场生成新的 `current`；四臂分别从同一冻结 R0 开始，确实逐轮回灌。每方向 128 Heun 步；4 图先做 128/256 精化和 reference 自往返检查。
- root 冻结输入已读到 `clean=float32[16,4,32,32]`，labels 转 long 后查询；后续场与状态保持 FP32。因此 wrapper 的 `z.new_full` 当前会生成正确的 FP32 时间。
- **未来 FP64 状态边界：** wrapper 第 51 行的时间 dtype 跟随 z；若之后将 z 改 FP64，须像上面的示例显式构造 FP32 model time，并把返回速度转换回 state dtype。当前运行不需修改；本审计没有修改已冻结源码。
- XL 与 small 每次网络调用成本不同，request 已记录这一点。总 `calls` 是调用次数，不能解释为等 FLOPs。源图重建改善仍不是新 Gaussian 噪声生成的质量证明。

本次核对 request 与当前源码 hash 相符，输入 SHA256 为 `4430c6b96f9531a1f85c8c143a3faeaa9356d92b97e6bef9f2ac800a9744fd3f`。源码指纹如下，补充 wrapper request 未独立冻结的 XL loader/model 文件身份：

| 文件 | SHA256 |
|---|---|
| `experiments/fm_common_inverse_copy_20260913/xl_reference.py` | `6d431bd839cf21a887f1101b62e0d003578965ef67ea2649ca7dd66add01c2c9` |
| `experiments/fm_common_inverse_copy_20260913/run.py` | `2c894a0ef1d34a4c655390694466abfbaffea2648b7553db2d179da61696c786` |
| `experiments/run_internal_guidance_sit_audit.py` | `a76a8aaa9509536b861f8a7d365f872fd2eb4d71d43fbf11c5c922abdf4ada9d` |
| `research_repos/internal_guidance_study/Internal-Guidance/SiT/models/sit.py` | `205e1b7fabf106d5f720ed03f70bc43b4d3a93330834425ba091431ef7655e51` |
| `research_repos/internal_guidance_study/Internal-Guidance/SiT/loss.py` | `4c9844cdea6fef233865707fbd52d1b0cb5fee18ce00202cca74efc263671e5a` |
