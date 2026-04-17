# LAD

LAD 是一个面向单幅图像去雾的三阶段训练仓库。代码主线由以下三部分组成：

- `HazeAutoencoderKLResi`：去雾自编码器，编码端使用 `HazeAwareEncoder`，内部通过 DCP + DWT 生成初始 haze mass map `M0`，再用 `M-Flow` 做逐层雾质量图演化。
- `HazeLatentDiffusion`：在 latent 空间中做去雾扩散，文本条件来自 OpenCLIP，结构条件来自 haze mass map，而不是普通的 hazy latent。
- `LCGM + GuideOutputAffine`：在解码端注入颜色/亮度引导，用于 Stage3 微调和最终重建。

仓库内同时带有 `ldm`、`taming`、`basicsr` 的源码拷贝，因此大多数核心依赖已经 vendored 到仓库中；训练入口是 [`main.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/main.py)，推理入口是 [`scripts/infer.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/infer.py) 和 [`scripts/infer_os.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/infer_os.py)。

## 1. 仓库结构

- [`configs/stage1.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/stage1.yaml)：Stage1，自编码器预训练。
- [`configs/stage2.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/stage2.yaml)：Stage2，latent diffusion 训练。
- [`configs/stage3.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/stage3.yaml)：Stage3，冻结编码端后微调解码端、LCGM 和输出校正头。
- [`configs/infer.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/infer.yaml)：扩散推理配置模板。
- [`data/data.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/data/data.py)：当前仓库实际用到的数据集定义。
- [`hazy_mass/`](/Users/baijingyuan/Desktop/ijcai26/LAD/hazy_mass)：DCP、DWT 与 haze mass map 计算。
- [`MFlow/aniso_mflow.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/MFlow/aniso_mflow.py)：各向异性扩散驱动的 `M-Flow`。
- [`motivation/`](/Users/baijingyuan/Desktop/ijcai26/LAD/motivation)：论文动机图和可视化脚本，不参与主训练流程。

## 2. 数据格式

当前提交版本的训练配置实际使用的是 [`data.data.HazyPairDataset`](/Users/baijingyuan/Desktop/ijcai26/LAD/data/data.py)，目录格式必须是：

```text
your_dataset/
├── hazy/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── gt/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

要求：

- `hazy/` 和 `gt/` 通过文件名 stem 一一对应。
- 当前 `HazyPairDataset` 默认只枚举 `*.png`。
- 图像会在数据集内部 resize 到配置里的 `size`，然后以 `[0,1]` Tensor 返回；模型内部再统一映射到 `[-1,1]`。

## 3. 环境依赖

仓库没有提供 `requirements.txt`，按源码导入关系，至少需要以下 Python 包：

```bash
pip install torch torchvision pytorch-lightning omegaconf einops tqdm pillow numpy opencv-python scipy scikit-image matplotlib packaging wandb transformers open-clip-torch
pip install git+https://github.com/openai/CLIP.git
```

额外说明：

- `main.py` 默认使用 `WandbLogger`。如果不想联网记录，建议先执行 `export WANDB_MODE=offline`。
- Stage2 和推理使用 `FrozenOpenCLIPEmbedder`，而该类在 [`ldm/modules/encoders/modules.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/ldm/modules/encoders/modules.py) 中的默认权重路径写死为 `/disk8t/baijy/Stable_main/weights/open_clip_pytorch_model.bin`。你需要自行处理这个问题，常见做法有两种：
  - 直接把 OpenCLIP 权重放到同名路径。
  - 在 `configs/stage2.yaml` 和 `configs/infer.yaml` 的 `cond_stage_config.params` 中显式补上 `version: /your/path/open_clip_pytorch_model.bin`。

## 4. 训练前必须处理的路径

这个仓库中多个配置文件保留了作者机器上的绝对路径。训练或推理前，请先替换这些字段，或者用命令行覆盖：

- 数据路径：`data.params.train.params.dataroot`、`data.params.validation.params.dataroot`
- Stage1 checkpoint：`model.params.ckpt_path`
- Stage2 中 first-stage checkpoint：`model.params.first_stage_config.params.ckpt_path`
- 推理配置中的 diffusion / first-stage checkpoint：`configs/infer.yaml` 内的 `ckpt_path`

建议做法：

- Stage1 从头训练时，把 `configs/stage1.yaml` 里的 `model.params.ckpt_path` 改成 `null`。
- Stage2 训练时，把 `model.params.first_stage_config.params.ckpt_path` 指向训练好的 Stage1 权重。
- Stage3 训练时，把 `model.params.ckpt_path` 指向 Stage1 权重。
- 推理时，`scripts/infer.py` 和 `scripts/infer_os.py` 会在实例化模型时先读取 `configs/infer.yaml` 与 `configs/stage3.yaml` 里的 `ckpt_path`，所以这两个 YAML 中的绝对路径必须先改成可用路径或 `null`，否则脚本会在真正加载 `--ckpt` / `--vqgan_ckpt` 之前就报错。

## 5. 训练

训练入口统一是：

```bash
python main.py -t True --base <config> --gpus 0,
```

日志和权重默认保存在 `logs/<timestamp>_<name>/`。

### 5.1 Stage1：训练去雾自编码器

Stage1 目标是训练 [`HazeAutoencoderKLResi`](/Users/baijingyuan/Desktop/ijcai26/LAD/ldm/models/autoencoder.py)，编码端自动计算 `M0` 并通过 `HazeAwareEncoder` 注入 M-Flow。

```bash
python main.py -t True \
  --base configs/stage1.yaml \
  --gpus 0, \
  --name stage1 \
  model.params.ckpt_path=null \
  data.params.train.params.dataroot=/path/to/train_set \
  data.params.validation.params.dataroot=/path/to/val_set
```

说明：

- `configs/stage1.yaml` 的数据集是 `data.data.HazyPairDataset`。
- Stage1 只训练自编码器，不依赖 OpenCLIP。
- 损失函数是 LPIPS + discriminator，配置在 `lossconfig` 中。

### 5.2 Stage2：训练 latent diffusion

Stage2 训练 [`ldm.models.diffusion.ddpm.HazeLatentDiffusion`](/Users/baijingyuan/Desktop/ijcai26/LAD/ldm/models/diffusion/ddpm.py)。

```bash
python main.py -t True \
  --base configs/stage2.yaml \
  --gpus 0, \
  --name stage2 \
  model.params.ckpt_path=null \
  model.params.first_stage_config.params.ckpt_path=/path/to/stage1.ckpt \
  data.params.train.params.dataroot=/path/to/train_set \
  data.params.validation.params.dataroot=/path/to/val_set
```

说明：

- Stage2 会同时实例化：
  - first stage：`HazeAutoencoderKLResi`
  - text condition：`FrozenOpenCLIPEmbedder`
  - struct condition：`EncoderUNetModelWT`
- 当前实现中，结构条件来自 AE 编码器输出的 haze mass map，而不是普通的 hazy latent。
- `main.py` 中默认只让扩散部分的 `spade` 参数可训练，`structcond_stage_model` 会训练，first stage 默认冻结。

如果你想从一个已有的 diffusion checkpoint 继续训练，可以把 `model.params.ckpt_path` 改成对应 `.ckpt` 文件，而不是 `null`。

### 5.3 Stage3：微调解码端

Stage3 仍然训练 [`HazeAutoencoderKLResi`](/Users/baijingyuan/Desktop/ijcai26/LAD/ldm/models/autoencoder.py)，但 `freeze_dec=True` 时会冻结编码端和 `quant_conv`，只训练：

- `decoder`
- `post_quant_conv`
- `lcgm`
- `guide_head`
- discriminator

命令如下：

```bash
python main.py -t True \
  --base configs/stage3.yaml \
  --gpus 0, \
  --name stage3 \
  model.params.ckpt_path=/path/to/stage1.ckpt \
  data.params.train.params.dataroot=/path/to/train_set \
  data.params.validation.params.dataroot=/path/to/val_set
```

## 6. 推理

当前仓库提供两个推理脚本：

- [`scripts/infer.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/infer.py)：固定 resize 到 `input_size`，适合 `512x512` 流程。
- [`scripts/infer_os.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/infer_os.py)：按原图尺寸读取，pad 到 128 的倍数后推理，再裁回原尺寸，适合真实场景图片。

两者共同特点：

- `--init-img` 输入的是“目录”，不是单张图。
- 输出目录下若已存在同名结果，会自动跳过。
- 推理 prompt 写死在脚本里：

```text
(masterpiece:2), (best quality:2), (realistic:2), (very clear:2),(haze-free:2)
```

如果想改文本条件，需要直接修改脚本中的 `text_init`。

### 6.1 固定分辨率推理

```bash
python scripts/infer.py \
  --config configs/infer.yaml \
  --ckpt /path/to/stage2.ckpt \
  --vqgan_ckpt /path/to/stage3.ckpt \
  --init-img /path/to/hazy_images \
  --outdir results/infer_512 \
  --ddpm_steps 200 \
  --input_size 512 \
  --colorfix_type wavelet
```

参数说明：

- `--ckpt`：Stage2 diffusion checkpoint。
- `--vqgan_ckpt`：Stage3 decoder / VQ checkpoint。
- `--ddpm_steps`：采样步数；越大越慢。
- `--colorfix_type`：`nofix`、`adain`、`wavelet` 三选一。
- `--dec_w`：控制 `Decoder_Mix` 融合权重。

### 6.2 原尺寸推理

```bash
python scripts/infer_os.py \
  --config configs/infer.yaml \
  --ckpt /path/to/stage2.ckpt \
  --vqgan_ckpt /path/to/stage3.ckpt \
  --init-img /path/to/hazy_images \
  --outdir results/infer_os \
  --ddpm_steps 200 \
  --colorfix_type wavelet
```

这版脚本会：

- 原尺寸读取图像；
- 右侧和下侧 pad 到 128 的倍数；
- 完成扩散采样与解码；
- 最后裁回原始高宽。

如果你的测试图不是正方形，优先使用这个脚本。

## 7. 可选预处理

[`scripts/precompute_m0.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/precompute_m0.py) 可以离线计算 haze mass map 并保存为 `.npy`：

```bash
python scripts/precompute_m0.py \
  --hazy_dir /path/to/hazy_images \
  --out_dir /path/to/m0_npy \
  --size 512 \
  --device cuda
```

不过要说明的是：当前仓库提交的 `stage1.yaml`、`stage2.yaml`、`stage3.yaml` 并没有直接使用这个离线 `m0` 流程；它们主要依赖模型内部在线计算的 haze mass map。这个脚本更适合你后续切换到 [`ResizePairedDataset`](/Users/baijingyuan/Desktop/ijcai26/LAD/data/data.py) 时使用。

## 8. 代码层面的补充说明

基于当前源码，主流程建议按下面理解：

- Stage1：学习一个带 haze-aware encoder 的去雾自编码器。
- Stage2：在 Stage1 latent 空间上训练条件扩散，文本条件由 OpenCLIP 提供，结构条件由 haze mass map 提供。
- Stage3：冻结编码端，只微调解码端与颜色引导模块，提升最终输出质量。

另外两点也值得注意：

- [`configs/exp_e.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/exp_e.yaml) 与 [`ldm/models/diffusion/haze_ldm.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/ldm/models/diffusion/haze_ldm.py) 更像实验性或旧版本路径；当前仓库真正打通的训练/推理主线是 `stage1/stage2/stage3 + ddpm.py + infer*.py`。
- `configs/infer.yaml` 中的 `data` 和 `test_data` 字段对 `scripts/infer.py` / `scripts/infer_os.py` 实际并不生效；推理脚本直接从图像目录读入，不走 Lightning datamodule。

## 9. 常见问题

### 9.1 一启动就报 checkpoint 路径不存在

先检查以下文件中的绝对路径是否已经替换：

- [`configs/stage1.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/stage1.yaml)
- [`configs/stage2.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/stage2.yaml)
- [`configs/stage3.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/stage3.yaml)
- [`configs/infer.yaml`](/Users/baijingyuan/Desktop/ijcai26/LAD/configs/infer.yaml)
- [`ldm/modules/encoders/modules.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/ldm/modules/encoders/modules.py) 中 `FrozenOpenCLIPEmbedder` 的默认权重路径

### 9.2 推理脚本没有读取我的单张图片

`--init-img` 需要传目录，脚本内部使用 `os.listdir()` 枚举文件。

### 9.3 为什么推理时还需要 `--vqgan_ckpt`

因为当前推理是两段式的：

- Stage2 diffusion 负责在 latent 空间采样；
- Stage3 AE / decoder 负责把 latent 解码成最终图像。

所以 `--ckpt` 和 `--vqgan_ckpt` 缺一不可。
