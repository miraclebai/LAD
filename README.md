# LAD

本仓库是论文 *From Haze Degradation to Haze Evolution: A Physically Consistent Generative Framework for Image Dehazing* 的官方实现。

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

## 3. 训练前的配置调整

这个仓库中多个配置文件保留了作者机器上的绝对路径。训练或推理前，请先替换这些字段，或者用命令行覆盖：

- 数据路径：`data.params.train.params.dataroot`、`data.params.validation.params.dataroot`
- Stage1 checkpoint：`model.params.ckpt_path`
- Stage2 中 first-stage checkpoint：`model.params.first_stage_config.params.ckpt_path`
- 推理配置中的 diffusion / first-stage checkpoint：`configs/infer.yaml` 内的 `ckpt_path`

建议：

- Stage1 从头训练时，把 `configs/stage1.yaml` 里的 `model.params.ckpt_path` 改成 `null`。
- Stage2 训练时，把 `model.params.first_stage_config.params.ckpt_path` 指向训练好的 Stage1 权重。
- Stage3 训练时，把 `model.params.ckpt_path` 指向 Stage1 权重。
- 推理时，`scripts/infer.py` 和 `scripts/infer_os.py` 会在实例化模型时先读取 `configs/infer.yaml` 与 `configs/stage3.yaml` 里的 `ckpt_path`，所以这两个 YAML 中的绝对路径必须先改成可用路径或 `null`，否则脚本会在真正加载 `--ckpt` / `--vqgan_ckpt` 之前就报错。

## 4. 训练

训练入口统一是：

```bash
python main.py -t True --base <config> --gpus 0,
```

训练日志与模型权重默认保存在 `logs/<timestamp>_<name>/`。

### 4.1 Stage1：训练去雾自编码器

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

### 4.2 Stage2：训练 latent diffusion

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

### 4.3 Stage3：微调解码端

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

## 5. 推理

仓库提供两个推理脚本：

- [`scripts/infer.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/infer.py)：固定 resize 到 `input_size`，适合 `512x512` 流程。
- [`scripts/infer_os.py`](/Users/baijingyuan/Desktop/ijcai26/LAD/scripts/infer_os.py)：按原图尺寸读取，pad 到 128 的倍数后推理，再裁回原尺寸，适合真实场景图片。

两者共同特性如下：

- `--init-img` 输入的是“目录”，不是单张图。
- 输出目录下若已存在同名结果，会自动跳过。
- 推理 prompt 写死在脚本里：

```text
(masterpiece:2), (best quality:2), (realistic:2), (very clear:2),(haze-free:2)
```

如果想改文本条件，需要直接修改脚本中的 `text_init`。

### 5.1 固定分辨率推理

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

### 5.2 原尺寸推理

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

该脚本的处理流程为：

- 原尺寸读取图像；
- 右侧和下侧 pad 到 128 的倍数；
- 完成扩散采样与解码；
- 最后裁回原始高宽。

若测试图像不满足固定分辨率假设，建议优先采用该脚本。
