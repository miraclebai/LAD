# LAD

本仓库是论文 *From Haze Degradation to Haze Evolution: A Physically Consistent Generative Framework for Image Dehazing* 的官方实现。

## 1. 仓库结构

- [`configs/stage1.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/stage1.yaml)：Stage1，自编码器预训练。
- [`configs/stage2.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/stage2.yaml)：Stage2，latent diffusion 训练。
- [`configs/stage3.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/stage3.yaml)：Stage3，冻结编码端后微调解码端、LCGM 和输出校正头。
- [`configs/infer.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/infer.yaml)：扩散推理配置模板。
- [`data/data.py`](https://github.com/miraclebai/LAD/blob/main/data/data.py)：当前仓库实际用到的数据集定义。
- [`hazy_mass/`](https://github.com/miraclebai/LAD/tree/main/hazy_mass)：DCP、DWT 与 haze mass map 计算。
- [`MFlow/aniso_mflow.py`](https://github.com/miraclebai/LAD/blob/main/MFlow/aniso_mflow.py)：各向异性扩散驱动的 `M-Flow`。
- [`motivation/`](https://github.com/miraclebai/LAD/tree/main/motivation)：论文动机图和可视化脚本，不参与主训练流程。

## 2. 数据格式

当前提交版本的训练配置实际使用的是 [`data.data.HazyPairDataset`](https://github.com/miraclebai/LAD/blob/main/data/data.py)，目录格式必须是：

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

## 3. 训练前的配置调整

这个仓库中多个配置文件保留了作者机器上的绝对路径。训练或推理前，请先替换这些字段，或者用命令行覆盖：

- 数据路径：`data.params.train.params.dataroot`、`data.params.validation.params.dataroot`
- Stage1 checkpoint：`model.params.ckpt_path`
- Stage2 中 first-stage checkpoint：`model.params.first_stage_config.params.ckpt_path`
- 推理配置中的 diffusion / first-stage checkpoint：`configs/infer.yaml` 内的 `ckpt_path`

## 4. 训练

训练入口统一是：

```bash
python main.py -t True --base <config> --gpus 0,
```

训练日志与模型权重默认保存在 `logs/<timestamp>_<name>/`。

### 4.1 Stage1：训练去雾自编码器

Stage1 目标是训练 [`HazeAutoencoderKLResi`](https://github.com/miraclebai/LAD/blob/main/ldm/models/autoencoder.py)，编码端自动计算 `M0` 并通过 `HazeAwareEncoder` 注入 M-Flow。

```bash
python main.py -t True \
  --base configs/stage1.yaml \
  --gpus 0, \
  --name stage1 \
```

### 4.2 Stage2：训练 latent diffusion

Stage2 训练 [`ldm.models.diffusion.ddpm.HazeLatentDiffusion`](https://github.com/miraclebai/LAD/blob/main/ldm/models/diffusion/ddpm.py)。

```bash
python main.py -t True \
  --base configs/stage2.yaml \
  --gpus 0, \
  --name stage2 \
```

### 4.3 Stage3：微调解码端

命令如下：

```bash
python main.py -t True \
  --base configs/stage3.yaml \
  --gpus 0, \
  --name stage3 \
```

## 5. 推理

仓库提供两个推理脚本：

- [`scripts/infer.py`](https://github.com/miraclebai/LAD/blob/main/scripts/infer.py)：固定分辨率推理。
- [`scripts/infer_os.py`](https://github.com/miraclebai/LAD/blob/main/scripts/infer_os.py)：原始分辨率推理。

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
