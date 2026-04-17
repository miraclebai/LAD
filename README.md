# LAD

This repository is the official implementation of the paper *From Haze Degradation to Haze Evolution: A Physically Consistent Generative Framework for Image Dehazing*.

## 1. Repository Structure

- [`configs/stage1.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/stage1.yaml): Stage 1, autoencoder pre-training.
- [`configs/stage2.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/stage2.yaml): Stage 2, latent diffusion training.
- [`configs/stage3.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/stage3.yaml): Stage 3, decoder fine-tuning with frozen encoder.
- [`configs/infer.yaml`](https://github.com/miraclebai/LAD/blob/main/configs/infer.yaml): inference configuration.
- [`data/data.py`](https://github.com/miraclebai/LAD/blob/main/data/data.py): dataset definitions used by the current training pipeline.
- [`hazy_mass/`](https://github.com/miraclebai/LAD/tree/main/hazy_mass): DCP, DWT, and haze mass map computation.
- [`MFlow/aniso_mflow.py`](https://github.com/miraclebai/LAD/blob/main/MFlow/aniso_mflow.py): anisotropic diffusion based `M-Flow` module.

## 2. Data Format

The current training configuration uses [`data.data.HazyPairDataset`](https://github.com/miraclebai/LAD/blob/main/data/data.py). The dataset directory should be organized as follows:

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

## 3. Configuration Preparation

Several configuration files still contain absolute paths from the original training environment. Before training or inference, please replace or override the following fields:

- Data path: `data.params.train.params.dataroot`, `data.params.validation.params.dataroot`
- Stage 1 checkpoint: `model.params.ckpt_path`
- Stage 2 first-stage checkpoint: `model.params.first_stage_config.params.ckpt_path`
- Diffusion / first-stage checkpoint in inference: `ckpt_path` in `configs/infer.yaml`

## 4. Training

The unified training entry is:

```bash
python main.py -t True --base <config> --gpus 0,
```

Training logs and checkpoints are saved to `logs/<timestamp>_<name>/` by default.

### 4.1 Stage 1: Dehazing Autoencoder Training

Stage 1 trains [`HazeAutoencoderKLResi`](https://github.com/miraclebai/LAD/blob/main/ldm/models/autoencoder.py).

```bash
python main.py -t True \
  --base configs/stage1.yaml \
  --gpus 0, \
  --name stage1 \
```

### 4.2 Stage 2: Latent Diffusion Training

Stage 2 trains [`ldm.models.diffusion.ddpm.HazeLatentDiffusion`](https://github.com/miraclebai/LAD/blob/main/ldm/models/diffusion/ddpm.py).

```bash
python main.py -t True \
  --base configs/stage2.yaml \
  --gpus 0, \
  --name stage2 \
```

### 4.3 Stage 3: Decoder Fine-Tuning

The training command is:

```bash
python main.py -t True \
  --base configs/stage3.yaml \
  --gpus 0, \
  --name stage3 \
```

## 5. Inference

The repository provides two inference scripts:

- [`scripts/infer.py`](https://github.com/miraclebai/LAD/blob/main/scripts/infer.py): fixed-resolution inference.
- [`scripts/infer_os.py`](https://github.com/miraclebai/LAD/blob/main/scripts/infer_os.py): original-resolution inference.

### 5.1 Fixed-Resolution Inference

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

### 5.2 Original-Resolution Inference

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
