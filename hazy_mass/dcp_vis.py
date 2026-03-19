#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCP haze visualization + grayscale export (standalone script)

功能：
1) 读取一张 RGB 图（[0,1]）
2) 用 dcp_haze_strength 计算 dark / transmission t / haze strength h=1-t
3) 保存灰度图：dark.png, t.png, h.png
4) 保存可视化拼图：dcp_haze_maps.png（2x3）

依赖：
- torch
- numpy
- pillow
- matplotlib
- 你的 dcp.py（同目录下，或可被 Python import 到）

用法：
python dcp_visualize.py --img "/path/to/00015.png" --out_dir "./out"
"""

import os
import argparse
from typing import Optional

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ✅ 如果你的文件是 haze_mass/dcp.py，请用：
# from haze_mass.dcp import dcp_haze_strength
# ✅ 如果你的文件是当前目录 dcp.py，请用：
from dcp import dcp_haze_strength


# --------------------------
# I/O helpers
# --------------------------

def load_rgb01(img_path: str) -> torch.Tensor:
    """Load RGB image and return tensor [1,3,H,W] in [0,1]."""
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3] in [0,1]
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return ten


def to_hw(x: torch.Tensor) -> np.ndarray:
    """[1,1,H,W] -> [H,W] numpy"""
    if x.dim() == 4:
        return x.detach().cpu().squeeze(0).squeeze(0).numpy()
    elif x.dim() == 2:
        return x.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported shape for to_hw: {tuple(x.shape)}")


def save_gray_map(
    x: torch.Tensor,
    save_path: str,
    normalize: bool = True,
) -> None:
    """
    Save single-channel tensor as 8-bit grayscale PNG.

    Args:
        x: [1,1,H,W] or [H,W] torch tensor
        save_path: output path (e.g. out/haze_strength_h.png)
        normalize: whether min-max normalize to [0,1] before saving
                   - True: better contrast for visualization
                   - False: preserves absolute scale (assumes x already in [0,1])
    """
    if x.dim() == 4:
        x = x.squeeze(0).squeeze(0)  # [H,W]
    elif x.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

    x_np = x.detach().cpu().numpy().astype(np.float32)

    if normalize:
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    else:
        x_np = np.clip(x_np, 0.0, 1.0)

    x_uint8 = (x_np * 255.0).round().astype(np.uint8)
    Image.fromarray(x_uint8, mode="L").save(save_path)


# --------------------------
# Main visualization
# --------------------------

def visualize_dcp_maps(
    img_path: str,
    out_dir: str = "./out",
    patch_size: int = 15,
    omega: float = 0.95,
    top_percent: float = 0.001,
    mosaic_name: str = "dcp_haze_maps.png",
    save_gray: bool = True,
    gray_normalize: bool = False,
) -> None:
    """
    Compute and visualize DCP-related maps.

    Saves:
    - (optional) dark_channel.png, transmission_t.png, haze_strength_h.png in out_dir
    - mosaic figure (2x3) in out_dir/mosaic_name
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) load
    img = load_rgb01(img_path)  # [1,3,H,W] in [0,1]

    # 2) compute
    dark, t, h = dcp_haze_strength(
        img, patch_size=patch_size, omega=omega, top_percent=top_percent
    )

    # 3) save grayscale maps (single-channel PNG)
    if save_gray:
        save_gray_map(dark, os.path.join(out_dir, "dark_channel.png"), normalize=gray_normalize)
        save_gray_map(t,    os.path.join(out_dir, "transmission_t.png"), normalize=gray_normalize)
        save_gray_map(h,    os.path.join(out_dir, "haze_strength_h.png"), normalize=gray_normalize)

    # 4) mosaic visualization
    img_np  = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    dark_np = to_hw(dark)
    t_np    = to_hw(t)
    h_np    = to_hw(h)

    # overlay 用一个可视化归一化（仅用于叠加显示）
    h_vis = (h_np - h_np.min()) / (h_np.max() - h_np.min() + 1e-8)

    fig = plt.figure(figsize=(12, 8))

    ax = plt.subplot(2, 3, 1)
    ax.imshow(img_np)
    ax.set_title("Input")
    ax.axis("off")

    ax = plt.subplot(2, 3, 2)
    ax.imshow(dark_np, cmap="gray")
    ax.set_title("Dark channel")
    ax.axis("off")

    ax = plt.subplot(2, 3, 3)
    ax.imshow(t_np, cmap="gray")
    ax.set_title("Transmission t(x)")
    ax.axis("off")

    ax = plt.subplot(2, 3, 4)
    ax.imshow(h_np, cmap="gray")
    ax.set_title("Haze strength h(x)=1-t(x)")
    ax.axis("off")

    ax = plt.subplot(2, 3, 5)
    ax.imshow(img_np)
    ax.imshow(h_vis, cmap="gray", alpha=0.5)
    ax.set_title("Overlay: input + haze map")
    ax.axis("off")

    ax = plt.subplot(2, 3, 6)
    ax.imshow(img_np)
    ax.imshow(1.0 - t_np, cmap="gray", alpha=0.5)
    ax.set_title("Overlay: input + (1 - t)")
    ax.axis("off")

    plt.tight_layout()
    mosaic_path = os.path.join(out_dir, mosaic_name)
    plt.savefig(mosaic_path, dpi=200)
    plt.show()

    print(f"[OK] Saved mosaic to: {mosaic_path}")
    if save_gray:
        print(f"[OK] Saved gray maps to: {out_dir}")
        print(f"     - dark_channel.png")
        print(f"     - transmission_t.png")
        print(f"     - haze_strength_h.png")


# --------------------------
# CLI
# --------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=str, default="/Users/baijingyuan/Desktop/result/00015.png", help="Path to input image (RGB).")
    p.add_argument("--out_dir", type=str, default="./out", help="Output directory.")
    p.add_argument("--patch_size", type=int, default=15, help="DCP patch size.")
    p.add_argument("--omega", type=float, default=0.95, help="DCP omega.")
    p.add_argument("--top_percent", type=float, default=0.001, help="Top percent for atmospheric light.")
    p.add_argument("--mosaic_name", type=str, default="dcp_haze_maps.png", help="Mosaic filename.")
    p.add_argument("--no_gray", action="store_true", help="Do not save grayscale PNG maps.")
    p.add_argument(
        "--gray_normalize",
        action="store_true",
        help="Min-max normalize maps before saving grayscale PNG (better contrast, not absolute scale).",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    visualize_dcp_maps(
        img_path=args.img,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        omega=args.omega,
        top_percent=args.top_percent,
        mosaic_name=args.mosaic_name,
        save_gray=(not args.no_gray),
        gray_normalize=args.gray_normalize,
    )

"""
示例（你的路径）：
python dcp_visualize.py \
  --img "/Users/baijingyuan/Desktop/result/00015.png" \
  --out_dir "./dcp_vis" \
  --patch_size 15 --omega 0.95 --top_percent 0.001

如果你希望灰度图“更显著”（对比更强）：
加 --gray_normalize
"""
