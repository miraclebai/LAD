#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save DWT-LL visualizations and their ABSOLUTE difference map.

For each input image, save THREE grayscale PNGs:
  1) <stem>_LL_Y.png        : DWT-LL on luminance Y
  2) <stem>_LL_RGB.png      : DWT-LL on naive RGB grayscale (avg)
  3) <stem>_LL_absdiff.png  : |LL_Y - LL_RGB| (absolute difference)

Notes:
- LL is upsampled to original resolution (H,W)
- All outputs are min-max normalized to [0,1] for visualization
- Output images are clean grayscale PNGs (no title, no border)

Requirements:
- torch, numpy, pillow
- dwt_init importable
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ⚠️ Adjust to your repo structure if needed
# e.g. from haze_mass.dwt import dwt_init
from dwt import dwt_init


# -----------------------
# IO + conversions
# -----------------------

def load_rgb01(img_path: str) -> torch.Tensor:
    """Load RGB image -> [1,3,H,W] float32 in [0,1]."""
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten


def rgb_to_luminance(img01: torch.Tensor) -> torch.Tensor:
    """RGB -> Y luminance, [1,1,H,W]."""
    r, g, b = img01[:, 0:1], img01[:, 1:2], img01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_gray_avg(img01: torch.Tensor) -> torch.Tensor:
    """Naive RGB grayscale avg, [1,1,H,W]."""
    return img01.mean(dim=1, keepdim=True)


def pad_to_even(x: torch.Tensor) -> torch.Tensor:
    """Pad reflect so H,W are even (required by many DWT implementations)."""
    _, _, H, W = x.shape
    if (H % 2) != 0 or (W % 2) != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2), mode="reflect")
    return x


def minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image min-max normalization to [0,1]."""
    xf = x.view(1, -1)
    x_min = xf.min(dim=1, keepdim=True)[0]
    x_max = xf.max(dim=1, keepdim=True)[0]
    return ((xf - x_min) / (x_max - x_min + eps)).view_as(x)


def compute_dwt_ll(x01: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Compute DWT-LL only.
    Return: [1,1,H,W] in [0,1] (normalized)
    """
    x01 = pad_to_even(x01)
    LL, _ = dwt_init(x01)  # LL: [1,1,H/2,W/2]

    H, W = out_hw
    LL_up = F.interpolate(LL, size=(H, W), mode="bilinear", align_corners=False)
    return minmax_norm(LL_up)


def save_gray_png(x01: torch.Tensor, out_path: str) -> None:
    """Save [1,1,H,W] tensor to 8-bit grayscale PNG."""
    arr = x01.squeeze(0).squeeze(0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(out_path)


# -----------------------
# Batch processing
# -----------------------

def collect_images(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        pp = Path(p)
        if pp.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                paths.extend(sorted(pp.glob(ext)))
        else:
            paths.append(pp)
    return [p for p in paths if p.exists() and p.is_file()]


def process_one(img_path: str, out_dir: str) -> None:
    img = load_rgb01(img_path)
    _, _, H, W = img.shape
    stem = Path(img_path).stem

    # LL on Y
    y = rgb_to_luminance(img)
    LL_y = compute_dwt_ll(y, (H, W))
    out_y = os.path.join(out_dir, f"{stem}_LL_Y.png")
    save_gray_png(LL_y, out_y)

    # LL on RGB grayscale avg
    g = rgb_to_gray_avg(img)
    LL_g = compute_dwt_ll(g, (H, W))
    out_g = os.path.join(out_dir, f"{stem}_LL_RGB.png")
    save_gray_png(LL_g, out_g)

    # Absolute difference
    absdiff = torch.abs(LL_y - LL_g)
    absdiff = minmax_norm(absdiff)
    out_d = os.path.join(out_dir, f"{stem}_LL_absdiff.png")
    save_gray_png(absdiff, out_d)

    print(f"[OK] {img_path}")
    print(f"  -> {out_y}")
    print(f"  -> {out_g}")
    print(f"  -> {out_d}   (|LL_Y - LL_RGB|)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["/Users/baijingyuan/Desktop/result/ROI/hazy_mass/00015.png"],
        help="Image file(s) or directory."
    )
    parser.add_argument("--out_dir", type=str, default="./dwt_ll_absdiff")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imgs = collect_images(args.inputs)
    if not imgs:
        print("[ERR] No images found.")
        return

    for p in imgs:
        process_one(str(p), args.out_dir)


if __name__ == "__main__":
    main()
