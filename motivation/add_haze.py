# motivation/vis_global_thin_haze.py
"""
Apply a global thin haze to a single RGB image.

This script is ONLY for visualization / control experiments:
- uniform haze
- very light (thin fog)
- physically interpretable

Output:
  input.png   (original)
  output.png  (globally hazed)
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


def img_to_tensor_01(path: str) -> torch.Tensor:
    """RGB image -> [1,3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0)


def save_rgb_01(x: torch.Tensor, path: str) -> None:
    """Save [1,3,H,W] in [0,1] as PNG."""
    x = x.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    arr = (x * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def apply_global_haze(
    img01: torch.Tensor,
    transmission: float = 0.9,
    A: float = 1.0,
) -> torch.Tensor:
    """
    Apply global haze using atmospheric scattering model.

    Args:
        img01: [1,3,H,W] in [0,1]
        transmission: t in (0,1], closer to 1 = thinner haze
        A: atmospheric light (1.0 = white fog)

    Returns:
        hazy image in [0,1]
    """
    t = float(transmission)
    A_rgb = torch.full_like(img01, float(A))
    hazy = img01 * t + A_rgb * (1.0 - t)
    return hazy.clamp(0, 1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="/Users/baijingyuan/Desktop/result/8.png")
    parser.add_argument("--outdir", type=str, default="/Users/baijingyuan/Desktop/result/8_hazy.png")

    # haze strength
    parser.add_argument(
        "--t",
        type=float,
        default=0.76,
        help="global transmission (0.85~0.95 recommended for thin haze)",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=0.9,
        help="atmospheric light (1.0 = white haze)",
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load
    img01 = img_to_tensor_01(args.img)

    # apply haze
    hazy = apply_global_haze(img01, transmission=args.t, A=args.A)

    # save
    save_rgb_01(img01, os.path.join(args.outdir, "input.png"))
    save_rgb_01(hazy, os.path.join(args.outdir, "output.png"))

    print(f"Saved to {args.outdir}")
    print(f"Global thin haze: transmission t={args.t}, A={args.A}")


if __name__ == "__main__":
    main()
