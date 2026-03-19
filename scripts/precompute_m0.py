import os, glob, argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from hazy_mass.hazy_mass import compute_hazy_mass_map


def load_rgb_01(path: str, size: int | None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), resample=Image.BICUBIC)
    t = TF.to_tensor(img)  # [0,1], [3,H,W]
    return t


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazy_dir", type=str, default = "/disk8t/baijy/dataset/Hazyset/hazy")
    parser.add_argument("--out_dir", type=str, default = "/disk8t/baijy/dataset/Hazyset/m0")
    parser.add_argument("--size", type=int, default=512, help="必须与训练时 resize 后的分辨率一致")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exts", type=str, default="png,jpg,jpeg,bmp,tif,tiff")
    # compute_hazy_mass_map params
    parser.add_argument("--dcp_patch_size", type=int, default=15)
    parser.add_argument("--dcp_omega", type=float, default=0.95)
    parser.add_argument("--dcp_top_percent", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    exts = tuple("." + e.strip().lower() for e in args.exts.split(","))
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.hazy_dir, "**", f"*{e}"), recursive=True)
    paths = sorted(paths)

    for p in paths:
        stem = Path(p).stem
        out_path = os.path.join(args.out_dir, stem + ".npy")
        if os.path.exists(out_path):
            continue

        hazy = load_rgb_01(p, args.size).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
        m0 = compute_hazy_mass_map(
            hazy,
            input_range="0_1",  # 这里明确 input 是 [0,1]
            dcp_patch_size=args.dcp_patch_size,
            dcp_omega=args.dcp_omega,
            dcp_top_percent=args.dcp_top_percent,
            alpha=args.alpha,
            beta=args.beta,
        )  # [1,1,H,W] in [0,1]

        m0_np = m0.squeeze(0).squeeze(0).clamp(0, 1).float().cpu().numpy()  # [H,W]
        np.save(out_path, m0_np)

    print(f"Done. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
