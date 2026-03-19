# motivation/vis_roi_haze_diffusion_xywh_min.py
"""
ROI-based local haze injection + anisotropic diffusion visualization (minimal outputs)

Input: one image (treated as hazy RGB in [0,1])
Steps:
  1) compute M0 = f(DCP + DWT)
  2) build soft ROI mask from OpenCV (x,y,w,h)
  3) inject local haze on M0 -> M0_aug
  4) evolve M0_aug by anisotropic diffusion for T steps -> M0_after
  5) OPTIONAL: synthesize an "output.png" RGB by applying local haze to input
     (simple atmospheric scattering model with constant A)

Save ONLY:
  - input.png          (original RGB)
  - output.png         (RGB after adding local haze in ROI; for intuition)
  - M0.png             (paper-defined haze mass map from input)
  - M0_aug.png         (M0 after ROI haze injection)
  - M0_after.png       (after diffusion)
  - dM_diverging.png   (M0_after - M0_aug, diverging red/blue)
No curves/csv/extra plots.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from hazy_mass.hazy_mass import compute_hazy_mass_map

try:
    from MFlow.aniso_mflow import AnisotropicDiffusionStep
except Exception:
    from motivation.aniso_mflow import AnisotropicDiffusionStep


# -------------------- I/O helpers --------------------

def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _img_to_tensor_01(path: str) -> torch.Tensor:
    """Read RGB image -> [1,3,H,W] float in [0,1]."""
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0)


def _save_rgb_01(x_b3hw: torch.Tensor, path: str) -> None:
    """Save [1,3,H,W] in [0,1] as RGB png."""
    x = x_b3hw.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    arr = (x * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _norm01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.min()
    x = x / (x.max() + eps)
    return x


def _save_gray01_map(x_b1hw: torch.Tensor, path: str) -> None:
    """Save [1,1,H,W] as grayscale with per-image minmax."""
    x = _norm01(x_b1hw.detach())
    arr = (x[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_diverging(dm_b1hw: torch.Tensor, path: str, vlim: float | None = None) -> None:
    """
    Save ΔM diverging:
      positive -> red, negative -> blue
    """
    import matplotlib.pyplot as plt

    dm = dm_b1hw.detach()[0, 0].cpu().numpy()
    if vlim is None:
        vlim = float(np.quantile(np.abs(dm), 0.995) + 1e-8)
    vlim = max(vlim, 1e-6)

    plt.figure(figsize=(5, 5), dpi=200)
    plt.axis("off")
    plt.imshow(dm, cmap="seismic", vmin=-vlim, vmax=vlim)
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


# -------------------- mask + haze injection --------------------

def _build_soft_roi_mask(
    H: int, W: int,
    x1: int, y1: int, x2: int, y2: int,
    feather: int = 25,
    device="cpu"
) -> torch.Tensor:
    """
    Soft rectangular mask in [0,1] with smooth edges.
    """
    x1, x2 = sorted([max(0, x1), min(W, x2)])
    y1, y2 = sorted([max(0, y1), min(H, y2)])

    hard = torch.zeros((1, 1, H, W), device=device)
    hard[:, :, y1:y2, x1:x2] = 1.0

    if feather <= 0:
        return hard

    k = int(max(3, feather * 2 + 1))
    if k % 2 == 0:
        k += 1
    sigma = max(1e-6, feather / 2.0)

    xs = torch.arange(k, device=device).float() - (k - 1) / 2.0
    g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    g_x = g.view(1, 1, 1, k)
    g_y = g.view(1, 1, k, 1)

    pad = k // 2
    m = F.pad(hard, (pad, pad, pad, pad), mode="reflect")
    m = F.conv2d(m, g_x)
    m = F.conv2d(m, g_y)

    m = m / (m.max() + 1e-8)
    return m.clamp(0, 1)


def _inject_haze_on_M0(M0: torch.Tensor, mask: torch.Tensor, delta: float, adaptive: bool) -> torch.Tensor:
    """
    Inject local haze on M0 (haze mass map).
    If adaptive: M0 + delta*(1-M0)*mask (prevents saturation)
    Else:        M0 + delta*mask
    """
    if adaptive:
        return (M0 + float(delta) * (1.0 - M0) * mask).clamp(0, 1)
    return (M0 + float(delta) * mask).clamp(0, 1)


def _synthesize_rgb_local_haze(
    img01: torch.Tensor,
    mask: torch.Tensor,
    delta_rgb: float,
    A: float = 1.0
) -> torch.Tensor:
    """
    Create an RGB "output.png" by applying local haze in ROI using a simple scattering model:
      I_hazy = I * t + A * (1 - t)
    where t = 1 - delta_rgb * mask  (so larger delta -> smaller transmission -> more haze)

    This is ONLY for visualization intuition; the diffusion still happens on M0.
    """
    img01 = img01.clamp(0, 1)
    mask3 = mask.repeat(1, 3, 1, 1)

    # transmission in [t_min, 1]
    t = (1.0 - float(delta_rgb) * mask3).clamp(0.2, 1.0)
    A_rgb = torch.full_like(img01, float(A))
    return (img01 * t + A_rgb * (1.0 - t)).clamp(0, 1)


# -------------------- main --------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    # x=293, y=121, w=80, h=39
    # 6 93 68 67
    ap.add_argument("--img", type=str, default="/Users/baijingyuan/Desktop/result/8_hazy.png")
    ap.add_argument("--outdir", type=str, default="/Users/baijingyuan/Desktop/result/vis_roi_diffusion_min_1")
    ap.add_argument("--device", type=str, default="cuda")

    # ROI from OpenCV: (x, y, w, h)
    ap.add_argument("--roi_xywh", type=int, nargs=4, required=True, metavar=("x", "y", "w", "h"))

    # M0 (paper) params
    ap.add_argument("--dcp_patch_size", type=int, default=15)
    ap.add_argument("--dcp_omega", type=float, default=0.95)
    ap.add_argument("--dcp_top_percent", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)

    # local haze injection (on M0)
    ap.add_argument("--delta", type=float, default=0.27, help="M0 haze boost (recommend 0.10~0.20)")
    ap.add_argument("--adaptive", action="store_true", help="use M0 + delta*(1-M0)*mask (recommended)")
    ap.add_argument("--feather", type=int, default=25)

    # diffusion
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--base_kappa", type=float, default=0.03)
    ap.add_argument("--base_lambda", type=float, default=0.33)

    # RGB output haze (visual only)
    ap.add_argument("--delta_rgb", type=float, default=0.35,
                    help="strength of local haze applied to RGB output.png. "
                         "If not set, uses same value as --delta.")

    args = ap.parse_args()
    _ensure_dir(args.outdir)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) load RGB
    img01 = _img_to_tensor_01(args.img).to(dev)  # [1,3,H,W]
    H, W = img01.shape[-2], img01.shape[-1]
    _save_rgb_01(img01, os.path.join(args.outdir, "input.png"))

    # 2) compute M0
    M0 = compute_hazy_mass_map(
        hazy=img01,
        input_range="0_1",
        dcp_patch_size=args.dcp_patch_size,
        dcp_omega=args.dcp_omega,
        dcp_top_percent=args.dcp_top_percent,
        alpha=args.alpha,
        beta=args.beta,
    ).clamp(0, 1)  # [1,1,H,W]
    _save_gray01_map(M0, os.path.join(args.outdir, "M0.png"))

    # 3) ROI -> x1,y1,x2,y2
    x, y, w, h = args.roi_xywh
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    x1, x2 = sorted([max(0, x1), min(W, x2)])
    y1, y2 = sorted([max(0, y1), min(H, y2)])

    # 4) soft mask
    mask = _build_soft_roi_mask(H, W, x1, y1, x2, y2, feather=args.feather, device=dev)  # [1,1,H,W]

    # 5) inject haze on M0 -> M0_aug
    M0_aug = _inject_haze_on_M0(M0, mask, delta=args.delta, adaptive=args.adaptive)
    _save_gray01_map(M0_aug, os.path.join(args.outdir, "M0_aug.png"))

    # 6) synthesize RGB "output.png" (local haze)
    delta_rgb = float(args.delta if args.delta_rgb is None else args.delta_rgb)
    out_rgb = _synthesize_rgb_local_haze(img01, mask, delta_rgb=delta_rgb, A=1.0)
    _save_rgb_01(out_rgb, os.path.join(args.outdir, "output.png"))

    # 7) diffusion on M0_aug
    step = AnisotropicDiffusionStep(
        num_iters=1,
        base_kappa=args.base_kappa,
        base_lambda=args.base_lambda,
        guided_by_feat=False,
        feat_channels=None,
    ).to(dev).eval()

    M = M0_aug.clone()
    for _ in range(args.T):
        M = step(M).clamp(0, 1)
    M0_after = M
    _save_gray01_map(M0_after, os.path.join(args.outdir, "M0_after.png"))

    # 8) dM diverging (after - aug)
    dM = M0_after - M0_aug
    _save_diverging(dM, os.path.join(args.outdir, "dM_diverging.png"))

    print(f"Saved: input.png, output.png, M0.png, M0_aug.png, M0_after.png, dM_diverging.png -> {args.outdir}")
    print(f"ROI (x,y,w,h)=({x},{y},{w},{h}) -> (x1,y1,x2,y2)=({x1},{y1},{x2},{y2})")
    print(f"delta(M0)={args.delta}, adaptive={args.adaptive}, feather={args.feather}")
    print(f"delta_rgb(output)={delta_rgb}")
    print(f"diffusion: T={args.T}, kappa={args.base_kappa}, lambda={args.base_lambda}")


if __name__ == "__main__":
    main()

# 6, 93, 68, 67