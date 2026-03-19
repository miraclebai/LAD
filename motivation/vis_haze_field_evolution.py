# motivation/vis_haze_field_evolution.py
"""
Motivation visualization (paper-friendly metrics):
- Use paper-defined haze mass map M0 = f(DCP + DWT) from hazy_mass/hazy_mass.py
- Evolve M by anisotropic diffusion for T steps (no network features, no conductance C)

Outputs:
  input.png
  M0.png
  M_after.png
  dM_diverging.png
  grid_M.png
  stats.txt

Statistics:
  A) Peak decay (max) percent change
  B) R: Area shrink ratio above a fixed high-haze threshold (defined on M0):
        R = |{M_after >= thr}| / |{M0 >= thr}| * 100, thr = quantile(M0, q_high)
  C) Top-p strong-haze region decreasing fraction (region defined on M0):
        #{x in S: M_after(x) < M0(x)} / |S| * 100, S = top_p pixels in M0
  D) Low quantile (q_low) percent change (weak haze lift)
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from hazy_mass.hazy_mass import compute_hazy_mass_map
from motivation.aniso_mflow import AnisotropicDiffusionStep


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _img_to_tensor_01(path: str) -> torch.Tensor:
    """Read image -> [1,3,H,W] float in [0,1]."""
    img = Image.open(path).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0)
    return x


def _norm01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.min()
    x = x / (x.max() + eps)
    return x


def _save_gray_01(x_b1hw: torch.Tensor, path: str) -> None:
    """Save [B,1,H,W] with B=1 as grayscale png using per-image minmax."""
    x = x_b1hw.detach()
    x = _norm01(x)
    arr = (x[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_diverging_dm(dm_b1hw: torch.Tensor, path: str, vlim: float | None = None) -> None:
    """
    Save ΔM = M_after - M0 as diverging red-blue image.
    Positive: red (haze increases), Negative: blue (haze decreases).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _save_gray_01(dm_b1hw.abs(), path.replace(".png", "_abs_gray.png"))
        return

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


def _make_grid_png(tensors_bchw: list[torch.Tensor], nrow: int, path: str) -> None:
    """tensors_bchw: list of [1,3,H,W] in [0,1]"""
    tiles = [t[0] for t in tensors_bchw]  # -> [3,H,W]
    grid = make_grid(tiles, nrow=nrow, padding=2)
    arr = (grid.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _pct_change(before: float, after: float, eps: float = 1e-6) -> float:
    """Percent change: (after - before) / (before + eps) * 100"""
    return (after - before) / (before + eps) * 100.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default="/Users/baijingyuan/Desktop/result/09_indoor_hazy.jpg")
    ap.add_argument("--outdir", type=str, default="vis_motivation")
    ap.add_argument("--device", type=str, default="cuda")

    # diffusion evolution
    ap.add_argument("--T", type=int, default=24, help="number of outer diffusion steps")
    ap.add_argument("--base_kappa", type=float, default=0.03)
    ap.add_argument("--base_lambda", type=float, default=0.25)

    # paper-defined M0 parameters (must match your paper)
    ap.add_argument("--dcp_patch_size", type=int, default=15)
    ap.add_argument("--dcp_omega", type=float, default=0.95)
    ap.add_argument("--dcp_top_percent", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)

    # strong haze region S for decreasing fraction
    ap.add_argument("--top_p", type=float, default=0.15, help="top-p region for strong haze (defined on M0)")
    # weak haze shift
    ap.add_argument("--q_low", type=float, default=0.15, help="low quantile for weak haze shift")
    # R: area shrink threshold quantile
    ap.add_argument("--q_high", type=float, default=0.99, help="high quantile in M0 defining threshold for area shrink")

    args = ap.parse_args()

    _ensure_dir(args.outdir)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) load hazy image in [0,1]
    hazy01 = _img_to_tensor_01(args.img).to(dev)  # [1,3,H,W]
    Image.open(args.img).convert("RGB").save(os.path.join(args.outdir, "input.png"))

    # 2) compute paper-defined M0 (DCP + DWT fusion)
    M0 = compute_hazy_mass_map(
        hazy=hazy01,
        input_range="0_1",
        dcp_patch_size=args.dcp_patch_size,
        dcp_omega=args.dcp_omega,
        dcp_top_percent=args.dcp_top_percent,
        alpha=args.alpha,
        beta=args.beta,
    ).clamp(0, 1)  # [1,1,H,W]

    # 3) diffusion evolution (pure, no features)
    step = AnisotropicDiffusionStep(
        num_iters=1,
        base_kappa=args.base_kappa,
        base_lambda=args.base_lambda,
        guided_by_feat=False,
        feat_channels=None,
    ).to(dev).eval()

    M = M0.clone()
    Ms = [M0.clone()]
    for _ in range(args.T):
        M = step(M).clamp(0, 1)
        Ms.append(M.clone())

    M_after = Ms[-1]
    dM = M_after - M0

    # 4) save maps
    _save_gray_01(M0, os.path.join(args.outdir, "M0.png"))
    _save_gray_01(M_after, os.path.join(args.outdir, "M_after.png"))
    _save_diverging_dm(dM, os.path.join(args.outdir, "dM_diverging.png"))

    # 5) grid of M evolution
    Ms_rgb = [_norm01(m).repeat(1, 3, 1, 1) for m in Ms]  # [1,3,H,W]
    _make_grid_png(Ms_rgb, nrow=len(Ms_rgb), path=os.path.join(args.outdir, "grid_M.png"))

    # 6) metrics
    m0 = M0.view(-1)
    ma = M_after.view(-1)

    # A) peak decay
    max0 = float(m0.max().item())
    maxA = float(ma.max().item())
    peak_decay_pct = _pct_change(max0, maxA)  # negative means decay

    # B) R: area shrink ratio above fixed threshold (thr defined on M0)
    thr_high = float(torch.quantile(m0, args.q_high).item())
    area0 = int((m0 >= thr_high).sum().item())
    areaA = int((ma >= thr_high).sum().item())
    R = (areaA / max(area0, 1)) * 100.0  # percent
    area_shrink_pct = 100.0 - R          # percent shrink vs baseline

    # C) decreasing fraction inside S (top_p defined on M0)
    thr_S = float(torch.quantile(m0, 1.0 - args.top_p).item())
    S = (m0 >= thr_S)
    dec_frac = float((ma[S] < m0[S]).float().mean().item()) * 100.0 if S.any() else float("nan")

    # D) low-quantile shift (weak haze lift)
    q0 = float(torch.quantile(m0, args.q_low).item())
    qA = float(torch.quantile(ma, args.q_low).item())
    qlow_pct = _pct_change(q0, qA)  # positive means lift

    stats: list[str] = []
    stats.append(f"Input: {args.img}")
    stats.append(f"T (outer steps): {args.T}")
    stats.append(f"Diffusion params: kappa={args.base_kappa}, lambda={args.base_lambda}")
    stats.append("")
    stats.append("[Metrics]")
    stats.append(f"A) Peak decay (max): M0={max0:.6f} -> M_after={maxA:.6f} | Δ% = {peak_decay_pct:.3f}%  (negative=decay)")
    stats.append("")
    stats.append(f"[R: Area shrink above high-haze threshold (thr from M0 quantile q{args.q_high:.2f})]")
    stats.append(f"thr_high = {thr_high:.6f}")
    stats.append(f"area(M0 >= thr_high)      = {area0}")
    stats.append(f"area(M_after >= thr_high) = {areaA}")
    stats.append(f"B) R = area_after / area_0 = {R:.2f}%  (smaller=more shrink)")
    stats.append(f"   Area shrink = {area_shrink_pct:.2f}%  (100%-R)")
    stats.append("")
    stats.append(f"[Strong haze region S: Top {int(args.top_p*100)}% of M0 (fixed region)]")
    stats.append(f"C) Decreasing fraction in S: P(M_after < M0 | x in S) = {dec_frac:.2f}%")
    stats.append("")
    stats.append(f"[Weak haze shift]")
    stats.append(f"D) Low quantile q{args.q_low:.2f} shift: M0={q0:.6f} -> M_after={qA:.6f} | Δ% = {qlow_pct:.3f}%  (positive=lift)")

    with open(os.path.join(args.outdir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stats))

    print("\n".join(stats))
    print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
