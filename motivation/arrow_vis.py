import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def to_4d(x: torch.Tensor) -> torch.Tensor:
    """
    保证输入为 [B, C, H, W] 形状。
    支持 [C,H,W] / [H,W] 输入。
    """
    if x.dim() == 2:          # H, W
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:        # C, H, W
        x = x.unsqueeze(0)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"[DCP] Unsupported tensor shape: {x.shape}")
    return x


__all__ = [
    "AnisotropicDiffusionStep",
    "HazeFeatureModulation",
    "HazeMFlowUnit",
]




def _ensure_same_spatial(m: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """
    确保雾图 M 和 feature F 在空间尺寸上匹配。
    如不匹配，则用双线性插值将 M 调整到 feat 的 H,W。
    """
    m = to_4d(m)
    feat = to_4d(feat)
    _, _, H_f, W_f = feat.shape
    _, _, H_m, W_m = m.shape
    if (H_f, W_f) != (H_m, W_m):
        m = F.interpolate(m, size=(H_f, W_f), mode="bilinear", align_corners=False)
    return m


class AnisotropicDiffusionStep(nn.Module):
    r"""
    各向异性扩散 (Perona-Malik 风格) 对 hazy mass map M 的更新模块。

    PDE 形式（离散近似）：
      ∂M/∂t = div( c(x) ∇M )

    离散更新：
      M_new(i,j) = M(i,j) + λ * [ cN * (M_N - M) + cS * (M_S - M)
                                + cE * (M_E - M) + cW * (M_W - M) ]

      其中 cN = g(|M_N - M| / kappa)，g 是边缘抑制函数（例如 exp(-s^2)）。

    特性：
      - 在平坦区域（|∇M|小），c ≈ 1，扩散强，局部平滑；
      - 在边缘/物体边界（|∇M|大），c ≈ 0，扩散弱，保持边界。

    可选引导：
      - 若提供 feature F，则使用 F 的全局统计适度调整 kappa/lambda，
        使得在某些特征分布下扩散更强/更弱。
    """

    def __init__(
        self,
        num_iters: int = 3,
        base_kappa: float = 0.1,
        base_lambda: float = 0.2,
        guided_by_feat: bool = True,
        feat_channels: Optional[int] = None,
        guide_reduction: int = 16,
    ) -> None:
        """
        Args:
            num_iters: 迭代步数（扩散次数），每步都是一个小时间步。
            base_kappa: 控制边缘敏感度的基础参数，越小越容易把小梯度当作边缘。
            base_lambda: 时间步长 λ，越大扩散越快。理论上 2D 4 邻域下建议 <= 0.25。
            guided_by_feat: 是否使用特征 F 作为引导，调整 kappa / lambda。
            feat_channels: 若 guided_by_feat=True，则必须指定 F 的通道数 C_F。
            guide_reduction: 引导 MLP 的 hidden 通道缩减比例。
        """
        super().__init__()
        assert num_iters > 0, "num_iters should be positive."
        self.num_iters = num_iters
        self.base_kappa = float(base_kappa)
        self.base_lambda = float(base_lambda)

        self.guided_by_feat = guided_by_feat and (feat_channels is not None)
        if self.guided_by_feat:
            hidden = max(4, feat_channels // guide_reduction)
            self.guide_mlp = nn.Sequential(
                nn.Linear(feat_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2),  # 输出两个标量：Δκ, Δλ
            )
        else:
            self.guide_mlp = None

    def _compute_kappa_lambda(
        self, m: torch.Tensor, feat: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算每个样本的有效 kappa 和 lambda。
        若不使用引导，则所有样本共享常数。
        若使用引导，则根据 feature 全局统计做轻微缩放。

        Returns:
            kappa_eff: [B,1,1,1]
            lambda_eff: [B,1,1,1]
        """
        B = m.shape[0]

        # 默认常数形式
        kappa = torch.full(
            (B, 1, 1, 1), self.base_kappa, device=m.device, dtype=m.dtype
        )
        lam = torch.full(
            (B, 1, 1, 1), self.base_lambda, device=m.device, dtype=m.dtype
        )

        if self.guided_by_feat and feat is not None:
            # 利用 F 的全局统计调整 kappa / lambda（轻微缩放，防止不稳定）
            feat = to_4d(feat)
            # GAP over H,W -> [B, C_F]
            desc = F.adaptive_avg_pool2d(feat.abs(), output_size=1).view(B, -1)
            deltas = self.guide_mlp(desc)  # [B,2]
            # 使用 tanh 压缩在 [-1,1]，然后缩放到 [0.5,1.5] 附近
            scale = 1.0 + 0.3 * torch.tanh(deltas)  # [B,2], 大约在 [0.7,1.3]
            scale_kappa = scale[:, 0:1].view(B, 1, 1, 1)
            scale_lam = scale[:, 1:2].view(B, 1, 1, 1)

            kappa = kappa * scale_kappa
            lam = lam * scale_lam

        # 对 lambda 做额外 clamp，避免过大导致数值不稳定
        lam = lam.clamp(min=0.01, max=0.25)
        # kappa 一般不需要太严的 clamp，但避免为负
        kappa = kappa.clamp(min=1e-4)

        return kappa, lam


    @staticmethod
    def _diffusion_update(
        m: torch.Tensor,
        kappa: torch.Tensor,
        lam: torch.Tensor,
        eps: float = 1e-8,
        return_debug: bool = False,   # NEW
    ):
        B, C, H, W = m.shape
        assert C == 1

        m_pad = F.pad(m, (1, 1, 1, 1), mode="reflect")

        center = m_pad[:, :, 1:-1, 1:-1]
        north  = m_pad[:, :, 0:-2, 1:-1]
        south  = m_pad[:, :, 2:,   1:-1]
        west   = m_pad[:, :, 1:-1, 0:-2]
        east   = m_pad[:, :, 1:-1, 2:]

        dN = north - center
        dS = south - center
        dW = west  - center
        dE = east  - center

        gN = torch.exp(- (dN.abs() / (kappa + eps)) ** 2)
        gS = torch.exp(- (dS.abs() / (kappa + eps)) ** 2)
        gW = torch.exp(- (dW.abs() / (kappa + eps)) ** 2)
        gE = torch.exp(- (dE.abs() / (kappa + eps)) ** 2)

        update = gN * dN + gS * dS + gW * dW + gE * dE
        m_new = center + lam * update

        if not return_debug:
            return m_new

        # ✅ 离散通量（来自真实更新项）
        Fx = (gE * dE) - (gW * dW)
        Fy = (gS * dS) - (gN * dN)

        debug = {
            "M": center,      # 当前步中心（用于画热力图）
            "Fx": Fx,
            "Fy": Fy,
            "cE": gE, "cW": gW, "cN": gN, "cS": gS,
            "kappa": kappa,
            "lam": lam,
        }
        return m_new, debug

    def forward(self, m, feat=None, return_debug: bool = False, debug_last_only: bool = True):
        m = to_4d(m)
        kappa, lam = self._compute_kappa_lambda(m, feat)

        m_new = m
        debug_list = []
        for _ in range(self.num_iters):
            if return_debug:
                m_new, dbg = self._diffusion_update(m_new, kappa, lam, return_debug=True)
                debug_list.append(dbg)
            else:
                m_new = self._diffusion_update(m_new, kappa, lam)

        m_new = torch.clamp(m_new, 0.0, 1.0)

        if not return_debug:
            return m_new

        if debug_last_only:
            return m_new, debug_list[-1]  # 只要最后一步，最干净
        return m_new, debug_list



def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def load_mass(path):
    M = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if M.max() > 1.5:
        M /= 255.0
    return np.clip(M, 0, 1)

def crop_np(arr, roi):
    x, y, w, h = roi
    return arr[y:y+h, x:x+w]

def crop_t(t, roi):
    x, y, w, h = roi
    return t[..., y:y+h, x:x+w]

@torch.no_grad()
def get_flux_from_real_impl(mass_01: np.ndarray, num_iters=3, kappa=0.1, lam=0.2):
    """调用真实 AnisotropicDiffusionStep，拿最后一步 debug（Fx,Fy,M）。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = AnisotropicDiffusionStep(
        num_iters=num_iters,
        base_kappa=kappa,
        base_lambda=lam,
        guided_by_feat=False,
        feat_channels=None,
    ).to(device).eval()

    m = torch.from_numpy(mass_01).float()[None, None].to(device)
    m_out, dbg = diff(m, feat=None, return_debug=True, debug_last_only=True)

    M = dbg["M"][0, 0].detach().cpu().numpy()
    Fx = dbg["Fx"][0, 0].detach().cpu().numpy()
    Fy = dbg["Fy"][0, 0].detach().cpu().numpy()
    # 用平均导通系数做一个“结构阻断mask”（可选）
    c = (dbg["cE"] + dbg["cW"] + dbg["cN"] + dbg["cS"]) / 4.0
    c = c[0, 0].detach().cpu().numpy()
    return M, Fx, Fy, c

def draw_inset_flux_on_rgb(
    rgb, M, Fx, Fy, c,
    roi=(40, 40, 120, 120),
    inset_loc="upper right",
    out="flux_inset.png",
    cmap="inferno",
    step=16,              # 建议 12~20，越大越干净（块越大）
    topq=0.90,            # 建议 0.85~0.95，只保留最强通量块
    c_thr=0.40,           # 建议 0.30~0.50，边缘(导通小)不画
    arrow_len=18,         # 箭头显示长度（像素单位）
    roi_box_color=(0, 1, 0),
    roi_box_lw=2.5,
    inset_size="38%",     # inset 大小
    interpolation="bilinear",
):
    """
    主图：原RGB + ROI框
    inset：ROI内的 M 热力图 + 真实通量箭头（块级聚合 + 稀疏筛选 + 导通阈值）
    关键改动：不再逐点画箭头，而是每个 step×step 块只画一个“代表箭头”，显著减少杂乱。
    """
    # ---- ROI crop ----
    x, y, w, h = roi
    M_roi  = crop_np(M,  roi)
    Fx_roi = crop_np(Fx, roi)
    Fy_roi = crop_np(Fy, roi)
    c_roi  = crop_np(c,  roi)

    # ---- block aggregation（每 step×step 块聚合成一个代表向量）----
    H, W = M_roi.shape
    h2 = (H // step) * step
    w2 = (W // step) * step

    # ROI 太小会导致 block 不足，直接兜底：缩小 step
    if h2 < step or w2 < step:
        # 最少保证 2×2 个 block
        step = max(4, min(step, H // 2, W // 2))
        h2 = (H // step) * step
        w2 = (W // step) * step

    M_roi  = M_roi[:h2, :w2]
    Fx_roi = Fx_roi[:h2, :w2]
    Fy_roi = Fy_roi[:h2, :w2]
    c_roi  = c_roi[:h2, :w2]

    # reshape to blocks: [Bh, step, Bw, step]
    Bh = h2 // step
    Bw = w2 // step

    Fx_blk = Fx_roi.reshape(Bh, step, Bw, step)
    Fy_blk = Fy_roi.reshape(Bh, step, Bw, step)
    c_blk  = c_roi.reshape(Bh, step, Bw, step)

    mag_blk = np.sqrt(Fx_blk**2 + Fy_blk**2)

    # 用 magnitude 做权重：块内强通量像素贡献更大
    wgt = mag_blk / (mag_blk.sum(axis=(1, 3), keepdims=True) + 1e-8)

    Fx_rep = (Fx_blk * wgt).sum(axis=(1, 3))   # [Bh, Bw]
    Fy_rep = (Fy_blk * wgt).sum(axis=(1, 3))
    c_rep  = c_blk.mean(axis=(1, 3))
    mag_rep = np.sqrt(Fx_rep**2 + Fy_rep**2)

    # ---- 筛选：只保留最强通量块 + 导通系数较大（突出“结构阻断”）----
    thr = np.quantile(mag_rep, topq)
    mask = (mag_rep >= thr) & (c_rep >= c_thr)

    # 如果筛掉太多导致几乎没箭头：自动放宽一点（防止空图）
    if mask.sum() < 3:
        thr = np.quantile(mag_rep, min(0.80, topq))
        mask = (mag_rep >= thr) & (c_rep >= max(0.20, c_thr - 0.15))

    # ---- 单位化方向 + 固定显示长度（更干净）----
    U = np.zeros_like(Fx_rep)
    V = np.zeros_like(Fy_rep)
    U[mask] = Fx_rep[mask] / (mag_rep[mask] + 1e-8) * arrow_len
    V[mask] = Fy_rep[mask] / (mag_rep[mask] + 1e-8) * arrow_len

    # 每个 block 的中心坐标（像素坐标）
    yy, xx = np.mgrid[0:Bh, 0:Bw]
    xx = xx * step + step * 0.5
    yy = yy * step + step * 0.5

    # ---- 绘图：主图 RGB + ROI 框 + inset ----
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    ax.imshow(rgb)
    ax.axis("off")

    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=roi_box_lw,
        edgecolor=roi_box_color,
        facecolor="none"
    )
    ax.add_patch(rect)

    axins = inset_axes(ax, width=inset_size, height=inset_size, loc=inset_loc, borderpad=1.0)
    axins.imshow(M_roi, cmap=cmap, vmin=0, vmax=1, interpolation=interpolation)

    axins.quiver(
        xx[mask], yy[mask], U[mask], V[mask],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.012,
        headwidth=4.8,
        headlength=6.5,
        headaxislength=5.2,
        alpha=0.95
    )

    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor(roi_box_color)
        spine.set_linewidth(roi_box_lw)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("saved:", out)


if __name__ == "__main__":
    # 你自己的路径
    rgb_path = "/Users/baijingyuan/Desktop/result/vis_roi_diffusion_min/output.png"
    mass_path = "/Users/baijingyuan/Desktop/result/vis_roi_diffusion_min/M0_aug.png"

    rgb = load_rgb(rgb_path)
    mass = load_mass(mass_path)

    M, Fx, Fy, c = get_flux_from_real_impl(mass, num_iters=3, kappa=0.1, lam=0.2)

    # ✅ 你要的：只显示一个区域（ROI）
    roi = (293, 121, 80, 39)  # 改成你想看的区域 (x,y,w,h)

    draw_inset_flux_on_rgb(
        rgb, M, Fx, Fy, c,
        roi=roi,
        inset_loc="upper right",
        out="flux_inset.png",
        cmap="inferno",   # 红黄热力风格（推荐）
        step=10,          # 箭头稀疏度（更大更干净）
        topq=0.78,        # 只保留更强的 22%（减少杂乱）
        c_thr=0.35,       # 结构边缘处（c小）不画
        arrow_len=18
    )