# haze_mass/dcp.py

from typing import Tuple

import torch
import torch.nn.functional as F


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


def dark_channel(
    img: torch.Tensor,
    patch_size: int = 15
) -> torch.Tensor:
    """
    Dark Channel: min_c min_{y∈Ω(x)} I^c(y)

    Args:
        img: [B, 3, H, W] in [0,1]
        patch_size: 局部窗口大小

    Returns:
        dark: [B,1,H,W] in [0,1]
    """
    img = to_4d(img)  # [B,C,H,W]
    # 先在通道维取最小 → [B,1,H,W]
    dc = img.min(dim=1, keepdim=True)[0]

    # 再做局部最小：用 -max_pool 实现 erosion
    pad = patch_size // 2
    dc_pad = F.pad(dc, (pad, pad, pad, pad), mode="reflect")
    dark = -F.max_pool2d(-dc_pad, kernel_size=patch_size, stride=1, padding=0)
    return dark


def estimate_atmospheric_light(
    img: torch.Tensor,
    dark: torch.Tensor,
    top_percent: float = 0.001
) -> torch.Tensor:
    """
    基于 dark channel 估计大气光 A。

    Args:
        img:   [B,3,H,W] in [0,1]
        dark:  [B,1,H,W] in [0,1]
        top_percent: 选 dark 最大的 top 百分比像素来估计 A

    Returns:
        A: [B,3,1,1]
    """
    img = to_4d(img)
    dark = to_4d(dark)

    b, _, h, w = img.shape
    flat_dark = dark.view(b, -1)        # [B,HW]
    flat_img = img.view(b, 3, -1)       # [B,3,HW]

    num_pixels = h * w
    k = max(1, int(num_pixels * top_percent))

    # 每张图里 dark 最大的 k 个位置
    vals, idx = torch.topk(flat_dark, k, dim=1, largest=True, sorted=False)  # [B,k]

    A_list = []
    for bi in range(b):
        indices = idx[bi]               # [k]
        rgb = flat_img[bi, :, indices]  # [3,k]
        A_list.append(rgb.mean(dim=1))  # [3]

    A = torch.stack(A_list, dim=0)      # [B,3]
    A = A.view(b, 3, 1, 1)
    return A


def estimate_transmission(
    img: torch.Tensor,
    A: torch.Tensor,
    patch_size: int = 15,
    omega: float = 0.95
) -> torch.Tensor:
    """
    透射率估计:
      t(x) ≈ 1 - ω * min_c min_{y∈Ω(x)} I^c(y) / A^c

    Args:
        img: [B,3,H,W] in [0,1]
        A:   [B,3,1,1]
        patch_size: 窗口
        omega: 控制保留雾的比例

    Returns:
        t: [B,1,H,W] in [0,1]
    """
    img = to_4d(img)
    A = A.view(-1, 3, 1, 1)

    # 归一化 I / A
    normed = img / (A + 1e-8)
    dark_normed = dark_channel(normed, patch_size=patch_size)  # [B,1,H,W]

    t = 1.0 - omega * dark_normed
    t = torch.clamp(t, 0.05, 1.0)  # 避免过小
    return t


def dcp_haze_strength(
    img: torch.Tensor,
    patch_size: int = 15,
    omega: float = 0.95,
    top_percent: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于 DCP 计算雾相关量：

    Args:
        img: [B,3,H,W] in [0,1]

    Returns:
        dark: [B,1,H,W] Dark Channel
        t:    [B,1,H,W] 透射率
        h:    [B,1,H,W] 雾强度 h = 1 - t
    """
    img = to_4d(img)
    dark = dark_channel(img, patch_size=patch_size)
    A = estimate_atmospheric_light(img, dark, top_percent=top_percent)
    t = estimate_transmission(img, A, patch_size=patch_size, omega=omega)
    h = 1.0 - t
    return dark, t, h
