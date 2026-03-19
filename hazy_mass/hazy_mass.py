# haze_mass/hazy_mass.py

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from hazy_mass.dcp import dcp_haze_strength, to_4d
from hazy_mass.dwt import dwt_init


InputRange = Literal["-1_1", "0_1"]




def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    """
    确保图像为 3 通道 [B,3,H,W]。
    若输入为 [B,1,H,W]，则复制三次。
    """
    x = to_4d(x)
    b, c, h, w = x.shape
    if c == 3:
        return x
    elif c == 1:
        return x.repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"[HazyMass] Expected 1 or 3 channels, got {c}.")


def _to_01(x: torch.Tensor, input_range: InputRange) -> torch.Tensor:
    """
    把输入图像统一变到 [0,1]，与 Dataset 对齐：
      - Dataset 输出 hazy/clear ∈ [-1,1]
      - 这里如果 input_range='-1_1'，则先映射到 [0,1]
    """
    if input_range == "-1_1":
        return (x + 1.0) * 0.5
    elif input_range == "0_1":
        return x
    else:
        raise ValueError(f"[HazyMass] Unsupported input_range: {input_range}")


def _rgb_to_luminance(img: torch.Tensor) -> torch.Tensor:
    """
    简单 RGB -> Y 亮度 (BT.601/Rec.709 风格)：
      Y = 0.299 R + 0.587 G + 0.114 B
    输入 img: [B,3,H,W] in [0,1]
    返回 Y:  [B,1,H,W]
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def _min_max_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对每个 batch 的 map 做 min-max 归一化到 [0,1]。
    x: [B,1,H,W]
    """
    b = x.shape[0]
    x_flat = x.view(b, -1)
    x_min = x_flat.min(dim=1, keepdim=True)[0]
    x_max = x_flat.max(dim=1, keepdim=True)[0]
    x_norm = (x_flat - x_min) / (x_max - x_min + eps)
    return x_norm.view_as(x)


def compute_hazy_mass_map(
    hazy: torch.Tensor,
    input_range: InputRange = "-1_1",
    dcp_patch_size: int = 15,
    dcp_omega: float = 0.95,
    dcp_top_percent: float = 0.001,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    """
    计算初始 hazy mass map M0 = f(DCP, DWT)：

      1) 利用 DCP 得到雾强度 h_dcp (1 - t)
      2) 利用 DWT 在亮度通道提取低频 LL 和高频能量 E
      3) 融合成 M0，并归一化到 [0,1]

    Args:
        hazy: 输入带雾图像，shape [B,3,H,W] 或兼容形状。
              若 input_range='-1_1'，则假定值域在 [-1,1]（与当前 Dataset 对齐）。
              若 input_range='0_1'，则假定已经在 [0,1]。
        input_range: "-1_1" 或 "0_1"。
        dcp_patch_size: DCP 的局部窗口大小。
        dcp_omega: DCP 里的 ω，控制透射率估计强度。
        dcp_top_percent: A 估计时选取的 top 比例。
        alpha: DCP 雾强度在融合中的权重。
        beta: 高频抑制项权重（越大，对结构边缘越“减雾”）。

    Returns:
        M0: [B,1,H,W], in [0,1]，表示初始 hazy mass map。
    """
    # 1) 统一到 [0,1]，并保证 3 通道
    hazy = _ensure_3ch(hazy)                  # [B,3,H,W]
    hazy_01 = _to_01(hazy, input_range)       # [0,1]

    # 2) DCP-based 雾强度 h_dcp
    _, t, h_dcp = dcp_haze_strength(
        hazy_01,
        patch_size=dcp_patch_size,
        omega=dcp_omega,
        top_percent=dcp_top_percent,
    )  # h_dcp: [B,1,H,W] in [0,1]

    # 3) DWT on luminance
    y = _rgb_to_luminance(hazy_01)            # [B,1,H,W]
    B, C, H, W = y.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h != 0 or pad_w != 0:
        y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
    # dwt_init 期望 [B,C,H,W]，这里 C=1
    LL, H_cat = dwt_init(y)                   # [B,1,H/2,W/2], [B,3,H/2,W/2]

    # 将低频上采样回原分辨率，便于和 h_dcp 融合
    _, _, H, W = hazy_01.shape
    LL_up = F.interpolate(LL, size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]

    # 高频能量：LH/HL/HH 的 L1 能量，用于抑制边缘处“误判雾”
    # H_cat: [B,3,H/2,W/2]，对通道求 L1 加和
    E = H_cat.abs().sum(dim=1, keepdim=True)  # [B,1,H/2,W/2]
    E_up = F.interpolate(E, size=(H, W), mode="bilinear", align_corners=False)    # [B,1,H,W]

    # 4) 归一化 LL_up / E_up
    LL_norm = _min_max_norm(LL_up)            # [B,1,H,W]
    E_norm = _min_max_norm(E_up)              # [B,1,H,W]

    # 5) 融合：
    #    - h_dcp: DCP 估计的雾强度
    #    - LL_norm: 大尺度亮度 / 低频雾分布
    #    - E_norm: 高频结构，边缘处不希望 M 太大（避免把结构当雾）
    #
    #    一个简单且直观的形式：
    #      M = α * h_dcp + (1-α) * LL_norm - β * E_norm
    M = alpha * h_dcp + (1.0 - alpha) * LL_norm - beta * E_norm

    # 6) 截断到 [0,1]
    M = torch.clamp(M, 0.0, 1.0)               # [B,1,H,W]
    return M


class HazyMassMapGenerator(nn.Module):
    """
    一个 nn.Module 封装，用于在网络中直接调用：
        M0 = hazy_mass_generator(hazy)

    默认假定 hazy 的值域为 [-1,1]，与当前 PairedDataset 保持一致。
    """

    def __init__(
        self,
        input_range: InputRange = "-1_1",
        dcp_patch_size: int = 15,
        dcp_omega: float = 0.95,
        dcp_top_percent: float = 0.001,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_range = input_range
        self.dcp_patch_size = dcp_patch_size
        self.dcp_omega = dcp_omega
        self.dcp_top_percent = dcp_top_percent
        self.alpha = alpha
        self.beta = beta

    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        """
        hazy: [B,3,H,W] or compatible, value range depends on input_range.
        返回: [B,1,H,W] in [0,1]
        """
        return compute_hazy_mass_map(
            hazy=hazy,
            input_range=self.input_range,
            dcp_patch_size=self.dcp_patch_size,
            dcp_omega=self.dcp_omega,
            dcp_top_percent=self.dcp_top_percent,
            alpha=self.alpha,
            beta=self.beta,
        )
