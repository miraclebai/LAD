# haze_mass/dwt.py

import torch
import torch.nn as nn


def dwt_init(x: torch.Tensor):
    """
    参考你提供的 DWT 实现：
      x01 = x[:, :, 0::2, :] / 2
      x02 = x[:, :, 1::2, :] / 2
      ...
      x_LL, x_HL, x_LH, x_HH

    Args:
        x: [B,C,H,W]

    Returns:
        x_LL: [B,C,H/2,W/2]
        x_H_cat: [B,3C,H/2,W/2]  (concat(HL,LH,HH) at channel dim)
    """
    if x.dim() != 4:
        raise ValueError(f"[DWT] Expected 4D tensor [B,C,H,W], got {x.shape}")

    # [B,C,H,W]
    # 下采样行方向：偶数行 / 奇数行
    x01 = x[:, :, 0::2, :] / 2.0   # [B,C,H/2,W]
    x02 = x[:, :, 1::2, :] / 2.0   # [B,C,H/2,W]

    # 再在列方向下采样：偶数列 / 奇数列
    x1 = x01[:, :, :, 0::2]        # [B,C,H/2,W/2]
    x2 = x02[:, :, :, 0::2]        # [B,C,H/2,W/2]
    x3 = x01[:, :, :, 1::2]        # [B,C,H/2,W/2]
    x4 = x02[:, :, :, 1::2]        # [B,C,H/2,W/2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    x_H_cat = torch.cat((x_HL, x_LH, x_HH), dim=1)  # [B,3C,H/2,W/2]
    return x_LL, x_H_cat


class DWT(nn.Module):
    """
    非学习的 DWT 核心（和你给的 DWT 一致，只是封装成 nn.Module）
    """

    def __init__(self):
        super().__init__()
        # 核心 dwt_init 无参数，所以不需要 requires_grad
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        return dwt_init(x)


class DWTTransform(nn.Module):
    """
    参考你给的 DWT_transform，实现一个可学习的 DWT 特征变换模块：

      输入 x: [B,in_channels,H,W]
      输出:
        low:  [B,out_channels,H/2,W/2]   (由 x_LL 经 1x1 conv 得到)
        high: [B,out_channels,H/2,W/2]   (由 [HL,LH,HH] concat 后经 1x1 conv 得到)

    你后续如果想在 Encoder 里用 DWT 特征（不只是做 M0），可以直接用这个模块。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.conv1x1_high = nn.Conv2d(
            in_channels * 3, out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        x_LL, x_H_cat = self.dwt(x)              # [B,C,H/2,W/2], [B,3C,H/2,W/2]
        low = self.conv1x1_low(x_LL)             # [B,out,H/2,W/2]
        high = self.conv1x1_high(x_H_cat)        # [B,out,H/2,W/2]
        return low, high
