import torch
import torch.nn as nn
import torch.nn.functional as F


class GuideOutputAffine(nn.Module):
    """
    最稳的 guide->输出校正头：
      x' = x * (1 + gamma) + beta

    - x: (B,3,H,W)  decoder 的输出（通常范围 [-1,1]）
    - guide: (B,G,Hg,Wg) 或 list[g0,g1,...]（默认取 g0）
    - 输出: (B,3,H,W)

    设计要点：
    - 最小耦合：只依赖 guide 的通道数 G，不依赖 decoder 中间层结构
    - zero-init：初始 gamma,beta ~ 0，行为≈恒等映射，训练非常稳
    """
    def __init__(self, guide_channels: int, hidden_channels: int = 128, out_channels: int = 3):
        super().__init__()
        self.guide_channels = int(guide_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)

        # 轻量 MLP-in-conv：G -> hidden -> 2*out_channels
        g = min(32, hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.guide_channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(g, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 2 * self.out_channels, kernel_size=1),
        )

        # zero-init 最后一层：初始为 identity
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, guide):
        if guide is None:
            return x

        if isinstance(guide, (list, tuple)):
            # 最稳：只用最高分辨率 g0
            if len(guide) == 0:
                return x
            guide = guide[0]

        if not torch.is_tensor(guide):
            raise TypeError(f"guide must be Tensor or list[Tensor], got {type(guide)}")

        if x.dim() != 4 or x.size(1) != self.out_channels:
            raise ValueError(f"x must be (B,{self.out_channels},H,W), got {tuple(x.shape)}")
        if guide.dim() != 4 or guide.size(1) != self.guide_channels:
            raise ValueError(f"guide must be (B,{self.guide_channels},Hg,Wg), got {tuple(guide.shape)}")

        B, C, H, W = x.shape

        # 对齐空间分辨率
        if guide.shape[-2:] != (H, W):
            guide = F.interpolate(guide, size=(H, W), mode="bilinear", align_corners=False)

        gb = self.net(guide)  # (B, 2C, H, W)
        gamma, beta = torch.chunk(gb, 2, dim=1)

        return x * (1.0 + gamma) + beta
