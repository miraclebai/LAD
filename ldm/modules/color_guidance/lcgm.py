import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to luma-like Y (approx. BT.601).
    Input: rgb in either [0,1] or [-1,1], shape (B,3,H,W)
    Output: y in [-1,1] if input in [-1,1], else in [0,1], shape (B,1,H,W)
    """
    if rgb.dim() != 4 or rgb.size(1) != 3:
        raise ValueError(f"Expected rgb (B,3,H,W), got {tuple(rgb.shape)}")

    # weights as constants; do NOT force YCbCr round-trip
    w = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    y = (rgb * w).sum(dim=1, keepdim=True)
    return y


class _ResBlock(nn.Module):
    """
    Simple, stable residual block:
      GN -> SiLU -> Conv -> GN -> SiLU -> Conv, with residual.
    """
    def __init__(self, ch: int, groups: int = 32):
        super().__init__()
        g = min(groups, ch)  # safe even when ch < groups
        self.norm1 = nn.GroupNorm(g, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

        # stable init: small residual at start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class _Down(nn.Module):
    """
    Stable downsample by strided conv (keeps model fully convolutional).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LCGM(nn.Module):
    """
    Luminance/Color Guidance Module (LCGM)
    - Most stable version: ONLY reads RGB input.
    - Produces a multi-scale guide pyramid: List[Tensor], each (B, guide_channels, H/2^i, W/2^i)

    Design goals:
    1) No dependency on encoder internal feature shapes => fewer integration bugs.
    2) Use luma prior (Y from RGB) as a guidance cue, but stay in RGB pipeline (no YCbCr->RGB round-trip).
    3) Output channels fixed; injection module adapts to decoder channels via 1x1 conv.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        guide_channels: int = 32,
        num_scales: int = 4,
        use_luma: bool = True,
        groups: int = 32,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("LCGM most-stable version expects RGB input with in_channels=3.")
        if num_scales < 1:
            raise ValueError("num_scales must be >= 1")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.guide_channels = guide_channels
        self.num_scales = num_scales
        self.use_luma = use_luma

        # Input stem: RGB (+ optional Y) -> base_channels
        stem_in = 3 + (1 if use_luma else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

        # Per-scale trunk and heads
        trunks = []
        downs = []
        heads = []

        ch = base_channels
        for s in range(num_scales):
            # a couple of residual blocks at each scale (stable, low-risk)
            trunk = nn.Sequential(
                _ResBlock(ch, groups=groups),
                _ResBlock(ch, groups=groups),
            )
            trunks.append(trunk)

            # head projects to fixed guide_channels
            head = nn.Conv2d(ch, guide_channels, kernel_size=1)
            heads.append(head)

            # downsample for next scale except last
            if s != num_scales - 1:
                downs.append(_Down(ch, ch))  # keep channels constant for stability
            else:
                downs.append(nn.Identity())

        self.trunks = nn.ModuleList(trunks)
        self.downs = nn.ModuleList(downs)
        self.heads = nn.ModuleList(heads)

        # Stable init for heads: small output at start
        for h in self.heads:
            nn.init.zeros_(h.bias)

    def forward(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        """
        rgb: (B,3,H,W), typically in [-1,1] or [0,1]
        returns: [g0, g1, ..., g_{S-1}]
        """
        if rgb.dim() != 4 or rgb.size(1) != 3:
            raise ValueError(f"Expected rgb (B,3,H,W), got {tuple(rgb.shape)}")

        x = rgb
        if self.use_luma:
            y = _rgb_to_luma(rgb)
            x = torch.cat([rgb, y], dim=1)

        h = self.stem(x)

        guides: list[torch.Tensor] = []
        for s in range(self.num_scales):
            h = self.trunks[s](h)
            g = self.heads[s](h)
            guides.append(g)
            h = self.downs[s](h)

        return guides
