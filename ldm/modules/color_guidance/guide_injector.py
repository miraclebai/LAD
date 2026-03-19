import torch
import torch.nn as nn
import torch.nn.functional as F


class GuideFiLM(nn.Module):
    """
    Most stable guide injection operator: FiLM (gamma, beta)
      out = h * (1 + gamma) + beta

    - Accepts decoder feature h: (B, C, H, W)
    - Accepts guide feature g: (B, G, Hg, Wg)  (channels G can be fixed, e.g., 32)
    - Internally:
        1) resize g -> (H,W)
        2) 1x1 conv project to 2C channels -> split into gamma/beta
        3) zero-init last conv to start as identity mapping (gamma~0,beta~0)
    """
    def __init__(
        self,
        in_guide_channels: int,
        out_feat_channels: int,
        hidden_channels: int = 128,
        use_norm: bool = True,
    ):
        super().__init__()
        self.in_guide_channels = int(in_guide_channels)
        self.out_feat_channels = int(out_feat_channels)

        layers = []
        if use_norm:
            # GroupNorm is robust for small batch sizes
            g = min(32, hidden_channels)
            layers += [
                nn.Conv2d(self.in_guide_channels, hidden_channels, 1),
                nn.GroupNorm(g, hidden_channels),
                nn.SiLU(inplace=True),
            ]
        else:
            layers += [
                nn.Conv2d(self.in_guide_channels, hidden_channels, 1),
                nn.SiLU(inplace=True),
            ]

        self.pre = nn.Sequential(*layers)
        self.to_gb = nn.Conv2d(hidden_channels, 2 * self.out_feat_channels, kernel_size=1)

        # zero-init to make modulation start near identity (very stable)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if h.dim() != 4:
            raise ValueError(f"h must be BCHW, got {tuple(h.shape)}")
        if guide is None:
            return h
        if guide.dim() != 4:
            raise ValueError(f"guide must be BCHW, got {tuple(guide.shape)}")

        B, C, H, W = h.shape

        # resize guide spatially to match h
        if guide.shape[-2:] != (H, W):
            guide = F.interpolate(guide, size=(H, W), mode="bilinear", align_corners=False)

        x = self.pre(guide)
        gb = self.to_gb(x)
        gamma, beta = torch.chunk(gb, 2, dim=1)

        # FiLM
        return h * (1.0 + gamma) + beta
