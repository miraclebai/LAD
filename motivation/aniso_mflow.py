# motivation/aniso_mflow.py
"""
Minimal, stable anisotropic diffusion step for motivation visualization.

This file intentionally keeps the API simple:
- AnisotropicDiffusionStep.forward(m, feat=None) -> m_out  (ONLY ONE return)
No conductance C, no intermediates, no tuples (to avoid unpack pitfalls).

You can later switch to the full MFlow implementation for method section,
but for motivation we only need pure field evolution on M.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse your repo's to_4d (same behavior as haze_mass/dcp.py)
from hazy_mass.dcp import to_4d


class AnisotropicDiffusionStep(nn.Module):
    r"""
    Perona–Malik style anisotropic diffusion on haze mass map M.

    PDE (discrete 4-neighborhood):
      M_new = M + λ * sum_dir ( g(|dM_dir|/kappa) * dM_dir )
    with g(s) = exp(-s^2)

    This suppresses diffusion across sharp jumps (edges) and
    smooths within flat regions.

    For motivation, this acts as "spatial propagation" of haze intensity.
    """

    def __init__(
        self,
        num_iters: int = 1,
        base_kappa: float = 0.1,
        base_lambda: float = 0.2,
        guided_by_feat: bool = False,
        feat_channels: Optional[int] = None,
        guide_reduction: int = 16,
    ) -> None:
        super().__init__()
        assert num_iters > 0
        self.num_iters = int(num_iters)
        self.base_kappa = float(base_kappa)
        self.base_lambda = float(base_lambda)

        # keep guidance code for future use, but disabled by default for motivation
        self.guided_by_feat = bool(guided_by_feat and (feat_channels is not None))
        if self.guided_by_feat:
            hidden = max(4, int(feat_channels) // int(guide_reduction))
            self.guide_mlp = nn.Sequential(
                nn.Linear(int(feat_channels), hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2),  # Δkappa, Δlambda
            )
        else:
            self.guide_mlp = None

    def _compute_kappa_lambda(
        self, m: torch.Tensor, feat: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = m.shape[0]
        kappa = torch.full((B, 1, 1, 1), self.base_kappa, device=m.device, dtype=m.dtype)
        lam = torch.full((B, 1, 1, 1), self.base_lambda, device=m.device, dtype=m.dtype)

        if self.guided_by_feat and feat is not None and self.guide_mlp is not None:
            feat = to_4d(feat)
            desc = F.adaptive_avg_pool2d(feat.abs(), output_size=1).view(B, -1)
            deltas = self.guide_mlp(desc)  # [B,2]
            scale = 1.0 + 0.3 * torch.tanh(deltas)  # ~[0.7,1.3]
            kappa = kappa * scale[:, 0:1].view(B, 1, 1, 1)
            lam = lam * scale[:, 1:2].view(B, 1, 1, 1)

        lam = lam.clamp(min=0.01, max=0.25)
        kappa = kappa.clamp(min=1e-4)
        return kappa, lam

    @staticmethod
    def _diffusion_update(
        m: torch.Tensor,
        kappa: torch.Tensor,
        lam: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        m = to_4d(m)
        B, C, H, W = m.shape
        assert C == 1, "Expected 1-channel M."

        m_pad = F.pad(m, (1, 1, 1, 1), mode="reflect")  # [B,1,H+2,W+2]

        center = m_pad[:, :, 1:-1, 1:-1]
        north  = m_pad[:, :, 0:-2, 1:-1]
        south  = m_pad[:, :, 2:,   1:-1]
        west   = m_pad[:, :, 1:-1, 0:-2]
        east   = m_pad[:, :, 1:-1, 2:  ]

        dN = north - center
        dS = south - center
        dW = west  - center
        dE = east  - center

        denom = (kappa + eps)
        gN = torch.exp(- (dN.abs() / denom) ** 2)
        gS = torch.exp(- (dS.abs() / denom) ** 2)
        gW = torch.exp(- (dW.abs() / denom) ** 2)
        gE = torch.exp(- (dE.abs() / denom) ** 2)

        update = gN * dN + gS * dS + gW * dW + gE * dE
        m_new = center + lam * update
        return m_new

    def forward(self, m: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Always returns a single tensor m_out in [0,1].
        """
        m = to_4d(m)
        B, C, H, W = m.shape
        assert C == 1, f"Expected M to have 1 channel, got {C}."

        kappa, lam = self._compute_kappa_lambda(m, feat)

        m_new = m
        for _ in range(self.num_iters):
            m_new = self._diffusion_update(m_new, kappa, lam)

        return torch.clamp(m_new, 0.0, 1.0)
