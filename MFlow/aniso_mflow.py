# MFlow/aniso_mflow.py
"""
各向异性扩散驱动的 hazy mass map 演化模块 (M-flow)。

提供三个核心模块：
  1. AnisotropicDiffusionStep
       - 对 hazy mass map M ∈ [0,1] 做各向异性扩散，强化“局部平滑 + 保边界”。
       - 可选使用特征 F 作为引导，调整扩散强度（kappa）和时间步长（lambda）。

  2. HazeFeatureModulation
       - 使用当前层的 M_k 对同分辨率的特征 F_k 做调制 (spatial gate + channel-wise FiLM)。

  3. HazeMFlowUnit
       - 封装 (F, M) -> (F', M')：
           M' = AnisotropicDiffusionStep(M, F)
           F' = HazeFeatureModulation(F, M')
       - 这是在 Encoder 中每一层调用的基本单元。

注意：
  - 这里假定输入的 M 已经是 [0,1]，来自前面的 DWT+DCP HazyMassMapGenerator。
  - 不对特征 F 做任何归一化，只在内部用它来引导扩散或 FiLM。
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from hazy_mass.dcp import to_4d

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
    ) -> torch.Tensor:
        """
        单步各向异性扩散更新。

        Args:
            m: [B,1,H,W]，当前 mass map（在 [0,1] 附近）
            kappa: [B,1,1,1]，边缘敏感度
            lam:   [B,1,1,1]，时间步长

        Returns:
            m_new: [B,1,H,W]，更新后的 mass map
        """
        B, C, H, W = m.shape
        assert C == 1, "AnisotropicDiffusionStep currently expects 1-channel M."

        # 复制边界，便于计算 N,S,E,W
        m_pad = F.pad(m, (1, 1, 1, 1), mode="reflect")  # [B,1,H+2,W+2]

        center = m_pad[:, :, 1:-1, 1:-1]   # [B,1,H,W]
        north = m_pad[:, :, 0:-2, 1:-1]   # 上
        south = m_pad[:, :, 2:, 1:-1]     # 下
        west  = m_pad[:, :, 1:-1, 0:-2]   # 左
        east  = m_pad[:, :, 1:-1, 2:]     # 右

        dN = north - center
        dS = south - center
        dW = west  - center
        dE = east  - center

        # 梯度幅值用于导通系数 c 的计算
        # 这里使用 exp( - (|d|/kappa)^2 )
        gN = torch.exp(- (dN.abs() / (kappa + eps)) ** 2)
        gS = torch.exp(- (dS.abs() / (kappa + eps)) ** 2)
        gW = torch.exp(- (dW.abs() / (kappa + eps)) ** 2)
        gE = torch.exp(- (dE.abs() / (kappa + eps)) ** 2)

        # 各向异性扩散更新
        update = gN * dN + gS * dS + gW * dW + gE * dE  # [B,1,H,W]
        m_new = center + lam * update

        return m_new

    def forward(
        self,
        m: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:    hazy mass map，形状 [B,1,H,W] 或兼容形状，值域应为 [0,1]。
            feat: 可选的 feature F，用于指导扩散强度（不做归一化，只用其统计量）。

        Returns:
            m_out: [B,1,H,W]，在 [0,1] 范围内的更新后 mass map。
        """
        m = to_4d(m)
        B, C, H, W = m.shape
        assert C == 1, f"Expected M to have 1 channel, got {C}."

        kappa, lam = self._compute_kappa_lambda(m, feat)  # [B,1,1,1]

        m_new = m
        for _ in range(self.num_iters):
            m_new = self._diffusion_update(m_new, kappa, lam)

        # 保证输出仍在 [0,1]
        m_new = torch.clamp(m_new, 0.0, 1.0)
        return m_new                                      # [B,1,H,W]


class HazeFeatureModulation(nn.Module):
    r"""
    使用当前 hazy mass map M_k 对特征 F_k 做调制的模块。

    调制方式包括两部分（可以同时开启或单独使用）：
      1. Spatial gate:
           G_s = sigmoid(Conv(M_k))  => F' = F * G_s
         让雾浓区域的特征被抑制或重新编码，雾轻区域的特征更保留。

      2. Channel-wise FiLM:
           m_global = GAP(M_k) -> MLP -> (gamma, beta)
           F' = F * (1 + gamma) + beta
         通过全局雾强度调整各通道的缩放与偏置。
    """

    def __init__(
        self,
        feat_channels: int,
        m_channels: int = 1,
        use_spatial: bool = True,
        use_channel: bool = True,
        channel_reduction: int = 16,
    ) -> None:
        super().__init__()
        assert feat_channels > 0, "feat_channels must be positive."

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        if self.use_spatial:
            # 3x3 卷积生成一个空间 gate（单通道），再经过 sigmoid
            self.spatial_conv = nn.Conv2d(
                m_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.spatial_conv = None

        if self.use_channel:
            hidden = max(4, feat_channels // channel_reduction)
            self.channel_mlp = nn.Sequential(
                nn.Linear(m_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * feat_channels),  # 输出 gamma 和 beta
            )
        else:
            self.channel_mlp = None

    def forward(self, feat: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: 当前层特征 F_k，形状 [B,C,H,W]。
            m:    当前层对应的 hazy mass map M_k，形状 [B,1,H,W]（或兼容形状），值域 [0,1]。

        Returns:
            feat_mod: 调制后的特征，形状同 feat。
        """
        feat = to_4d(feat)
        m = _ensure_same_spatial(m, feat)

        B, C, H, W = feat.shape
        _, C_m, _, _ = m.shape

        out = feat

        # Channel-wise FiLM（全局雾强度 -> gamma, beta）
        if self.use_channel and self.channel_mlp is not None:
            # 对 M 做 GAP -> [B, C_m]
            m_global = F.adaptive_avg_pool2d(m, output_size=1).view(B, C_m)
            # MLP -> [B, 2*C]
            gamma_beta = self.channel_mlp(m_global)  # [B, 2C]
            gamma, beta = torch.split(gamma_beta, C, dim=1)  # 各 [B,C]

            # 用 tanh 限制调整幅度，避免过大
            gamma = torch.tanh(gamma).view(B, C, 1, 1)  # [-1,1] 范围
            beta = torch.tanh(beta).view(B, C, 1, 1)

            # (1 + gamma) 保证缩放在 [0,2] 附近
            out = out * (1.0 + gamma) + beta

        # Spatial gate（空间位置依赖的 gating）
        if self.use_spatial and self.spatial_conv is not None:
            gate = self.spatial_conv(m)              # [B,1,H,W]
            gate = torch.sigmoid(gate)               # [0,1]
            out = out * gate                         # 按空间位置调整特征强度

        return out


class HazeMFlowUnit(nn.Module):
    r"""
    单层 M-flow 单元：将“各向异性扩散 + 特征调制”组合在一起。

      输入： F_k, M_k
      过程：
        1. M_k' = AnisotropicDiffusionStep(M_k, F_k)
        2. F_k' = HazeFeatureModulation(F_k, M_k')
      输出： F_k', M_k'

    这就是在 Encoder 各层使用的基本积木。
    """

    def __init__(
        self,
        feat_channels: int,
        m_channels: int = 1,
        # diffusion parameters
        diffusion_iters: int = 3,
        base_kappa: float = 0.1,
        base_lambda: float = 0.2,
        guided_by_feat: bool = True,
        guide_reduction: int = 16,
        # modulation parameters
        use_spatial_modulation: bool = True,
        use_channel_modulation: bool = True,
        channel_reduction: int = 16,
        # clamp for M
        m_min: float = 0.0,
        m_max: float = 1.0,
    ) -> None:
        super().__init__()

        self.diffusion = AnisotropicDiffusionStep(
            num_iters=diffusion_iters,
            base_kappa=base_kappa,
            base_lambda=base_lambda,
            guided_by_feat=guided_by_feat,
            feat_channels=feat_channels if guided_by_feat else None,
            guide_reduction=guide_reduction,
        )

        self.modulation = HazeFeatureModulation(
            feat_channels=feat_channels,
            m_channels=m_channels,
            use_spatial=use_spatial_modulation,
            use_channel=use_channel_modulation,
            channel_reduction=channel_reduction,
        )

    def forward(
        self,
        feat: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: 特征 F_k，形状 [B,C,H,W]。
            m:    hazy mass map M_k，形状 [B,1,H,W]（或兼容形状），值域 [0,1]。

        Returns:
            feat_out: 调制后的特征 F_k'，形状与 feat 相同。
            m_out:    各向异性扩散之后的 mass map M_k'，形状与 m 对齐，值域 [m_min, m_max]。
        """
        feat = to_4d(feat)
        m = _ensure_same_spatial(m, feat)

        # 1) 各向异性扩散：更新 M
        m_out = self.diffusion(m, feat)  # [B,1,H,W]

        # 2) 用更新后的 M 调制特征
        feat_out = self.modulation(feat, m_out)  # [B,C,H,W]

        return feat_out, m_out