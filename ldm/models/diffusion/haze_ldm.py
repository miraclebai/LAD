# ldm/models/diffusion/haze_ldm.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ldm.models.diffusion.ddpm import LatentDiffusion


Tensor = torch.Tensor


# ---------------------------
# Utilities
# ---------------------------

@dataclass
class EncodeWithHazeOut:
    posterior: Any               # DiagonalGaussianDistribution or similar
    M_L: Tensor                  # (B,1,Hm,Wm)
    M_0: Optional[Tensor] = None # (B,1,H,W) or similar
    extras: Optional[Dict[str, Any]] = None


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _unpack_encode_with_haze(ret: Any) -> EncodeWithHazeOut:
    """
    Robustly unpack output of first_stage_model.encode_with_haze(x).

    Supported patterns:
      1) (posterior, M_L, M_0)
      2) (posterior, M_L)
      3) dict with keys: posterior|post, M_L|ml|M, M_0|m0 optional
      4) namespace-like object with attrs: posterior, M_L, M_0
    """
    if isinstance(ret, dict):
        posterior = ret.get("posterior", ret.get("post", None))
        M_L = ret.get("M_L", ret.get("ml", ret.get("M", None)))
        M_0 = ret.get("M_0", ret.get("m0", None))
        extras = {k: v for k, v in ret.items() if k not in {"posterior", "post", "M_L", "ml", "M", "M_0", "m0"}}
        if posterior is None or M_L is None:
            raise RuntimeError("encode_with_haze returned dict but missing posterior or M_L.")
        return EncodeWithHazeOut(posterior=posterior, M_L=M_L, M_0=M_0, extras=extras)

    if hasattr(ret, "posterior") and hasattr(ret, "M_L"):
        poster = getattr(ret, "posterior")
        ml = getattr(ret, "M_L")
        m0 = getattr(ret, "M_0", None)
        extras = {}
        return EncodeWithHazeOut(posterior=poster, M_L=ml, M_0=m0, extras=extras)

    if isinstance(ret, (tuple, list)):
        if len(ret) == 3:
            posterior, M_L, M_0 = ret
            return EncodeWithHazeOut(posterior=posterior, M_L=M_L, M_0=M_0, extras=None)
        if len(ret) == 2:
            posterior, M_L = ret
            return EncodeWithHazeOut(posterior=posterior, M_L=M_L, M_0=None, extras=None)

    raise RuntimeError(
        "encode_with_haze() output format unsupported. "
        "Expected tuple/list (posterior, M_L[, M_0]) or dict or object with attrs."
    )


def _get_unet_context_dim(model: Any) -> int:
    """
    Best-effort: discover expected context dim of UNet cross-attention.

    Common patterns in SD/StableSR:
      self.model.diffusion_model.context_dim
      self.model.model.diffusion_model.context_dim
      self.model.context_dim
    """
    candidates = []
    # DiffusionWrapper often stores as self.model.diffusion_model
    if hasattr(model, "diffusion_model") and hasattr(model.diffusion_model, "context_dim"):
        candidates.append(int(model.diffusion_model.context_dim))
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model") and hasattr(model.model.diffusion_model, "context_dim"):
        candidates.append(int(model.model.diffusion_model.context_dim))
    if hasattr(model, "context_dim"):
        candidates.append(int(model.context_dim))

    if len(candidates) == 0:
        raise RuntimeError(
            "Cannot infer UNet context_dim. "
            "Please pass `unet_context_dim` in HazeLatentDiffusion init."
        )
    # choose first valid
    return candidates[0]


def _normalize_to_crossattn_context(c: Any) -> Optional[Tensor]:
    """
    Normalize `c` to Tensor of shape (B, N, D) or None.
    LatentDiffusion sometimes returns:
      - Tensor
      - list with one Tensor
      - dict with key 'c_crossattn' / 'crossattn'
    """
    if c is None:
        return None
    if _is_tensor(c):
        # could be (B,D) -> make (B,1,D)
        if c.ndim == 2:
            return c[:, None, :]
        return c
    if isinstance(c, (list, tuple)):
        if len(c) == 0:
            return None
        # typical: [Tensor(B,N,D)]
        t = c[0]
        return _normalize_to_crossattn_context(t)
    if isinstance(c, dict):
        # common: {'c_crossattn': [Tensor], 'c_concat': [Tensor]}
        for k in ("c_crossattn", "crossattn", "context"):
            if k in c:
                return _normalize_to_crossattn_context(c[k])
        return None
    return None


# ---------------------------
# Token Encoders
# ---------------------------

class PatchTokenEncoder(nn.Module):
    """
    Generic patch -> token encoder for 2D maps.
    Input:  (B,C,H,W)
    Output: (B, N, D)
    """
    def __init__(
        self,
        in_ch: int,
        dim: int,
        patch: int = 8,
        add_cls_token: bool = True,
        use_pos_embed: bool = True,
        max_hw_tokens: int = 4096,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.dim = dim
        self.patch = patch
        self.add_cls_token = add_cls_token
        self.use_pos_embed = use_pos_embed
        self.max_hw_tokens = max_hw_tokens

        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch, padding=0)

        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

        if add_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.cls, std=0.02)

        if use_pos_embed:
            # learnable pos embed (1, 1+max_hw_tokens, dim)
            base = 1 + max_hw_tokens if add_cls_token else max_hw_tokens
            self.pos = nn.Parameter(torch.zeros(1, base, dim))
            nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.proj(x)  # (B, D, H', W')
        x = rearrange(x, "b d hp wp -> b (hp wp) d")  # tokens
        x = self.norm(x)
        x = x + self.mlp(x)

        if self.add_cls_token:
            cls = self.cls.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)

        if self.use_pos_embed:
            # handle variable token length by slicing or interpolating
            n = x.shape[1]
            if n <= self.pos.shape[1]:
                pos = self.pos[:, :n, :]
            else:
                # rare if very large image -> truncate (safe fallback)
                pos = self.pos[:, :self.pos.shape[1], :]
                x = x[:, :pos.shape[1], :]
            x = x + pos

        return x

    def unconditional(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        """
        Unconditional tokens: just one CLS token + zeros tokens (or only CLS)
        to support CFG. Keep shape (B,1,D) for stability.
        """
        device = device if device is not None else self.cls.device
        if self.add_cls_token:
            cls = self.cls.to(device).expand(batch_size, 1, -1)
            return cls
        return torch.zeros(batch_size, 1, self.dim, device=device)


class ContextMerger(nn.Module):
    """
    Merge multiple context token sets into a single (B, N_total, context_dim),
    with per-source projection and optional gating.

    This gives you a *full-feature* conditioning system:
      - content tokens (from hazy)
      - haze tokens (from M_L)
      - any future tokens (text, depth, etc.)
    """
    def __init__(
        self,
        context_dim: int,
        content_dim: int,
        haze_dim: int,
        use_gates: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_dim = context_dim

        self.content_proj = nn.Linear(content_dim, context_dim) if content_dim != context_dim else nn.Identity()
        self.haze_proj = nn.Linear(haze_dim, context_dim) if haze_dim != context_dim else nn.Identity()

        self.use_gates = use_gates
        if use_gates:
            # gates are conditioned on global summary of each source
            self.content_gate = nn.Sequential(nn.Linear(context_dim, context_dim), nn.Sigmoid())
            self.haze_gate = nn.Sequential(nn.Linear(context_dim, context_dim), nn.Sigmoid())

        self.drop = nn.Dropout(dropout)

    def forward(self, content_tokens: Tensor, haze_tokens: Tensor) -> Tensor:
        # project to UNet context dim
        c = self.content_proj(content_tokens)
        h = self.haze_proj(haze_tokens)

        if self.use_gates:
            # global token summary (mean over tokens)
            c_sum = c.mean(dim=1)
            h_sum = h.mean(dim=1)
            c = c * self.content_gate(c_sum)[:, None, :]
            h = h * self.haze_gate(h_sum)[:, None, :]

        ctx = torch.cat([c, h], dim=1)
        return self.drop(ctx)


# ---------------------------
# HazeLatentDiffusion
# ---------------------------

class HazeLatentDiffusion(LatentDiffusion):
    """
    Full-featured dehaze conditional diffusion in latent space.

    Training:
      - target: clear image -> z_clear (first stage encode)
      - condition: hazy image -> content tokens (from hazy latent)
      - haze condition: M_L from AE.encode_with_haze(hazy) -> haze tokens
      - UNet does cross-attention with merged context tokens.

    Inference:
      - given hazy image -> make conditions -> sample z -> decode -> clear prediction
    """

    def __init__(
        self,
        # haze token encoder
        haze_token_dim: int = 256,
        haze_patch: int = 8,
        haze_add_cls: bool = True,
        haze_pos_embed: bool = True,
        # content token encoder (default: from hazy latent)
        content_token_dim: int = 256,
        content_patch: int = 4,  # latent is small; patch 4 often ok for 64x64 latent
        content_add_cls: bool = True,
        content_pos_embed: bool = True,
        # merge
        unet_context_dim: Optional[int] = None,
        use_context_gates: bool = True,
        context_dropout: float = 0.0,
        # CFG / condition dropout
        drop_prob_content: float = 0.1,
        drop_prob_haze: float = 0.1,
        # keys
        hazy_key: str = "lq",
        clear_key: str = "gt",
        # whether first stage gradients
        first_stage_trainable: bool = False,
        **kwargs,
    ):
        """
        kwargs: passed to LatentDiffusion (configs typically define model.target & params)
        """
        super().__init__(**kwargs)

        self.hazy_key = hazy_key
        self.clear_key = clear_key

        self.drop_prob_content = float(drop_prob_content)
        self.drop_prob_haze = float(drop_prob_haze)

        self.first_stage_trainable = bool(first_stage_trainable)
        if not self.first_stage_trainable:
            self.first_stage_model.eval()
            for p in self.first_stage_model.parameters():
                p.requires_grad_(False)

        # Infer UNet context dim
        ctx_dim = unet_context_dim if unet_context_dim is not None else _get_unet_context_dim(self.model)

        # token encoders
        self.haze_encoder = PatchTokenEncoder(
            in_ch=1,
            dim=haze_token_dim,
            patch=haze_patch,
            add_cls_token=haze_add_cls,
            use_pos_embed=haze_pos_embed,
            max_hw_tokens=4096,
            mlp_ratio=2.0,
            dropout=context_dropout,
        )

        # content tokens from hazy latent (shape [B, C_lat, H_lat, W_lat])
        # We do patch-tokenization on hazy latent to preserve content required for dehazing.
        # Need latent channels: typically 4. We'll discover at runtime; use lazy conv if needed.
        self._content_in_ch = None
        self._lazy_content_proj: Optional[nn.Conv2d] = None
        self.content_token_dim = content_token_dim
        self.content_patch = content_patch
        self.content_add_cls = content_add_cls
        self.content_pos_embed = content_pos_embed
        self.content_encoder: Optional[PatchTokenEncoder] = None  # created lazily

        # merger -> ctx
        # content_dim/haze_dim are token encoder dims
        self.ctx_merger = ContextMerger(
            context_dim=ctx_dim,
            content_dim=content_token_dim,
            haze_dim=haze_token_dim,
            use_gates=use_context_gates,
            dropout=context_dropout,
        )

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _maybe_build_content_encoder(self, in_ch: int) -> None:
        if self.content_encoder is not None and self._content_in_ch == in_ch:
            return
        self._content_in_ch = in_ch
        self.content_encoder = PatchTokenEncoder(
            in_ch=in_ch,
            dim=self.content_token_dim,
            patch=self.content_patch,
            add_cls_token=self.content_add_cls,
            use_pos_embed=self.content_pos_embed,
            max_hw_tokens=4096,
            mlp_ratio=2.0,
            dropout=0.0,
        )

    def _encode_clear_to_z(self, x_clear: Tensor) -> Any:
        """
        Use LatentDiffusion's first-stage encode for target clear.
        Returns posterior distribution (DiagonalGaussianDistribution).
        """
        # LatentDiffusion has encode_first_stage -> posterior
        posterior = self.encode_first_stage(x_clear)
        return posterior

    @torch.no_grad()
    def _encode_haze_map(self, x_hazy: Tensor) -> EncodeWithHazeOut:
        """
        Call first_stage_model.encode_with_haze(hazy) -> posterior_hazy, M_L, ...
        M_L produced by your HazeAwareEncoder branch (mass flow refined).
        """
        ae = self.first_stage_model
        if not hasattr(ae, "encode_with_haze"):
            raise RuntimeError(
                "HazeLatentDiffusion requires first_stage_model to implement encode_with_haze(x). "
                "Your HazeAutoencoderKLResi should provide it."
            )
        ret = ae.encode_with_haze(x_hazy)
        out = _unpack_encode_with_haze(ret)
        if out.M_L is None or not _is_tensor(out.M_L):
            raise RuntimeError("encode_with_haze did not return a valid tensor M_L.")
        # enforce shape (B,1,H,W)
        if out.M_L.ndim == 3:
            out.M_L = out.M_L[:, None, :, :]
        if out.M_L.shape[1] != 1:
            # If implementation returns multi-channel haze features, compress to 1 channel
            out.M_L = out.M_L.mean(dim=1, keepdim=True)
        return out

    def _maybe_drop_condition(self, tokens: Tensor, drop_prob: float) -> Tensor:
        """
        Classifier-free guidance: randomly drop condition per-sample.
        If dropped -> replace with unconditional token(s).
        """
        if drop_prob <= 0:
            return tokens

        b = tokens.shape[0]
        device = tokens.device
        mask = (torch.rand(b, device=device) < drop_prob)  # True means drop

        if not mask.any():
            return tokens

        # Construct unconditional tokens with same feature dim
        # We keep shape (B, 1, D) for dropped samples, then broadcast/replace first token
        # More stable: use dedicated encoder unconditional.
        uncond = torch.zeros(b, 1, tokens.shape[-1], device=device)
        # replace entire tokens with uncond token for those samples
        tokens = tokens.clone()
        tokens[mask] = uncond[mask]
        return tokens

    def _build_context(self, x_hazy: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Build merged cross-attn context tokens from:
          - content tokens: hazy latent tokens
          - haze tokens: from M_L (refined haze map)

        Returns:
          context: (B, N, context_dim)
          debug: dict with intermediate tensors for logging
        """
        # 1) get haze branch outputs
        haze_out = self._encode_haze_map(x_hazy)
        M_L = haze_out.M_L  # (B,1,Hm,Wm)
        haze_tokens = self.haze_encoder(M_L)

        # 2) content tokens from hazy latent (posterior of hazy)
        # For stability, use posterior mode (mean) as content representation for conditioning.
        # This is common in conditional LDM (use deterministic cond encoder).
        post_hazy = haze_out.posterior
        if hasattr(post_hazy, "mode"):
            z_hazy = post_hazy.mode()
        else:
            # fallback: sample
            z_hazy = post_hazy.sample()
        z_hazy = z_hazy * self.scale_factor  # match diffusion latent scale

        # ensure shape (B,C,H,W); if (B,H,W,C) etc, user must correct upstream
        if z_hazy.ndim != 4:
            raise RuntimeError(f"Unexpected hazy latent shape: {tuple(z_hazy.shape)}. Expected (B,C,H,W).")

        self._maybe_build_content_encoder(in_ch=z_hazy.shape[1])
        assert self.content_encoder is not None
        content_tokens = self.content_encoder(z_hazy)

        # 3) cfg-style condition dropout
        content_tokens = self._maybe_drop_condition(content_tokens, self.drop_prob_content)
        haze_tokens = self._maybe_drop_condition(haze_tokens, self.drop_prob_haze)

        # 4) merge to UNet context_dim
        context = self.ctx_merger(content_tokens, haze_tokens)

        debug = {
            "M_L": M_L.detach(),
            "z_hazy": z_hazy.detach(),
        }
        return context, debug

    # ---------------------------
    # Overridden LatentDiffusion API
    # ---------------------------

    def get_input(
        self,
        batch: Dict[str, Any],
        k: Optional[str] = None,
        return_first_stage_outputs: bool = False,
        force_c_encode: bool = False,
        cond_key: Optional[str] = None,
        return_original_cond: bool = False,
        bs: Optional[int] = None,
    ):
        """
        Full training input assembly.

        - target x: clear image tensor batch[clear_key]
        - condition: hazy image tensor batch[hazy_key]
        - produce z from clear
        - produce context tokens from hazy (content+haze tokens)
        """
        # 0) fetch tensors
        if self.clear_key not in batch:
            raise KeyError(f"Batch missing clear_key='{self.clear_key}'. Available keys: {list(batch.keys())}")
        if self.hazy_key not in batch:
            raise KeyError(f"Batch missing hazy_key='{self.hazy_key}'. Available keys: {list(batch.keys())}")

        x_clear = batch[self.clear_key]
        x_hazy = batch[self.hazy_key]

        # optional batch size truncation
        if bs is not None:
            x_clear = x_clear[:bs]
            x_hazy = x_hazy[:bs]

        # Ensure float
        x_clear = x_clear.float()
        x_hazy = x_hazy.float()

        # 1) encode clear (target)
        posterior = self._encode_clear_to_z(x_clear)
        if hasattr(posterior, "sample"):
            z = posterior.sample()
        else:
            raise RuntimeError("first_stage posterior has no .sample().")
        z = z * self.scale_factor

        # 2) build cross-attention context from hazy
        # If first_stage_trainable==True, allow gradients through haze path; else no_grad above.
        if self.first_stage_trainable:
            # if trainable, do not use no_grad in _encode_haze_map
            # We re-implement quickly: call encode_with_haze with grad.
            ae = self.first_stage_model
            ret = ae.encode_with_haze(x_hazy)
            haze_out = _unpack_encode_with_haze(ret)
            M_L = haze_out.M_L
            if M_L.ndim == 3:
                M_L = M_L[:, None, :, :]
            if M_L.shape[1] != 1:
                M_L = M_L.mean(dim=1, keepdim=True)
            haze_tokens = self.haze_encoder(M_L)

            post_hazy = haze_out.posterior
            z_hazy = post_hazy.mode() if hasattr(post_hazy, "mode") else post_hazy.sample()
            z_hazy = z_hazy * self.scale_factor
            self._maybe_build_content_encoder(in_ch=z_hazy.shape[1])
            content_tokens = self.content_encoder(z_hazy)  # type: ignore

            content_tokens = self._maybe_drop_condition(content_tokens, self.drop_prob_content)
            haze_tokens = self._maybe_drop_condition(haze_tokens, self.drop_prob_haze)
            context = self.ctx_merger(content_tokens, haze_tokens)

            cond_debug = {"M_L": M_L.detach(), "z_hazy": z_hazy.detach()}
        else:
            context, cond_debug = self._build_context(x_hazy)

        # 3) Compose condition dict in the format DiffusionWrapper expects
        # In CompVis LDM, apply_model usually expects:
        #   cond = {"c_crossattn": [context]} for crossattn mode
        # We comply with that.
        cond = {"c_crossattn": [context]}

        out = [z, cond]
        if return_first_stage_outputs:
            out.extend([x_clear, posterior])
        if return_original_cond:
            out.append(x_hazy)
        # attach debug for logging
        out.append(cond_debug)
        return out

    def apply_model(self, x_noisy: Tensor, t: Tensor, cond: Dict[str, Any], *args, **kwargs):
        """
        Ensure context is passed as `context` into UNet.
        LatentDiffusion.apply_model already does similar; we keep this override
        to be robust if upstream expects different cond format.
        """
        # Accept both our dict or raw tensor/list.
        context = None
        if isinstance(cond, dict):
            context = _normalize_to_crossattn_context(cond.get("c_crossattn", cond.get("context", None)))
        else:
            context = _normalize_to_crossattn_context(cond)

        # Many repos set self.model.conditioning_key == 'crossattn' and call self.model(x,t,context=context).
        return self.model(x_noisy, t, context=context)

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """
        Provide unconditional tokens for classifier-free guidance sampling.
        """
        # Build unconditional content tokens and haze tokens, then merge.
        if self.content_encoder is None:
            # fallback if never built: assume 4-channel latent
            self._maybe_build_content_encoder(in_ch=4)

        assert self.content_encoder is not None
        u_content = self.content_encoder.unconditional(batch_size, device=device)
        u_haze = self.haze_encoder.unconditional(batch_size, device=device)
        u_ctx = self.ctx_merger(
            # project expects real tokens; but unconditional are (B,1,D)
            u_content,
            u_haze
        )
        return {"c_crossattn": [u_ctx]}

    # ---------------------------
    # Public inference API
    # ---------------------------

    @torch.no_grad()
    def dehaze(
        self,
        x_hazy: Tensor,
        num_steps: int = 50,
        guidance_scale: float = 5.0,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        One-call dehaze inference:
          hazy -> build condition -> sample z_clear -> decode -> clear image

        Uses DDIM sampler if available; otherwise relies on parent sampling utilities.
        """
        device = x_hazy.device
        b = x_hazy.shape[0]

        # build conditional context
        context, cond_debug = self._build_context(x_hazy)

        cond = {"c_crossattn": [context]}
        uncond = self.get_unconditional_conditioning(b, device)

        # Use the built-in sampler interface if your base repo has it.
        # Many LDM repos provide self.sample_log or DDIMSampler. We try common patterns.

        # 1) try DDIMSampler if present
        if hasattr(self, "ddim_sampler") and self.ddim_sampler is not None:
            sampler = self.ddim_sampler
            shape = (self.channels, x_hazy.shape[2] // 8, x_hazy.shape[3] // 8)  # typical 8x down
            z, intermediates = sampler.sample(
                S=num_steps,
                conditioning=cond,
                batch_size=b,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uncond,
                eta=eta,
            )
        else:
            # 2) fallback to p_sample_loop if exists
            if not hasattr(self, "p_sample_loop"):
                raise RuntimeError(
                    "No sampler found. Your repo should provide DDIMSampler or p_sample_loop. "
                    "Please integrate your sampler and call it here."
                )
            shape = (b, self.channels, x_hazy.shape[2] // 8, x_hazy.shape[3] // 8)
            z = self.p_sample_loop(cond=cond, shape=shape, unconditional_guidance_scale=guidance_scale, unconditional_conditioning=uncond)
            intermediates = {}

        # decode
        x_clear = self.decode_first_stage(z / self.scale_factor)

        if return_intermediates:
            extras = {
                "z": z,
                "cond_debug": cond_debug,
                "intermediates": intermediates,
            }
            return x_clear, extras
        return x_clear
