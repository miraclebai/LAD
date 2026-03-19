import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# -----------------------------
# 1) Load grayscale image as mass field M in [0,1]
# -----------------------------
def load_gray_01(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    arr = np.clip(arr, 0, 1)
    return torch.from_numpy(arr)[None, None]  # [1,1,H,W]

# -----------------------------
# 2) Save as true grayscale PNG (no colormap, no borders)
# -----------------------------
def save_gray_png(t: torch.Tensor, path: str):
    arr = t[0, 0].detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    img = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)

# -----------------------------
# 3) Anisotropic diffusion (Perona–Malik, same as your M-Flow core)
# -----------------------------
@torch.no_grad()
def anisotropic_diffuse(m: torch.Tensor, iters=23, kappa=0.10, lam=0.20, eps=1e-8):
    assert lam <= 0.25, "lambda should be <= 0.25 for explicit 2D 4-neighbor scheme"

    kappa_t = torch.tensor(kappa, device=m.device, dtype=m.dtype).view(1,1,1,1)
    lam_t   = torch.tensor(lam,   device=m.device, dtype=m.dtype).view(1,1,1,1)

    out = m.clone()
    for _ in range(iters):
        m_pad = F.pad(out, (1,1,1,1), mode="reflect")
        c  = m_pad[:,:,1:-1,1:-1]
        n  = m_pad[:,:,0:-2,1:-1]
        s  = m_pad[:,:,2:,  1:-1]
        w  = m_pad[:,:,1:-1,0:-2]
        e  = m_pad[:,:,1:-1,2:]

        dN = n - c
        dS = s - c
        dW = w - c
        dE = e - c

        gN = torch.exp(- (dN.abs() / (kappa_t + eps)) ** 2)
        gS = torch.exp(- (dS.abs() / (kappa_t + eps)) ** 2)
        gW = torch.exp(- (dW.abs() / (kappa_t + eps)) ** 2)
        gE = torch.exp(- (dE.abs() / (kappa_t + eps)) ** 2)

        update = gN*dN + gS*dS + gW*dW + gE*dE
        out = c + lam_t * update
        out = out.clamp(0.0, 1.0)

    return out

# -----------------------------
# 4) Isotropic diffusion (heat equation baseline)
# -----------------------------
@torch.no_grad()
def isotropic_diffuse(m: torch.Tensor, iters=200, lam=0.20):
    assert lam <= 0.25, "lambda should be <= 0.25 for explicit 2D 4-neighbor scheme"

    lam_t = torch.tensor(lam, device=m.device, dtype=m.dtype).view(1,1,1,1)

    out = m.clone()
    for _ in range(iters):
        m_pad = F.pad(out, (1,1,1,1), mode="reflect")
        c  = m_pad[:,:,1:-1,1:-1]
        n  = m_pad[:,:,0:-2,1:-1]
        s  = m_pad[:,:,2:,  1:-1]
        w  = m_pad[:,:,1:-1,0:-2]
        e  = m_pad[:,:,1:-1,2:]

        update = (n-c) + (s-c) + (w-c) + (e-c)
        out = c + lam_t * update
        out = out.clamp(0.0, 1.0)

    return out

# -----------------------------
# 5) Run (ONLY TWO OUTPUTS)
# -----------------------------
if __name__ == "__main__":
    img_path = "/Users/baijingyuan/Desktop/result/arrow.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m0 = load_gray_01(img_path).to(device)

    # 调参：扩散“强度”主要用 iters 控制；lam 固定确保稳定；kappa 控制保边
    iters = 630     # 50/200/300/1000
    lam   = 0.20    # <= 0.25
    kappa = 0.13    # 0.05 更保边，0.2 更接近各向同性

    miso   = isotropic_diffuse(m0, iters=iters, lam=lam)
    maniso = anisotropic_diffuse(m0, iters=iters, kappa=kappa, lam=lam)

    # ✅ 只保存两张
    save_gray_png(miso,   "iso.png")
    save_gray_png(maniso, "aniso.png")

    print("Saved: iso.png (isotropic), aniso.png (anisotropic)")
