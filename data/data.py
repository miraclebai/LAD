# data/data.py

import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence, Union

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from hazy_mass.hazy_mass import compute_hazy_mass_map

__all__ = ["PairedDataset"]


def _index_images_by_stem(root: str,
                          exts: Sequence[str]) -> Dict[str, str]:
    """
    扫描 root 目录下所有图片，按“文件名去掉后缀”的 stem 建一个索引。
    比如 hazy/abc.png -> key = 'abc'
    """
    root = os.path.expanduser(root)
    mapping: Dict[str, str] = {}
    for ext in exts:
        pattern = os.path.join(root, "**", f"*{ext}")
        for p in glob(pattern, recursive=True):
            stem = Path(p).stem
            mapping[stem] = p
    return mapping


class PairedDataset(Dataset):
    """
    成对去雾数据集：从 hazy_dir 和 clear_dir 中，按文件名匹配 hazy/clear 图像。

    假定：
      - hazy_dir 下有 xxx.png
      - clear_dir 下有 xxx.png / xxx.jpg 等
    会按“文件名去掉后缀”的方式做一一对应。

    做可选的：
      - resize（统一大小）
      - 中心裁剪
      - 归一化到 [-1, 1]

    返回一个 dict:
      {
        "hazy":  Tensor[C,H,W] in [-1, 1],
        "clear": Tensor[C,H,W] in [-1, 1],
        "hazy_path": str,
        "clear_path": str,
      }
    """

    def __init__(
        self,
        hazy_dir: str,
        clear_dir: str,
        img_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        resize_to: Optional[Union[int, Tuple[int, int]]] = None,
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        normalize_to_neg_one_one: bool = True,
    ) -> None:
        """
        Args:
            hazy_dir: 含有带雾图像的目录
            clear_dir: 含有对应清晰图像的目录
            img_exts: 识别为图像的后缀
            resize_to: 若不为 None，则先 resize 到给定尺寸。
                       int -> 正方形 (size, size)
                       (h, w) -> 指定尺寸
            crop_size: 若不为 None，则进行中心裁剪。
                       int -> 正方形 (size, size)
                       (h, w) -> 指定尺寸
            normalize_to_neg_one_one: 若 True，则把图像从 [0,1] 映射到 [-1,1]
        """
        super().__init__()
        self.hazy_dir = os.path.expanduser(hazy_dir)
        self.clear_dir = os.path.expanduser(clear_dir)
        self.img_exts = tuple(img_exts)

        # 统一处理 resize/crop 参数
        if isinstance(resize_to, int):
            self.resize_to = (resize_to, resize_to)
        else:
            self.resize_to = resize_to

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.normalize_to_neg_one_one = normalize_to_neg_one_one

        # 建 hazy / clear 索引
        hazy_map = _index_images_by_stem(self.hazy_dir, self.img_exts)
        clear_map = _index_images_by_stem(self.clear_dir, self.img_exts)

        # 按 hazy 为主，匹配 clear
        pairs: List[Tuple[str, str]] = []
        missing: List[str] = []
        for stem, hazy_path in hazy_map.items():
            clear_path = clear_map.get(stem, None)
            if clear_path is None:
                missing.append(stem)
            else:
                pairs.append((hazy_path, clear_path))

        if not pairs:
            raise RuntimeError(
                f"[PairedDataset] No paired images found between "
                f"{self.hazy_dir} and {self.clear_dir}."
            )

        if missing:
            print(
                f"[PairedDataset] Warning: {len(missing)} hazy images "
                f"do not have a matching clear image (by filename stem). "
                f"Examples: {missing[:5]}"
            )

        # 按 hazy_path 排序，保证可复现
        pairs.sort(key=lambda x: x[0])
        self.pairs = pairs

        print(
            f"[PairedDataset] Found {len(self.pairs)} pairs in "
            f"'{self.hazy_dir}' (hazy) and '{self.clear_dir}' (clear)."
        )

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _apply_deterministic_transforms(
        self, hazy: Image.Image, clear: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        对 hazy / clear 做完全相同的确定性几何变换（无随机性），
        确保成对样本严格对齐。
        """
        # 1) resize（如果设置了）
        if self.resize_to is not None:
            th, tw = self.resize_to  # 这里视为 (h, w)
            hazy = hazy.resize((tw, th), resample=Image.BICUBIC)
            clear = clear.resize((tw, th), resample=Image.BICUBIC)

        # 2) 中心裁剪（如果设置了）
        if self.crop_size is not None:
            th, tw = self.crop_size
            w, h = hazy.size  # PIL: (w, h)
            if th <= h and tw <= w:
                top = max(0, (h - th) // 2)
                left = max(0, (w - tw) // 2)
                hazy = TF.crop(hazy, top, left, th, tw)
                clear = TF.crop(clear, top, left, th, tw)
            # 如果裁剪比原图大，则不裁剪；你可以改成先 resize 再裁剪

        return hazy, clear

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        转 Tensor，并根据配置决定是否归一化到 [-1, 1]。
        """
        tensor = TF.to_tensor(img)  # [0, 1]
        if self.normalize_to_neg_one_one:
            tensor = tensor * 2.0 - 1.0  # [-1, 1]
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hazy_path, clear_path = self.pairs[index]

        hazy_img = self._load_image(hazy_path)
        clear_img = self._load_image(clear_path)

        # 确定性几何变换（无随机）
        # hazy_img, clear_img = self._apply_deterministic_transforms(hazy_img, clear_img)

        hazy_tensor = self._to_tensor(hazy_img)
        clear_tensor = self._to_tensor(clear_img)

        return {
            "hazy": hazy_tensor,                # [-1,1]
            "clear": clear_tensor,              # [-1,1]
            "hazy_path": hazy_path,
            "clear_path": clear_path,
        }



import os
import cv2
import numpy as np
from torch.utils import data as data
from basicsr.utils import FileClient, imfrombytes, img2tensor, bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms import functional as TF
from PIL import Image


class ResizePairedDataset(data.Dataset):
    """
    A clean paired dataset loader.
    - No random crop
    - No padding / BORDER_REFLECT
    - Resize both LQ and GT to fixed resolution (default: 512×512)
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.io_backend_opt = opt['io_backend']
        self.size = opt.get("size", 512)
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.file_client = None
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        # file enumeration
        # ---- build stem->path maps (like PairedDataset) ----
        IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        def index_by_stem(folder: str):
            mapping = {}
            for fn in os.listdir(folder):
                p = os.path.join(folder, fn)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    stem = os.path.splitext(fn)[0]
                    mapping[stem] = p
            return mapping

        gt_map = index_by_stem(self.gt_folder)
        lq_map = index_by_stem(self.lq_folder)

        # ---- pair by stem (use LQ as anchor) ----
        self.paths = []
        missing = []
        for stem, lq_path in lq_map.items():
            gt_path = gt_map.get(stem, None)
            if gt_path is None:
                missing.append(stem)
            else:
                self.paths.append({"stem": stem, "gt": gt_path, "lq": lq_path})

        if not self.paths:
            raise RuntimeError(
                f"[ResizePairedDataset] No paired images found between "
                f"{self.gt_folder} and {self.lq_folder} by stem."
            )

        if missing:
            print(
                f"[ResizePairedDataset] Warning: {len(missing)} LQ images have no matching GT by stem. "
                f"Examples: {missing[:5]}"
            )

        # sort for reproducibility
        self.paths.sort(key=lambda x: x["lq"])


    def __getitem__(self, idx):
        if self.file_client is None:
            # 注意：pop 会修改原 dict；如果你 dataloader 多 worker，建议用 copy()
            backend_opt = dict(self.io_backend_opt)
            self.file_client = FileClient(backend_opt.pop('type'), **backend_opt)

        paths = self.paths[idx]
        size = self.size

        # load images
        img_gt = imfrombytes(self.file_client.get(paths['gt']), float32=True)
        img_lq = imfrombytes(self.file_client.get(paths['lq']), float32=True)

        # ---- Resize both to fixed size ----
        img_gt = cv2.resize(img_gt, (size, size), interpolation=cv2.INTER_CUBIC)
        img_lq = cv2.resize(img_lq, (size, size), interpolation=cv2.INTER_CUBIC)

        # ---- Optional YCbCr ----
        if self.opt.get('color', None) == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # ---- Convert to Tensor (RGB) ----
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)  # default: [0,1]
        if self.opt.get("img_range", "-1_1") == "-1_1":
            img_gt = img_gt * 2.0 - 1.0
            img_lq = img_lq * 2.0 - 1.0
        # ---- Optional Normalization (mean/std) ----
        if self.mean is not None and self.std is not None:
            TF.normalize(img_gt, self.mean, self.std, inplace=True)
            TF.normalize(img_lq, self.mean, self.std, inplace=True)

        # ---- (NEW) Load offline m0 (.npy) ----
        m0 = None
        m0_path = None
        m0_folder = self.opt.get("dataroot_m0", None)
        if m0_folder is not None:
            stem = os.path.splitext(os.path.basename(paths["lq"]))[0]
            m0_path = os.path.join(m0_folder, stem + self.opt.get("m0_ext", ".npy"))
            if not os.path.isfile(m0_path):
                raise FileNotFoundError(f"[ResizePairedDataset] missing m0: {m0_path}")

            m0_np = np.load(m0_path).astype(np.float32)  # [H,W] in [0,1]
            m0_np = cv2.resize(m0_np, (size, size), interpolation=cv2.INTER_NEAREST)
            m0 = torch.from_numpy(m0_np).unsqueeze(0)    # [1,H,W]

            # optional range: 0_1 or -1_1
            if self.opt.get("m0_range", "0_1") == "-1_1":
                m0 = m0 * 2.0 - 1.0

        return {
            "gt": img_gt,
            "lq": img_lq,
            "m0": m0,               # (NEW) tensor [1,H,W] or None
            "gt_path": paths['gt'],
            "lq_path": paths['lq'],
            "m0_path": m0_path,     # (NEW) str or None
        }


    def __len__(self):
        return len(self.paths)


class HazyPairDataset(Dataset):
    """
    简单成对去雾数据集：
      dataroot/hazy/*.png  为含雾图像
      dataroot/gt/*.png    为清晰 GT

    返回：
      {
        "lq": hazy [3,H,W] in [0,1],
        "gt": gt   [3,H,W] in [0,1],
        "m0": optional [1,H,W] in [0,1],
        "hazy_path": str,
        "gt_path": str,
      }
    """

    def __init__(self, opt):
        super().__init__()
        if hasattr(opt, "get"):
            dataroot    = opt.get("dataroot", None)
            size        = opt.get("size", 512)
            dynamic_m0  = opt.get("dynamic_m0", False)
        self.dataroot = dataroot
        self.size = size
        self.dynamic_m0 = dynamic_m0

        hazy_dir = os.path.join(dataroot, "hazy")
        gt_dir   = os.path.join(dataroot, "gt")

        hazy_paths = sorted(glob(os.path.join(hazy_dir, "*.png")))
        gt_paths   = sorted(glob(os.path.join(gt_dir, "*.png")))

        # 按 stem 对齐
        hazy_map = {Path(p).stem: p for p in hazy_paths}
        gt_map   = {Path(p).stem: p for p in gt_paths}
        common_stems = sorted(set(hazy_map.keys()) & set(gt_map.keys()))
        if not common_stems:
            raise RuntimeError(
                f"[HazyPairDataset] No pairs found under {hazy_dir} and {gt_dir}"
            )

        self.pairs = [
            {"hazy": hazy_map[stem], "gt": gt_map[stem], "stem": stem}
            for stem in common_stems
        ]
        print(f"[HazyPairDataset] Found {len(self.pairs)} pairs in '{hazy_dir}' and '{gt_dir}'.")

    def __len__(self):
        return len(self.pairs)

    def _load_and_resize_rgb(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.size is not None:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    def __getitem__(self, index):
        pair = self.pairs[index]
        hazy_img = self._load_and_resize_rgb(pair["hazy"])
        gt_img   = self._load_and_resize_rgb(pair["gt"])

        # 转为 tensor in [0,1]
        hazy = TF.to_tensor(hazy_img)  # [3,H,W], [0,1]
        gt   = TF.to_tensor(gt_img)

        sample = {
            "hazy": hazy,
            "gt": gt,
            "hazy_path": pair["hazy"],
            "gt_path": pair["gt"],
        }

        # 可选：动态计算 hazy mass map，供后续 Stage 使用（Stage1 不强制用）
        if self.dynamic_m0:
            with torch.no_grad():
                # compute_hazy_mass_map 期望 [B,3,H,W]
                hazy_batch = hazy.unsqueeze(0)  # [1,3,H,W]
                m0 = compute_hazy_mass_map(hazy_batch, input_range="0_1")  # [1,1,H,W]
                sample["m0"] = m0.squeeze(0)  # [1,H,W]

        return sample