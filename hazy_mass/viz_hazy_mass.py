import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from hazy_mass import HazyMassMapGenerator   # 修改为你的真实包路径


# ------- Dataset 里的预处理 -------
def _to_tensor(img: Image.Image, normalize_to_neg_one_one=True) -> torch.Tensor:
    """
    转 tensor，并（可选）归一化到 [-1,1]
    """
    tensor = TF.to_tensor(img)  # [0,1]
    if normalize_to_neg_one_one:
        tensor = tensor * 2.0 - 1.0
    return tensor



# ------- 主程序 -------
def main():
    input_path = "/Users/baijingyuan/Desktop/result/00015.png"
    output_path = "./hazy_mass_map.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1) 读取图像
    img = Image.open(input_path).convert("RGB")

    # 2) 预处理（到 [-1,1]）
    img_tensor = _to_tensor(img, normalize_to_neg_one_one=True)   # [3,H,W]

    # 3) 扩展 batch 维度
    img_tensor = img_tensor.unsqueeze(0)   # [1,3,H,W]

    # 4) 创建 HazyMassMapGenerator
    generator = HazyMassMapGenerator(
        input_range="-1_1",   # 重要：你的 hazy 输入是 [-1,1]
        dcp_patch_size=15,
        dcp_omega=0.95,
        dcp_top_percent=0.001,
        alpha=0.7,
        beta=0.3,
    )

    # 5) 计算 haze mass map

    with torch.no_grad():
        hazy_map = generator(img_tensor)  # [1,1,H,W]

    # 6) squeeze 并转 numpy

    hazy_map_np = hazy_map.squeeze(0).squeeze(0).cpu().numpy()  # [H,W]
    hazy_gray = (hazy_map_np * 255).astype('uint8')
    Image.fromarray(hazy_gray).save(output_path)
   


if __name__ == "__main__":
    main()
