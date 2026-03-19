"""make variations of input image (pad-load like previous version; keep other logic unchanged)"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F

from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


# === 固定 pad 倍数：你之前遇到过 UNet skip-cat / Swin window 的整除问题
# 这里稳妥起见用 128（满足 /8 后还能被 16 整除：128 = 8*16）
PAD_MULTIPLE = 128


def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   # e.g. [250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)


def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model


def _ceil_to_multiple(x: int, base: int) -> int:
	return int(math.ceil(x / base) * base)


def load_img_pad_keep(path: str):
	"""
	原尺寸读取，不 resize；
转成 [-1,1] Tensor，并 pad 到 PAD_MULTIPLE 的倍数（右/下补零）。
返回：
  img_pad: [1,3,Hpad,Wpad] in [-1,1]
  meta: orig_h, orig_w, pad_h, pad_w
	"""
	image = Image.open(path).convert("RGB")
	np_img = np.array(image).astype(np.float32) / 255.0  # H,W,3 in [0,1]
	orig_h, orig_w = np_img.shape[:2]

	t = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
	t = 2.0 * t - 1.0

	pad_h = _ceil_to_multiple(orig_h, PAD_MULTIPLE)
	pad_w = _ceil_to_multiple(orig_w, PAD_MULTIPLE)

	pad_bottom = pad_h - orig_h
	pad_right = pad_w - orig_w

	t_pad = F.pad(t, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)

	meta = {
		"orig_h": orig_h,
		"orig_w": orig_w,
		"pad_h": pad_h,
		"pad_w": pad_w,
		"pad_right": pad_right,
		"pad_bottom": pad_bottom,
	}

	print(f"loaded input image of size ({orig_w}, {orig_h}) from {path} -> padded to ({pad_w}, {pad_h})")
	return t_pad, meta


def crop_back(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
	"""
	x: [B,C,H,W] -> crop to [:orig_h, :orig_w]
	"""
	return x[:, :, :orig_h, :orig_w]


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--init-img", type=str, nargs="?", help="path to the input image",
						default="/disk8t/baijy/dataset/I-HAZE/1")
	parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
						default="results/output_1")
	parser.add_argument("--ddpm_steps", type=int, default=1000, help="number of ddpm sampling steps")
	parser.add_argument("--C", type=int, default=4, help="latent channels")
	parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16")
	parser.add_argument("--n_samples", type=int, default=1,
						help="how many samples to produce for each given prompt. A.k.a batch size")
	parser.add_argument("--config", type=str, default="configs/infer.yaml",
						help="path to config which constructs model")
	parser.add_argument("--ckpt", type=str,
						default="/disk8t/baijy/Stable_main_2/logs/2025-12-12T23-17-28_stage2/checkpoints/epoch=000016-v2.ckpt",
						help="path to checkpoint of model")
	parser.add_argument("--vqgan_ckpt", type=str,
						default="/disk8t/baijy/Stable_main_2/logs/2025-12-17T10-24-28_dec/checkpoints/epoch=000004-v9.ckpt",
						help="path to checkpoint of VQGAN model")
	parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
	parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast",
						help="evaluate at this precision")
	parser.add_argument("--input_size", type=int, default=512,
						help="(kept) no longer used for resizing in pad-load mode")
	parser.add_argument("--dec_w", type=float, default=0.98,
						help="weight for combining VQGAN and Diffusion")
	parser.add_argument("--colorfix_type", type=str, default="nofix",
						help="Color fix type: adain (used in paper); wavelet; nofix")

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	# VQGAN / AE
	vqgan_config = OmegaConf.load("configs/stage3.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w

	seed_everything(opt.seed)

	# Diffusion model
	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.to(device)

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	# ===== 重要：pad-load 后不同图尺寸不同，无法 torch.cat 组成 batch
	# 为保证“其他处理方式不改”，这里按图片逐张处理（等价于 batch=1）
	if opt.n_samples != 1:
		print(f"[Warn] pad-load mode forces per-image inference. Your --n_samples={opt.n_samples} will be effectively treated as 1.")

	# list images
	img_list_ori = sorted(os.listdir(opt.init_img))
	# 跳过已经有输出的（输出名仍按 basename.png）
	img_paths = []
	for item in img_list_ori:
		basename = os.path.splitext(os.path.basename(item))[0]
		out_file = os.path.join(outpath, basename + ".png")
		if os.path.exists(out_file):
			continue
		img_paths.append(os.path.join(opt.init_img, item))

	# schedule setup（保持原逻辑）
	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)

	# param print（保持原逻辑）
	param_list = []
	untrain_paramlist = []
	name_list = []
	for k, v in model.named_parameters():
		if 'spade' in k or 'structcond_stage_model' in k:
			param_list.append(v)
		else:
			name_list.append(k)
			untrain_paramlist.append(v)
	trainable_params = sum(p.numel() for p in param_list)
	untrainable_params = sum(p.numel() for p in untrain_paramlist)
	print(name_list)
	print(trainable_params)
	print(untrainable_params)

	param_list = []
	untrain_paramlist = []
	for k, v in vq_model.named_parameters():
		if 'fusion_layer' in k:
			param_list.append(v)
		elif 'loss' not in k:
			untrain_paramlist.append(v)
	trainable_params += sum(p.numel() for p in param_list)
	print(trainable_params)
	print(untrainable_params)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext

	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()

				for p in tqdm(img_paths, desc="Processing images"):
					# ===== pad-load (new) =====
					init_image, meta = load_img_pad_keep(p)
					init_image = init_image.to(device).clamp(-1, 1)  # [1,3,Hpad,Wpad]

					# ===== 原逻辑不改：encode / cond / sample / decode =====
					init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

					text_init = ['(masterpiece:2), (best quality:2), (realistic:2), (very clear:2),(haze-free:2)'] * init_image.size(0)
					semantic_c = model.cond_stage_model(text_init)

					noise = torch.randn_like(init_latent)
					x_T = noise

					posterior, enc_fea, m_l = model.first_stage_model.encode(init_image)
					struct_cond = m_l

					samples, _ = model.sample(
						cond=semantic_c,
						struct_cond=struct_cond,
						batch_size=init_image.size(0),
						timesteps=opt.ddpm_steps,
						time_replace=opt.ddpm_steps,
						x_T=x_T,
						return_intermediates=True
					)

					posterior, enc_fea, m_l = vq_model.encode(init_image)
					x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea)

					# colorfix（原逻辑不改）
					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, init_image)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, init_image)

					# ===== crop back to original size (new) =====
					x_samples = crop_back(x_samples, meta["orig_h"], meta["orig_w"])
					init_image_cropped = crop_back(init_image, meta["orig_h"], meta["orig_w"])

					# to [0,1]（原逻辑不改）
					x_samples = torch.clamp((x_samples + 1.0) / 2, min=0, max=1.0)

					# save（按原命名：basename.png）
					basename = os.path.splitext(os.path.basename(p))[0]
					x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
					x_sample = x_sample.astype(np.uint8)
					Image.fromarray(x_sample).save(os.path.join(outpath, basename + '.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n\nEnjoy.")


if __name__ == "__main__":
	main()
