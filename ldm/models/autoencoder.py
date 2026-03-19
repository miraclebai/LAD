from inspect import Parameter
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder, Decoder_Mix
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import HazeAwareEncoder, Decoder_Mix
from ldm.util import instantiate_from_config

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt
import random
import torchvision.transforms as transforms

from MFlow.aniso_mflow import HazeMFlowUnit
from hazy_mass.hazy_mass import compute_hazy_mass_map, HazyMassMapGenerator
from ldm.modules.color_guidance.guide_head import GuideOutputAffine
from ldm.modules.color_guidance.lcgm import LCGM


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        # if len(unexpected) > 0:
        #     print(f"Unexpected Keys: {unexpected}")

    def encode(self, x, return_encfea=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if return_encfea:
            return posterior, moments
        return posterior

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        x = x.to(memory_format=torch.contiguous_format).float()
        # x = x*2.0-1.0
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

class AutoencoderKLResi(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.decoder.fusion_w = fusion_w
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []

        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)
        self.synthesis_data = synthesis_data
        self.use_usm = use_usm
        self.test_gt = test_gt

        if freeze_dec:
            for name, param in self.named_parameters():
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # elif 'encoder' in name:
                #     param.requires_grad = True
                # elif 'quant_conv' in name and 'post_quant_conv' not in name:
                #     param.requires_grad = True
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)
        # untrainable_list = list(set(trainable_list).difference(set(missing_list)))
        # print('>>>>>>>>>>>>>>>>>untrainable_list>>>>>>>>>>>>>>>>>>>')
        # print(untrainable_list)

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing

    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = h
        return posterior, enc_fea

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(input)
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_, w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    def get_input(self, batch):
        input = batch['lq']
        gt = batch['gt']
        latent = batch['latent']
        sample = batch['sample']

        assert not torch.isnan(latent).any()

        input = input.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()
        latent = latent.to(memory_format=torch.contiguous_format).float() / 0.18215

        gt = gt * 2.0 - 1.0
        input = input * 2.0 - 1.0
        sample = sample * 2.0 -1.0

        return input, gt, latent, sample

    @torch.no_grad()
    def get_input_synthesis(self, batch, val=False, test_gt=False):

        jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        im_gt = batch['gt'].cuda()
        if self.use_usm:
            usm_sharpener = USMSharp().cuda()  # do usm sharpening
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs.degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob']
        if random.random() < self.configs.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs.degradation['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / self.configs.sf * scale),
                      int(ori_w / self.configs.sf * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob2']
        if random.random() < self.configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.configs.sf,
                          ori_w // self.configs.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.configs.sf,
                          ori_w // self.configs.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        gt_size = self.configs.degradation['gt_size']
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.sf)
        self.lq, self.gt = im_lq, im_gt

        self.lq = F.interpolate(
                self.lq,
                size=(self.gt.size(-2),
                      self.gt.size(-1)),
                mode='bicubic',
                )

        self.latent = batch['latent'] / 0.18215
        self.sample = batch['sample'] * 2 - 1.0
        # training pair pool
        if not val:
            self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq = self.lq*2 - 1.0
        self.gt = self.gt*2 - 1.0

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq
        y = self.gt
        x = x.to(self.device)
        y = y.to(self.device)

        if self.test_gt:
            return y, y, self.latent.to(self.device), self.sample.to(self.device)
        else:
            return x, y, self.latent.to(self.device), self.sample.to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.synthesis_data:
            inputs, gts, latents, _ = self.get_input_synthesis(batch, val=False)
        else:
            inputs, gts, latents, _ = self.get_input(batch)
        reconstructions, posterior = self(inputs, latents)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, gts, latents, _ = self.get_input(batch)

        reconstructions, posterior = self(inputs, latents)
        aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], 
                 prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        if self.synthesis_data:
            x, gts, latents, samples = self.get_input_synthesis(batch, val=False)
        else:
            x, gts, latents, samples = self.get_input(batch)
        x = x.to(self.device)
        latents = latents.to(self.device)
        samples = samples.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, latents)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                samples = self.to_rgb(samples)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        log["gts"] = gts
        log["samples"] = samples
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x



class HazeAutoencoderKLResi(pl.LightningModule):
    """
    HazeAutoencoderKLResi:
      - 使用 HazeAwareEncoder（内部带 M-flow）替代原始 Encoder。
      - 在 encode() 中根据输入 hazy 图像自动计算 hazy mass map M0，并传给 HazeAwareEncoder。
      - 保持与原 AutoencoderKLResi 大部分接口一致，方便直接替换。

    约定：
      * 输入/输出图像在本类内部仍然采用 StableSR 习惯：
          - 从 dataset 进来是 [0,1]
          - 在 get_input() / get_input_synthesis() 里统一映射到 [-1,1]
      * hazy mass map M0 在 [0,1]，shape [B,1,H,W]，在 encode() 中从 [-1,1] 的图像恢复到 [0,1] 后计算。
      * HazeAwareEncoder 在 __init__ 中应设置 use_mflow=True, m_channels=1（或与 mass map 一致）。
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 # ===== 新增：LCGM 旁路 =====
                 use_lcgm: bool = True,
                 lcgm_base_ch: int = 64,
                 lcgm_guide_channels: int = 32,
                 num_scales = 4,
                 lcgm_use_luma = True,
                 guide_head_hidden = 128,
                 *args, **kwargs
                ):
        super().__init__()

        self.image_key = image_key

        # ---------- 编码器 / 解码器 ----------
        # 使用 HazeAwareEncoder（带 M-flow），保持 Decoder_Mix 不变
        self.encoder = HazeAwareEncoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.decoder.fusion_w = fusion_w

        # ---------- VAE 相关 ----------
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        # ---------- 颜色可视化 ----------
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        # ---------- 其他配置 ----------
        self.synthesis_data = synthesis_data 
        self.use_usm = use_usm
        self.test_gt = test_gt

        # ---------- haze mass map 相关 ----------
        self.mass_estimator = HazyMassMapGenerator()

        # ---------- LCGM 相关 ----------
        self.guide_head_hidden = guide_head_hidden

        self.lcgm = None
        self.guide_head = None

        # 2) 纯参数方式启用（兼容你现在写法）
        self.lcgm = LCGM(
            in_channels=3,
            base_channels=lcgm_base_ch,
            guide_channels=lcgm_guide_channels,
            num_scales=num_scales,
            use_luma=lcgm_use_luma,
            
        )

        # 只要 lcgm 存在，就启用 guide_head（输出端可选校正头）
        if self.lcgm is not None:
            guide_channels = getattr(self.lcgm, "guide_channels", None)
            if guide_channels is None:
                guide_channels = lcgm_guide_channels
            self.guide_head = GuideOutputAffine(
                guide_channels=int(guide_channels),
                hidden_channels=int(self.guide_head_hidden),
                out_channels=3
            )
            self.use_lcgm = True
        else:
            self.use_lcgm = False


        # ---------- 从 ckpt 恢复 ----------
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []
        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)

        
        # ---------- 冻结参数策略 ----------
        '''
        if freeze_dec:
            for name, param in self.named_parameters():
                # 解码器中融合模块
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # 判别器
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                # M-flow 相关参数（希望在冻结 decoder 时也能训练 M-flow）
                elif 'mflow' in name or 'mid_mflow' in name:
                    param.requires_grad = True
                # 如果你希望 haze mass estimator 也可学习，可以在这里加：
                elif 'mass_estimator' in name:
                    param.requires_grad = True
                # 新增：LCGM 永远允许训练（如果启用）
                elif self.use_lcgm and ('lcgm' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            '''

        if freeze_dec:
            # Stage3：freeze encoder + quant_conv
            for p in self.encoder.parameters():      p.requires_grad = False
            for p in self.quant_conv.parameters():   p.requires_grad = False

            # train decoder + post_quant_conv + lcgm + guide_head + disc
            for p in self.decoder.parameters():         p.requires_grad = True
            for p in self.post_quant_conv.parameters(): p.requires_grad = True
            if self.lcgm is not None:
                for p in self.lcgm.parameters():        p.requires_grad = True
            if self.guide_head is not None:
                for p in self.guide_head.parameters():  p.requires_grad = True

            # discriminator
            if hasattr(self.loss, "discriminator"):
                for p in self.loss.discriminator.parameters():
                    p.requires_grad = True

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)


    # -------------------------------------------------------------------------
    # ckpt 加载
    # -------------------------------------------------------------------------
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                # 兼容 Stable Diffusion 风格的 ckpt key
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if not only_model:
            missing, unexpected = self.load_state_dict(sd, strict=False)
        else:
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing

    # -------------------------------------------------------------------------
    # encode / decode / forward
    # -------------------------------------------------------------------------
    def encode(self, x: torch.Tensor, *args, **kwargs):
        """
        x must be in range [-1, 1]
        编码阶段：
          1. 由输入 hazy 图像 x 计算 hazy mass map M0；
          2. 调用 HazeAwareEncoder(x, M0, return_fea=True)，得到特征 h 和中间特征 enc_fea；
          3. 通过 quant_conv 得到 moments，并构造 DiagonalGaussianDistribution。

        返回：
          posterior: DiagonalGaussianDistribution
          enc_fea  : list of encoder 中间特征（用于 Decoder_Mix 融合）
        """
        use_mflow = getattr(self.encoder, "use_mflow", False)
        if use_mflow:
            m0 = self.mass_estimator(x)
            h, m_l, enc_fea = self.encoder(x, m0, return_fea=True, return_m_pyr=False)
        else:
            h, enc_fea = self.encoder(x, return_fea=True)
            m_l = None

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        guides = None
        if self.use_lcgm and (self.lcgm is not None):
            guides = self.lcgm(x)   # 最稳：直接读 RGB，不读 enc_fea
            enc_fea = {"enc_fea": enc_fea, "guide": guides}

        if use_mflow:
            return posterior, enc_fea, m_l
        else:
            return posterior, enc_fea


    def encode_gt(self, x, new_encoder):
        """
        兼容原版接口：给定一个新的 encoder（例如 gt 分支的 encoder）编码 x。
        """
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea = None):
        """
        使用 Decoder_Mix 解码：
          - 将 VAE latent 通过 post_quant_conv 映射到 decoder 的 z_channels；
          - decoder(z, enc_fea) 输出重建图像。
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)  # enc_fea 可能是 list 或 dict

        guides = None
        if isinstance(enc_fea, dict):
            guides = enc_fea.get("guide", None)

        if self.use_lcgm and self.guide_head is not None and guides is not None:
            dec = self.guide_head(dec, guides)

        return dec


    def forward(self, input, latent=None, sample_posterior: bool = True):
        """
        统一的前向接口：

        Stage1（纯 VAE 去雾预训练）：
        - 只传 input（hazy），latent=None
        - encode(input) 得到 posterior, enc_fea_lq
        - z 由 posterior.sample() 或 posterior.mode() 得到
        - decode(z, enc_fea_lq) -> 重建图像，和 GT 做重建 + 感知 / 对抗 loss

        Stage2/后续（残差/外部 latent 模式）：
        - 传入 input（hazy） 和外部 latent（如由另一个 VAE 对 GT 编码）
        - 仍然 encode(input) 只为得到 enc_fea_lq 与 posterior (用于 KL 和判别器)
        - decode(external_latent, enc_fea_lq)
        """
        posterior, enc_fea_lq, _ = self.encode(input)

        if latent is None:
            # 纯 VAE 模式
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
        else:
            # 残差 / 外部 latent 模式
            z = latent

        dec = self.decode(z, enc_fea_lq)
        return dec, posterior


    # -------------------------------------------------------------------------
    # 训练用队列（原 StableSR 功能，保持不变）
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """训练对池，用于增加 batch 内退化多样性（原 StableSR 逻辑）。"""
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b * 50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_, w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    # -------------------------------------------------------------------------
    # 数据获取（Haze 用 PairedDataset：hazy + gt）
    # -------------------------------------------------------------------------
    def get_input(self, batch):
        """
        Haze 场景下的数据获取：

        期望最基本的字段：
        - 'lq' : 含雾图像 [0,1]
        - 'gt' : 清晰图像 [0,1]

        可选字段（用于残差/外部 latent 模式或日志）：
        - 'latent': 预计算 latent（未除 0.18215）
        - 'sample': 额外 sample 图像 [0,1]（若无则默认用 gt）

        映射规则：
        - lq / gt / sample: [0,1] -> [-1,1]
        - latent（如存在）: /0.18215
        """
        # 1) 取输入 & GT
        hazy = batch["hazy"]
        gt = batch["gt"]

        '''
        # 2) 可选 latent / sample
        latent = batch.get("latent", None)
        sample = batch.get("sample", gt)
        '''

        # 3) to float & contiguous
        hazy = hazy.to(memory_format=torch.contiguous_format).float()
        gt    = gt.to(memory_format=torch.contiguous_format).float()
        
        '''
        sample = sample.to(memory_format=torch.contiguous_format).float()
        if latent is not None:
            latent = latent.to(memory_format=torch.contiguous_format).float() / 0.18215
            assert not torch.isnan(latent).any()
        '''

        # 4) 把图像从 [0,1] 映射到 [-1,1]
        gt    = gt * 2.0 - 1.0
        hazy = hazy * 2.0 - 1.0
        # sample = sample * 2.0 - 1.0

        return hazy, gt


    @torch.no_grad()
    def get_input_synthesis(self, batch, val=False, test_gt=False):
        """
        保留原 StableSR 的合成退化 pipeline（一般在 haze 任务中可以不用）。
        注意：这里仍然是 “gt -> lq” 的 SR 风格逻辑。
        """
        jpeger = DiffJPEG(differentiable=False).cuda()
        im_gt = batch['gt'].cuda()
        if self.use_usm:
            usm_sharpener = USMSharp().cuda()
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation ----------------------- #
        out = filter2D(im_gt, kernel1)
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.configs.degradation['resize_prob'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        gray_noise_prob = self.configs.degradation['gray_noise_prob']
        if random.random() < self.configs.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation ---------------------- #
        if random.random() < self.configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)

        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.configs.degradation['resize_prob2'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(int(ori_h / self.configs.sf * scale),
                  int(ori_w / self.configs.sf * scale)),
            mode=mode,
        )

        gray_noise_prob = self.configs.degradation['gray_noise_prob2']
        if random.random() < self.configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        if random.random() < 0.5:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // self.configs.sf,
                      ori_w // self.configs.sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // self.configs.sf,
                      ori_w // self.configs.sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)

        im_lq = torch.clamp(out, 0, 1.0)

        gt_size = self.configs.degradation['gt_size']
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.sf)
        self.lq, self.gt = im_lq, im_gt

        self.lq = F.interpolate(
            self.lq,
            size=(self.gt.size(-2),
                  self.gt.size(-1)),
            mode='bicubic',
        )

        self.latent = batch['latent'] / 0.18215
        self.sample = batch['sample'] * 2 - 1.0

        if not val:
            self._dequeue_and_enqueue()

        self.lq = self.lq.contiguous()
        self.lq = self.lq * 2 - 1.0
        self.gt = self.gt * 2 - 1.0

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq.to(self.device)
        y = self.gt.to(self.device)

        if self.test_gt:
            return y, y, self.latent.to(self.device), self.sample.to(self.device)
        else:
            return x, y, self.latent.to(self.device), self.sample.to(self.device)

    # -------------------------------------------------------------------------
    # 训练 / 验证 step（保持接口）
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        inputs, gts = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(
                gts, reconstructions, posterior, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(
                gts, reconstructions, posterior, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, gts = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(
            gts, reconstructions, posterior, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        discloss, log_dict_disc = self.loss(
            gts, reconstructions, posterior, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    # -------------------------------------------------------------------------
    # 优化器
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            # list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            # list(self.quant_conv.parameters())+
            list(self.post_quant_conv.parameters())+
            list(self.lcgm.parameters()) +
            list(self.guide_head.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )

        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # -------------------------------------------------------------------------
    # 日志可视化
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        """
        额外可视化：
          - inputs: hazy
          - gts   : gt
          - samples: text-guided sample
          - reconstructions
          - mass_map: 由输入 hazy 计算出的 hazy mass map M0
        """

        log = dict()

        if self.synthesis_data:
            x, gts, latents, samples = self.get_input_synthesis(batch, val=False)
        else:
            x, gts = self.get_input(batch)

        x = x.to(self.device)

        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                xrec = self.to_rgb(xrec)
            log["reconstructions"] = xrec

        # 计算并记录 hazy mass map（输入 x 为 [-1,1]）
        # x 此时也是 [-1,1]
        if getattr(self.encoder, "use_mflow", False) and hasattr(self, "mass_estimator"):
            mass_map = self.mass_estimator(x)  # [B,1,H,W] in [0,1]
            log["mass_map"] = mass_map

        log["inputs"] = x
        log["gts"] = gts

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x