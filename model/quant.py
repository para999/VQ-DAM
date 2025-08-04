from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from model import dist


class VectorQuantizer2(nn.Module):
    def __init__(self,
                 vocab_size=4096,
                 Cvae=32,
                 using_znorm=False,
                 beta=0.25,
                 res_counts=0,
                 patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                 res_ratio=0.5,
                 share_res=4  # share_quant_res: args.qsr
                 ):
        super().__init__()
        self.vocab_size, self.Cvae = vocab_size, Cvae
        self.using_znorm = using_znorm
        self.beta = beta
        self.patch_nums = patch_nums

        self.quant_res_ratio = res_ratio
        if share_res == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_res = PhiNonShared([(Phi(Cvae, res_ratio) if abs(res_ratio) > 1e-6 else nn.Identity()) for _ in
                                           range(res_counts or len(self.patch_nums))])
        elif share_res == 1:  # all shared: only a single \phi for K scales
            self.quant_res = PhiShared(Phi(Cvae, res_ratio) if abs(res_ratio) > 1e-6 else nn.Identity())
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_res = PhiPartiallyShared(
                nn.ModuleList([(Phi(Cvae, res_ratio) if abs(res_ratio) > 1e-6 else nn.Identity()) for _ in
                               range(share_res)]))

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

    def embed_init(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) -> str:
        return (f'{self.patch_nums}, znorm={self.using_znorm}, beta={self.beta}  '
                f'|  Length={len(self.patch_nums)}, quant_res_ratio={self.quant_res_ratio}')

    def forward(self, f, ret_usages=False):
        # f: BCHW; return Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f.dtype
        if dtype != torch.float32:
            f = f.float()
        B, C, H, W = f.shape
        f_no_grad = f.detach()

        f_res = f_no_grad.clone()
        f_hat = torch.zeros_like(f_res)

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss = 0.0
            vocab_hit = torch.zeros(self.vocab_size, dtype=torch.float, device=f.device)  # shape: V
            len_nums = len(self.patch_nums)

            for i, patch_num in enumerate(self.patch_nums):
                # from small to large. find the nearest embedding
                if self.using_znorm:
                    res = F.interpolate(f_res, size=(patch_num, patch_num), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (i != len_nums - 1) else f_res.permute(0, 2, 3, 1).reshape(-1, C)  # res: NC
                    res = F.normalize(res, dim=-1)  # res: NC
                    idx = torch.argmax(res @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)  # idx: N
                else:
                    res = F.interpolate(f_res, size=(patch_num, patch_num), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (i != len_nums - 1) else f_res.permute(0, 2, 3, 1).reshape(-1, C)
                    d = torch.sum(res.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d.addmm_(res, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx = torch.argmin(d, dim=1)

                hit_V = idx.bincount(minlength=self.vocab_size).float()

                if self.training:
                    if dist.initialized():
                        handler = tdist.all_reduce(hit_V, async_op=True)

                # calculate loss
                idx = idx.view(B, patch_num, patch_num)  # idx: BHW
                h = F.interpolate(self.embedding(idx).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (i != len_nums - 1) else self.embedding(idx).permute(0, 3, 1, 2).contiguous()  # h: BCHW
                h = self.quant_res[i / (len_nums - 1)](h)  # h: BCHW

                f_hat = f_hat + h
                f_res -= h

                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[i].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[i].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[i].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / len_nums
            f_hat = (f_hat.data - f_no_grad).add_(f)

        world_size = tdist.get_world_size() if tdist.is_initialized() else 1
        margin = world_size * (f.numel() / f.shape[1]) / self.vocab_size * 0.08
        # margin = patch_num * patch_num / 100
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[i] >= margin).float().mean().item() * 100
                      for i, patch_num in enumerate(self.patch_nums)]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss

    def embed_to_fhat(self, multi_scale_h, max_scale=True, last_one=False):
        # multi_scale_h: List[torch.Tensor]; return Union[List[torch.Tensor], torch.Tensor]
        f_hat_list = []  # f_hat_list: [BCHW]
        B = multi_scale_h[0].shape[0]
        H = W = self.patch_nums[-1]
        len_nums = len(self.patch_nums)
        if max_scale:
            f_hat = multi_scale_h[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for i, patch_num in enumerate(self.patch_nums):  # from small to large
                h = multi_scale_h[i]  # h: BCHW
                if i < len(self.patch_nums) - 1:
                    h = F.interpolate(h, size=(H, W), mode='bicubic')
                h = self.quant_res[i / (len_nums - 1)](h)
                f_hat.add_(h)
                if last_one:
                    f_hat_list = f_hat
                else:
                    f_hat_list.append(f_hat.clone())
        else:
            # (we'll interpolate every token map to the max H W, like above)
            f_hat = multi_scale_h[0].new_zeros(B, self.Cvae, self.patch_nums[0], self.patch_nums[0], dtype=torch.float32)
            for i, patch_num in enumerate(self.patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(patch_num, patch_num), mode='bicubic')
                h = self.quant_res[i / (len_nums - 1)](multi_scale_h[i])
                f_hat.add_(h)
                if last_one:
                    f_hat_list = f_hat
                else:
                    f_hat_list.append(f_hat)

        return f_hat_list

    def f_to_idx_or_fhat(self, f, to_fhat, patch_nums=None):
        # f: BCHW; return List[Union[torch.Tensor, torch.LongTensor]]; idx: BL; to_fhat: bool
        # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f.shape
        f_no_grad = f.detach()
        f_res = f_no_grad.clone()
        f_hat = torch.zeros_like(f_res)

        f_hat_or_idx_list = []
        # f_hat_or_idx_list: List[torch.Tensor]

        patch_hws = [(patch_num, patch_num) if isinstance(patch_num, int) else (patch_num[0], patch_num[1])
                     for patch_num in (patch_nums or self.patch_nums)]  # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        len_nums = len(patch_hws)
        for i, (patch_h, patch_w) in enumerate(patch_hws):  # from small to large
            # find the nearest embedding
            z = F.interpolate(f_res, size=(patch_h, patch_w), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (i != len_nums - 1) else f_res.permute(0, 2, 3, 1).reshape(-1, C)  # z: NC
            if self.using_znorm:
                z = F.normalize(z, dim=-1)  # z: NC
                idx = torch.argmax(z @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)  # idx: N
            else:
                d = torch.sum(z.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d.addmm_(z, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx = torch.argmin(d, dim=1)

            idx = idx.view(B, patch_h, patch_w)  # idx: BHW
            # h: BCHW
            h = F.interpolate(self.embedding(idx).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (i != len_nums - 1) else self.embedding(idx).permute(0, 3, 1, 2).contiguous()
            h = self.quant_res[i / (len_nums - 1)](h)
            f_hat.add_(h)
            f_res.sub_(h)
            f_hat_or_idx_list.append(f_hat.clone() if to_fhat else idx.reshape(B, patch_h * patch_w))

        return f_hat_or_idx_list


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, res_ratio):
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.res_ratio = abs(res_ratio)

    def forward(self, h):
        # h: BCHW
        return h.mul(1 - self.res_ratio) + super().forward(h).mul_(self.res_ratio)


class PhiShared(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def __getitem__(self, _):
        return self.phi


class PhiPartiallyShared(nn.Module):
    def __init__(self, res_list):
        super().__init__()
        self.res_list = res_list
        K = len(res_list)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1):
        #  at_from_0_to_1: float
        return self.res_list[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, res_list):
        super().__init__(res_list)
        K = len(res_list)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1):
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self):
        return f'ticks={self.ticks}'
