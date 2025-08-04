"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.quant import VectorQuantizer2
from model.restorer import DAR


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        # conv_shortcut=False, conv_shortcut always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)

        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # B,HW,C
        k = k.view(B, C, H * W).contiguous()  # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w = F.softmax(w, dim=2)

        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()

        return x + self.proj_out(h)


def make_attn(in_channels, using_attn=True):
    return AttnBlock(in_channels) if using_attn else nn.Identity()


class DRE(nn.Module):
    def __init__(self,
                 channel=64,
                 channel_mult=(1, 1, 2, 2, 4),
                 num_resblocks=2,
                 dropout=0.0,
                 in_channels=3,
                 z_channels=32,
                 using_attn=True,
                 using_mid_attn=True):
        super().__init__()
        self.channel = channel
        self.num_resolutions = len(channel_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)  # 16
        self.num_res_blocks = num_resblocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.channel, 3, 1, 1)

        in_channel_mult = (1,) + tuple(channel_mult)  # (1, 1, 1, 2, 2, 4)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channel * in_channel_mult[i_level]
            block_out = channel * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_attn:
                    attn.append(make_attn(block_in, using_attn=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1 and i_level % 2 == 0:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_attn=using_mid_attn)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, z_channels, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        # for degradation classification
        self.cls_pool = nn.AdaptiveAvgPool2d(2)
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, z_channels * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(z_channels * 4, z_channels * 8),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        # downsampling
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1 and i_level % 2 == 0:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))  # B, channel*channel_mult[-1](256), 16, 16

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        h = self.pool(h)

        return h


class VQDAM(nn.Module):
    def __init__(self,
                 upscale=4,
                 vocab_size=4096,
                 z_channels=32,
                 channel=64,
                 dropout=0.0,
                 patch_nums=(1, 2, 4, 8, 16),
                 beta=0.25,  # commitment loss weight
                 quant_kernel_size=3,  # quant conv kernel size
                 res_ratio=0.5,  # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
                 share_res=4,  # use 4 \phi layers for K scales: partially-shared \phi
                 res_counts=0,  # if is 0: automatically set to len(patch_nums)
                 using_znorm=False,  # whether to normalize when computing the nearest neighbors
                 test=False,  # test mode
                 ):
        super().__init__()
        self.test = test
        self.vocab_size, self.Cvae = vocab_size, z_channels
        self.encoder = DRE(dropout=dropout, channel=channel, z_channels=z_channels, in_channels=3,
                               channel_mult=(1, 1, 2, 2, 4), num_resblocks=2, using_attn=True, using_mid_attn=True)
        self.decoder = DAR(upscale=upscale)

        self.quantize = VectorQuantizer2(vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
                                         res_counts=res_counts, patch_nums=patch_nums, res_ratio=res_ratio,
                                         share_res=share_res)
        self.quant_conv = torch.nn.Conv2d(in_channels=self.Cvae, out_channels=self.Cvae, kernel_size=quant_kernel_size,
                                          stride=1, padding=quant_kernel_size // 2)
        self.post_quant_conv = torch.nn.Conv2d(in_channels=self.Cvae, out_channels=self.Cvae,
                                               kernel_size=quant_kernel_size, stride=1, padding=quant_kernel_size // 2)

        if self.test:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    def forward(self, lr, ret_usages=False):
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(lr)), ret_usages=ret_usages)
        # print(f_hat.shape)
        sr = self.decoder(lr, self.post_quant_conv(f_hat))
        return sr, usages, vq_loss

    def load_state_dict(self, state_dict, strict=True):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != \
                self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict)


if __name__ == '__main__':
    upscale = 4
    height = 504
    width = 504
    batch_size = 1

    model = VQDAM(test=False, upscale=upscale).eval().cuda()

    HR = torch.randn(batch_size, 3, height, width).cuda()
    LR = torch.randn(batch_size, 3, height // upscale, width // upscale).cuda()
    print('HR shape', HR.shape)
    print('LR shape', LR.shape)

    SR, _, _ = model(LR, ret_usages=True)
    print('SR shape', SR.shape)

