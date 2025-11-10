import torch, torchvision
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torchvision.models import convnext_base
from einops import rearrange
from einops.layers.torch import Rearrange
import time

import clip
import os
import sys
# from net.arch_util import LayerNorm2d
# from net.local_arch import Local_Base
import torchvision.transforms as transforms
from huggingface_hub import PyTorchModelHubMixin


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



class MULTI_shuffle_high_text(nn.Module):
    def __init__(self, ch_dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, lin_ch=512, topk_ratio=0.5):
        super(MULTI_shuffle_high_text, self).__init__()
        self.dim = ch_dim

        # 文本特征线性层
        self.text_fc = nn.Sequential(
            nn.Linear(lin_ch, lin_ch),
            nn.ReLU(inplace=True),
            nn.Linear(lin_ch, 2 * ch_dim)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(2 * ch_dim, 2 * ch_dim // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * ch_dim // 8, 2 * ch_dim, bias=False),
            nn.Sigmoid()
        )
        self.topk = int(2 * ch_dim * topk_ratio)

        # 卷积层
        self.conv1x1 = nn.Conv2d(self.topk, 2 * ch_dim, kernel_size=1)  # 注意输入通道是topk，输出是2*ch_dim
        self.conv_out = nn.Conv2d(2 * ch_dim, ch_dim, kernel_size=1)

        # 归一化层
        self.norm1 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm2 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm3 = LayerNorm(ch_dim, LayerNorm_type)

        # 注意力模块和FFN
        self.select_attn = Topm_CrossAttention_Restormer_Privileged(ch_dim, num_heads, bias=False)
        self.ffn = FeedForward(ch_dim, ffn_expansion_factor, bias)


    def forward(self, vi_featur, ir_featur, text_code):
        b, c, h, w = vi_featur.shape
        device = vi_featur.device

        # 拼接多模态特征
        concat_features = torch.cat([vi_featur, ir_featur], dim=1)  # [B, 2C, H, W]

        # SE通道权重计算
        squeeze = self.global_pool(concat_features).view(b, -1)  # [B, 2C]
        channel_weights = self.se_fc(squeeze)                     # [B, 2C]

        _, topk_indices = torch.topk(channel_weights, self.topk, dim=1)
        topk_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)  # [B, topk, H, W]

        topk_indices = topk_indices.to(device)
        selected_feat = concat_features.gather(1, topk_indices)  # [B, topk, H, W]

        img_featur = self.conv1x1(selected_feat)  # [B, 2*dim, H, W]

        text_code = self.text_fc(text_code)  # [B, 2*dim]
        _, soft_indices = torch.topk(text_code, k=2 * self.dim, dim=1)  # [B, 2*dim]

        soft_indices = soft_indices.to(device)
        shuffled_img = img_featur[torch.arange(b, device=device).unsqueeze(1), soft_indices, :, :]  # [B, 2*dim, H, W]

        q = self.conv_out(shuffled_img)  # [B, dim, H, W]

        img_feature2 = self.conv_out(concat_features)

        # img_feature2 = vi_featur + ir_featur


        att = self.select_attn(self.norm1(q), self.norm2(img_feature2))

        output = att + self.ffn(self.norm3(att))
        return output, img_feature2





class Topm_CrossAttention_Restormer_Privileged(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_q, x_kv):
        """
        x_q: 查询特征 [B, C, H, W]
        x_kv: 生成 k 和 v 的特征 [B, C, H, W]
        gt_img: ground truth 图像 [B, 1, H, W]
        """
        b, c, h, w = x_q.shape
        N = h * w

        q = self.q_dwconv(self.q(x_q))    # [B, C, H, W]
        kv = self.kv_dwconv(self.kv(x_kv))  # [B, 2C, H, W]
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        # mask = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, Head, N, N]

        mask = torch.zeros_like(attn).bool()  # [B, Head, N, N]

        topk_index = torch.topk(attn, k=int(C * 0.9), dim=-1)[1]

        mask.scatter_(-1, topk_index, True)

        masked_attn = torch.where(mask, attn, torch.full_like(attn, float('-inf')))
        attn_score = F.softmax(masked_attn, dim=-1)
        out = attn_score @ v

        out = out * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out

class ChannelSelectAndExpandConvWithConvNeXt(nn.Module):
    def __init__(self, channels, reduction=8, keep_ratio=0.7, alpha_init=1.0, max_epoch=30, use_convnext=True):
        super().__init__()
        self.channels = channels
        self.keep_channels = int(channels * keep_ratio)
        self.max_epoch = max_epoch
        self.use_convnext = use_convnext

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        if use_convnext:
            convnext = convnext_base(pretrained=True)
            self.convnext = nn.Sequential(*list(convnext.children())[:-2])
            self.convnext_fc = nn.Linear(1024, channels)
        else:
            self.convnext = None
            self.convnext_fc = None

        self.expand_conv = nn.Conv2d(self.keep_channels, channels, kernel_size=1, bias=False)
        self.suo_conv = nn.Conv2d(channels, 3, kernel_size=1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.size()

        squeeze = self.global_pool(x).view(b, -1)  # [B, C]
        channel_weights = self.fc(squeeze)          # [B, C]

        if self.use_convnext is not None:
            x_conv = self.suo_conv(x)
            x_resized = F.interpolate(x_conv, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                convnext_feat = self.convnext(x_resized)
                convnext_feat = convnext_feat.flatten(2).mean(dim=2)  # global average pooling [B, 1024]

            convnext_weights = self.convnext_fc(convnext_feat)    # [B, C]

            channel_weights = channel_weights + self.alpha * torch.sigmoid(convnext_weights)

        _, indices = torch.topk(channel_weights, self.keep_channels, dim=1)  # [B, keep_C]
        indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)   # [B, keep_C, H, W]

        selected_feat = x.gather(1, indices)                                # [B, keep_C, H, W]

        expanded_feat = self.expand_conv(selected_feat)              # [B, C, H, W]

        return expanded_feat

class GeometricResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.affine_fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 1),  # gamma & beta
        )

    def forward(self, fused_feat, orig_feat):
        guide = self.avgpool(orig_feat)  # [B, C, 1, 1]
        affine = self.affine_fc(guide)   # [B, 2C, 1, 1]
        gamma, beta = torch.chunk(affine, 2, dim=1)
        return fused_feat * (1 + gamma) + beta


class UPFusion(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 device="cuda:1",
                 ):
        super(UPFusion, self).__init__()
        self.device = device

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1_vi = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_ir = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_gt = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_shuffle_channel1 = MULTI_shuffle_high_text(ch_dim=dim, num_heads=heads[0],
                                                             LayerNorm_type=LayerNorm_type,
                                                             ffn_expansion_factor=ffn_expansion_factor,
                                                             bias=bias)  # encoder level1 shuffle
        self.channel_select = ChannelSelectAndExpandConvWithConvNeXt(channels=dim)
        self.GEO_vi = GeometricResidualBlock(channels=dim)
        self.GEO_ir = GeometricResidualBlock(channels=dim)
        self.expand_conv = nn.Conv2d(dim, dim * 2, kernel_size=1)



        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2_vi = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_ir = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_gt = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_shuffle_channel2 = MULTI_shuffle_high_text(ch_dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                             LayerNorm_type=LayerNorm_type,
                                                             ffn_expansion_factor=ffn_expansion_factor,
                                                             bias=bias)  # encoder level2 shuffle

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3_vi = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_ir = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_gt = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_shuffle_channel3 = MULTI_shuffle_high_text(ch_dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                             LayerNorm_type=LayerNorm_type,
                                                             ffn_expansion_factor=ffn_expansion_factor,
                                                             bias=bias)  # encoder level3 shuffle

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent_vi = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_ir = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_gt = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_shuffle_channel = MULTI_shuffle_high_text(ch_dim=int(dim * 2 ** 3), num_heads=heads[3],
                                                           LayerNorm_type=LayerNorm_type,
                                                           ffn_expansion_factor=ffn_expansion_factor,
                                                           bias=bias)  # latent latent shuffle

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.ir_imb = OverlapPatchEmbed(1, dim)



    def forward(self, vi_img, ir_img, text_code, epoch=True):

        def rgb_to_y_channel(x):
            # x: [B, 3, H, W], assumed to be RGB in [0,1]
            R = x[:, 0:1, :, :]
            G = x[:, 1:2, :, :]
            B = x[:, 2:3, :, :]
            Y = 0.299 * R + 0.587 * G + 0.114 * B  # Y channel of YUV
            return Y

        if vi_img.shape[1] == 3:
            vi_img = rgb_to_y_channel(vi_img)

        if ir_img.shape[1] == 3:
            ir_img = rgb_to_y_channel(ir_img)


        inp_enc_ir = self.ir_imb(ir_img)
        inp_enc_vi = self.patch_embed(vi_img)

        inp_enc_level1 = (inp_enc_ir + inp_enc_vi) / 2
        inp_enc_level1 = self.channel_select(inp_enc_level1)
        inp_enc_level1 = self.expand_conv(inp_enc_level1)
        c = inp_enc_level1.shape[1] // 2
        inp_enc_level1_vi = inp_enc_level1[:, :c, :, :]
        inp_enc_level1_ir = inp_enc_level1[:, c:, :, :]
        inp_enc_level1_vi = self.GEO_vi(inp_enc_level1_vi, inp_enc_vi)
        inp_enc_level1_ir = self.GEO_ir(inp_enc_level1_ir, inp_enc_ir)

        out_enc_level1_ir = self.encoder_level1_ir(inp_enc_level1_ir)

        inp_enc_level2_ir = self.down1_2(out_enc_level1_ir)
        out_enc_level2_ir = self.encoder_level2_ir(inp_enc_level2_ir)

        inp_enc_level3_ir = self.down2_3(out_enc_level2_ir)
        out_enc_level3_ir = self.encoder_level3_ir(inp_enc_level3_ir)

        inp_enc_level4_ir = self.down3_4(out_enc_level3_ir)
        latent_ir = self.latent_ir(inp_enc_level4_ir)

        out_enc_level1_vi = self.encoder_level1_vi(inp_enc_level1_vi)

        inp_enc_level2_vi = self.down1_2(out_enc_level1_vi)
        out_enc_level2_vi = self.encoder_level2_vi(inp_enc_level2_vi)

        inp_enc_level3_vi = self.down2_3(out_enc_level2_vi)
        out_enc_level3_vi = self.encoder_level3_vi(inp_enc_level3_vi)

        inp_enc_level4_vi = self.down3_4(out_enc_level3_vi)
        latent_vi = self.latent_vi(inp_enc_level4_vi)

        latent, _ = self.latent_shuffle_channel(
            latent_vi, latent_ir, text_code
        )

        inp_dec_level3 = self.up4_3(latent)
        outt1, _ = self.encoder_shuffle_channel3(
            out_enc_level3_vi, out_enc_level3_ir, text_code
        )
        inp_dec_level3 = torch.cat([inp_dec_level3, outt1], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        outt2, _ = self.encoder_shuffle_channel2(
            out_enc_level2_vi, out_enc_level2_ir, text_code
        )
        inp_dec_level2 = torch.cat([inp_dec_level2, outt2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        outt3, _ = self.encoder_shuffle_channel1(
            out_enc_level1_vi, out_enc_level1_ir, text_code
        )
        inp_dec_level1 = torch.cat([inp_dec_level1, outt3], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1





