
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
from thop import profile
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# bchw->blc->blc->bchw
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# 先验提取模块
class PriorExtractionModule(nn.Module):
    def __init__(self, in_channels=3,dim=24):
        super(PriorExtractionModule, self).__init__()

        self.light_extraction = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=dim, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU()
        )

        self.rain_extraction = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=dim, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU()
        )

        self.over_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU()
        )

        self.under_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU()
        )


    def forward(self, x):
        max_c, _ = x.max(dim=1, keepdim=True)
        min_c, _ = x.min(dim=1, keepdim=True)

        illumination = self.light_extraction(x)

        mean_channel = illumination.mean(dim=1, keepdim=True)
        light_prior = torch.sigmoid(mean_channel)

        reflectance = x/(illumination+1e-7)

        overlight_feature = torch.cat([illumination,max_c],dim=1)

        underlight_feature = torch.cat([illumination,min_c],dim=1)

        overlight_map = self.over_conv(overlight_feature)

        mean_channel = overlight_map.mean(dim=1, keepdim=True)
        overlight_map = torch.sigmoid(mean_channel)

        underlight_map = self.under_conv(underlight_feature)
        mean_channel = underlight_map.mean(dim=1, keepdim=True)
        underlight_map = torch.sigmoid(mean_channel)

        rain_prior = self.rain_extraction(reflectance)

        return rain_prior,light_prior,overlight_map,underlight_map

# 加入雨先验
class RDGM(nn.Module):
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.):
        super(RDGM, self).__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.beta_conv = nn.Conv2d(in_channels=3, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.gamma_conv = nn.Conv2d(in_channels=3, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.proj = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.conv4 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,groups=dw_channel // 2)

        self.conv5 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,groups=dw_channel // 2)

        self.conv6 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,groups=dw_channel // 2)

        self.norm1 = LayerNorm(c, 'WithBias')

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp, rain_prior):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        # 雨先验的融合操作
        beta = self.beta_conv(rain_prior)
        gamma = self.gamma_conv(rain_prior)

        x = x*beta + gamma

        # dw_channel->dw_channel
        x = self.proj(x)

        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv4(x1)
        x2 = self.conv5(x2)
        x = x1*x2

        x = self.conv6(x) * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        return x

# 加入光先验
class LVSSM(nn.Module):
    def __init__(
            self,
            # 通道维度
            d_model,
            # 隐藏状态数量
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 3, kernel_size=1,padding=0,bias=bias)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm(self.d_inner, 'WithBias')
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1,padding=0,bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.alpha = nn.Parameter(torch.rand(1), requires_grad=True)

        self.light_proj = nn.Conv2d(in_channels=1, out_channels=self.d_inner, kernel_size=1, padding=0, stride=1)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor,light_prior):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x = x+light_prior

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward(self, x: torch.Tensor,light_prior,overlight_map,underlight_map, **kwargs):
        B, C ,H, W = x.shape

        xzz = self.in_proj(x)
        x, z1,z2 = xzz.chunk(3, dim=1)

        light_prior = self.light_proj(light_prior)
        overlight_map = self.light_proj(overlight_map)
        underlight_map = self.light_proj(underlight_map)

        x = self.act(self.conv2d1(x))
        y1, y2, y3, y4 = self.forward_core(x,light_prior)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = y.view(B, -1, H, W)
        y = self.out_norm(y)

        z1 = z1+overlight_map
        z1 = y * F.silu(z1)

        z2 = z2+underlight_map
        z2 = y * F.silu(z2)

        y = self.alpha*z1+(1-self.alpha)*z2

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = LVSSM(d_model=dim)
        self.ffn = RDGM(c=dim)

    def forward(self, x,rain_prior,light_prior,overlight_map,underlight_map):
        inc = x
        x = self.norm1(x)
        x = inc + self.attn(x,light_prior,overlight_map,underlight_map)
        x = self.ffn(x,rain_prior)+x

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample_with_same_channel(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_with_same_channel, self).__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(n_feat*4, n_feat, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
class NDMamba(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 24,
        num_blocks = [2,3,3,4],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 1.667,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(NDMamba, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4

        self.latent = nn.ModuleList([TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.ModuleList([TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.pem1 = PriorExtractionModule(dim=16)
        self.pem2 = PriorExtractionModule(dim=24)
        self.pem3 = PriorExtractionModule(dim=32)
        self.pem4 = PriorExtractionModule(dim=48)


    def forward(self, inp_img):

        img1 = inp_img
        img2 = F.interpolate(img1,scale_factor=0.5)
        img3 = F.interpolate(img2,scale_factor=0.5)
        img4 = F.interpolate(img3,scale_factor=0.5)


        rain_prior_level1, light_prior_level1, overlight_map_level1, underlight_map_level1 = self.pem1(img1)
        rain_prior_level2, light_prior_level2, overlight_map_level2, underlight_map_level2 = self.pem2(img2)
        rain_prior_level3, light_prior_level3, overlight_map_level3, underlight_map_level3 = self.pem3(img3)
        rain_prior_level4, light_prior_level4, overlight_map_level4, underlight_map_level4 = self.pem4(img4)


        inp_enc_level1 = self.patch_embed(inp_img)
        x = inp_enc_level1
        for block in self.encoder_level1:
            x = block(x, rain_prior_level1, light_prior_level1, overlight_map_level1, underlight_map_level1)
        out_enc_level1 = x

        inp_enc_level2 = self.down1_2(out_enc_level1)

        x = inp_enc_level2
        for block in self.encoder_level2:
            x = block(x, rain_prior_level2, light_prior_level2, overlight_map_level2, underlight_map_level2)
        out_enc_level2 = x

        inp_enc_level3 = self.down2_3(out_enc_level2)

        x = inp_enc_level3
        for block in self.encoder_level3:
            x = block(x, rain_prior_level3, light_prior_level3, overlight_map_level3, underlight_map_level3)
        out_enc_level3 = x

        inp_enc_level4 = self.down3_4(out_enc_level3)

        x = inp_enc_level4
        for block in self.latent:
            x = block(x, rain_prior_level4, light_prior_level4, overlight_map_level4, underlight_map_level4)
        latent = x

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        x = inp_dec_level3
        for block in self.decoder_level3:
            x = block(x, rain_prior_level3, light_prior_level3, overlight_map_level3, underlight_map_level3)
        out_dec_level3 = x

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        x = inp_dec_level2
        for block in self.decoder_level2:
            x = block(x, rain_prior_level2, light_prior_level2, overlight_map_level2, underlight_map_level2)
        out_dec_level2 = x

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        x = inp_dec_level1
        for block in self.decoder_level1:
            x = block(x, rain_prior_level1, light_prior_level1, overlight_map_level1, underlight_map_level1)
        out_dec_level1 = x

        x = out_dec_level1
        for block in self.refinement:
            x = block(x, rain_prior_level1, light_prior_level1, overlight_map_level1, underlight_map_level1)
        out_dec_level1 = x

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

if __name__ == '__main__':

    x = torch.randn(1,3,256,256).cuda()
    model = NDMamba(dim=24, num_blocks=[2,3,3,4], num_refinement_blocks=4).cuda()
    flops, params = profile(model, inputs=(x,))
    print(flops, params)
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        y = model(x)
    print(y.shape)