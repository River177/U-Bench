"""
DualBranchVSSNet: CNN + VSS 双分支交叉融合的医学图像分割模型

核心创新：
1. CNN 分支（ResNet34）：擅长提取局部纹理、边缘等低级特征
2. VSS 分支（VSSBlock）：擅长建模长程依赖、全局上下文
3. 跨分支融合模块（CrossFusion）：让两个分支互相增强
4. TA-MoSC 模块：自适应选择融合后的多尺度特征

设计灵感：
- TransFuse: CNN + Transformer 双分支
- CMT: CNN 和 Transformer 交叉
- 本模型用 VSS（状态空间）替代 Transformer，复杂度更低

架构示意：
┌─────────────────────────────────────────────────────────┐
│                        输入图像                          │
│                           │                             │
│         ┌─────────────────┴─────────────────┐           │
│         ▼                                   ▼           │
│    ┌─────────┐                         ┌─────────┐      │
│    │CNN 分支 │◄────── CrossFusion ──────►│VSS 分支│      │
│    │ ResNet  │          × 4 层          │VSSBlock│      │
│    └─────────┘                         └─────────┘      │
│         │                                   │           │
│         └─────────────┬─────────────────────┘           │
│                       ▼                                 │
│                 融合多尺度特征                            │
│                       │                                 │
│                       ▼                                 │
│                  TA-MoSC 路由                           │
│                       │                                 │
│                       ▼                                 │
│                    解码器                               │
│                       │                                 │
│                       ▼                                 │
│                   分割输出                              │
└─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, Callable, List
from functools import partial
import math

from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from .ta_mosc import MoE

# ==================== 从 mamba_sys.py 引入的核心组件 ====================

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    selective_scan_fn = None

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    selective_scan_fn_v1 = None


class SS2D(nn.Module):
    """二维选择性状态空间模块"""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
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

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
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
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
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
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        if selective_scan_fn is None and selective_scan_fn_v1 is None:
            raise ImportError("请安装 mamba_ssm 或 selective_scan 库")
        
        self.selective_scan = selective_scan_fn if selective_scan_fn is not None else selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([
            x.view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
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
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """VSS Block：LayerNorm -> SS2D -> DropPath -> 残差"""
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # 输入: (B, H, W, C)
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# ==================== 双分支融合核心组件 ====================

class PatchEmbed(nn.Module):
    """
    将图像切分成 patch 并嵌入到指定维度
    用于 VSS 分支的输入预处理
    """
    def __init__(self, in_chans=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H/patch, W/patch, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', embed_dim)
        x = self.norm(x)
        return x


class VSSStage(nn.Module):
    """
    VSS 分支的一个阶段
    包含多个 VSSBlock + 可选的下采样
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        drop_path: List[float] = [0.0],
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if i < len(drop_path) else drop_path[-1],
                d_state=d_state,
            )
            for i in range(depth)
        ])
        
        # 下采样：空间减半，通道翻倍
        if downsample:
            self.downsample = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
            )
            self.pool = nn.AvgPool2d(2, 2)
            self.out_dim = dim * 2
        else:
            self.downsample = None
            self.out_dim = dim
    
    def forward(self, x):
        # x: (B, H, W, C)
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            B, H, W, C = x.shape
            x = self.downsample(x)  # (B, H, W, 2C)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, 2C, H, W)
            x = self.pool(x)  # (B, 2C, H/2, W/2)
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H/2, W/2, 2C)
        
        return x


class CrossFusionModule(nn.Module):
    """
    跨分支融合模块
    
    CNN 和 VSS 分支的特征互相增强：
    1. CNN 特征经过通道注意力后增强 VSS 特征
    2. VSS 特征经过空间注意力后增强 CNN 特征
    3. 两种特征相加融合
    
    设计思想：
    - CNN 擅长局部特征 → 用通道注意力筛选重要通道
    - VSS 擅长全局特征 → 用空间注意力筛选重要位置
    """
    def __init__(self, cnn_dim: int, vss_dim: int, out_dim: int):
        """
        Args:
            cnn_dim: CNN 分支的通道数
            vss_dim: VSS 分支的通道数
            out_dim: 输出通道数
        """
        super().__init__()
        
        # 通道对齐
        self.cnn_align = nn.Conv2d(cnn_dim, out_dim, 1) if cnn_dim != out_dim else nn.Identity()
        self.vss_align = nn.Conv2d(vss_dim, out_dim, 1) if vss_dim != out_dim else nn.Identity()
        
        # CNN → VSS 的通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, 1),
            nn.Sigmoid()
        )
        
        # VSS → CNN 的空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_dim, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 融合后的细化
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, cnn_feat: torch.Tensor, vss_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_feat: CNN 特征 (B, C_cnn, H, W)
            vss_feat: VSS 特征 (B, H, W, C_vss) 注意是 BHWC 格式！
        
        Returns:
            cnn_enhanced: 增强后的 CNN 特征 (B, out_dim, H, W)
            vss_enhanced: 增强后的 VSS 特征 (B, H, W, out_dim)
        """
        # VSS 转换为 BCHW 格式
        vss_feat_bchw = vss_feat.permute(0, 3, 1, 2).contiguous()
        
        # 尺寸对齐（如果不一致）
        if cnn_feat.shape[2:] != vss_feat_bchw.shape[2:]:
            vss_feat_bchw = F.interpolate(vss_feat_bchw, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 通道对齐
        cnn_aligned = self.cnn_align(cnn_feat)
        vss_aligned = self.vss_align(vss_feat_bchw)
        
        # CNN → VSS: 通道注意力增强
        cnn_channel_weight = self.channel_attn(cnn_aligned)
        vss_enhanced_bchw = vss_aligned * cnn_channel_weight + vss_aligned
        
        # VSS → CNN: 空间注意力增强
        vss_spatial_weight = self.spatial_attn(vss_aligned)
        cnn_enhanced = cnn_aligned * vss_spatial_weight + cnn_aligned
        
        # 特征融合
        fused = torch.cat([cnn_enhanced, vss_enhanced_bchw], dim=1)
        fused = self.refine(fused)
        
        # 输出：CNN 格式 (BCHW) 和 VSS 格式 (BHWC)
        vss_enhanced = vss_enhanced_bchw.permute(0, 2, 3, 1).contiguous()
        
        return fused, vss_enhanced


class UpBlock(nn.Module):
    """解码器上采样块"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2 + skip_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==================== 主模型 ====================

class DualBranchVSSNet(nn.Module):
    """
    DualBranchVSSNet: CNN + VSS 双分支交叉融合网络
    
    架构：
    1. CNN 分支：ResNet34，提取局部特征
    2. VSS 分支：VSSBlock 堆叠，提取全局特征
    3. CrossFusion：每个尺度上两分支交叉融合
    4. TA-MoSC：自适应跳跃连接路由
    5. 解码器：逐层上采样恢复分辨率
    """
    def __init__(
        self,
        pretrained: bool = True,
        topk: int = 2,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        vss_dims: List[int] = [64, 128, 256, 512],
        vss_depths: List[int] = [2, 2, 2, 2],
        d_state: int = 16,
        drop_path_rate: float = 0.1,
    ):
        """
        Args:
            pretrained: 是否使用 TA-MoSC
            topk: MoE 专家选择数
            n_channels: 输入通道数
            n_classes: 输出类别数
            img_size: 输入尺寸
            vss_dims: VSS 分支各阶段通道数
            vss_depths: VSS 分支各阶段 VSSBlock 数量
            d_state: 状态空间维度
            drop_path_rate: DropPath 最大概率
        """
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        
        # ==================== CNN 分支（ResNet34）====================
        self.resnet = models.resnet34(pretrained=True)
        self.cnn_dims = [64, 64, 128, 256, 512]
        
        # 第一层
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.cnn_pool = nn.MaxPool2d(2, 2)
        
        # ResNet 各层
        self.cnn_layer1 = self.resnet.layer1  # 64 -> 64
        self.cnn_layer2 = self.resnet.layer2  # 64 -> 128
        self.cnn_layer3 = self.resnet.layer3  # 128 -> 256
        self.cnn_layer4 = self.resnet.layer4  # 256 -> 512
        
        # ==================== VSS 分支 ====================
        # Patch Embedding：将输入图像转为 patch 序列
        self.vss_patch_embed = PatchEmbed(
            in_chans=n_channels,
            embed_dim=vss_dims[0],
            patch_size=4  # 对应 CNN stem + pool 后的尺寸
        )
        
        # DropPath 递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(vss_depths))]
        
        # VSS 各阶段
        self.vss_stages = nn.ModuleList()
        for i in range(4):
            stage = VSSStage(
                dim=vss_dims[i],
                depth=vss_depths[i],
                d_state=d_state,
                drop_path=dpr[sum(vss_depths[:i]):sum(vss_depths[:i+1])],
                downsample=(i < 3),  # 最后一层不下采样
            )
            self.vss_stages.append(stage)
        
        # ==================== 跨分支融合模块 ====================
        # 融合输出的通道数（统一使用 CNN 的通道数）
        self.fusion_dims = [64, 128, 256, 512]
        
        self.cross_fusions = nn.ModuleList([
            CrossFusionModule(self.cnn_dims[i+1], vss_dims[i], self.fusion_dims[i])
            for i in range(4)
        ])
        
        # ==================== TA-MoSC 模块 ====================
        if pretrained:
            # 融合：64 + 128 + 256 + 512 = 960 -> 64
            total_ch = sum(self.fusion_dims)
            self.fuse = nn.Sequential(
                nn.Conv2d(total_ch, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            self.docker1 = self._create_docker(64, self.fusion_dims[0])
            self.docker2 = self._create_docker(64, self.fusion_dims[1])
            self.docker3 = self._create_docker(64, self.fusion_dims[2])
            self.docker4 = self._create_docker(64, self.fusion_dims[3])
        
        # ==================== 解码器 ====================
        self.decoder_dims = [32, 64, 128, 256]
        
        self.up4 = UpBlock(self.fusion_dims[3], self.fusion_dims[2], self.decoder_dims[3])
        self.up3 = UpBlock(self.decoder_dims[3], self.fusion_dims[1], self.decoder_dims[2])
        self.up2 = UpBlock(self.decoder_dims[2], self.fusion_dims[0], self.decoder_dims[1])
        self.up1 = UpBlock(self.decoder_dims[1], 64, self.decoder_dims[0])  # 与 stem 融合
        
        # ==================== 预测头 ====================
        self.pred = nn.Sequential(
            nn.Conv2d(self.decoder_dims[0], self.decoder_dims[0]//2, 1),
            nn.BatchNorm2d(self.decoder_dims[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_dims[0]//2, n_classes, 1)
        )
        
        self._init_weights()
    
    def _create_docker(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        数据流：
        1. CNN 分支和 VSS 分支并行处理
        2. 每个尺度上进行 CrossFusion
        3. 融合特征送入 TA-MoSC 进行路由
        4. 解码器逐层上采样
        
        Args:
            x: 输入图像 (B, C, H, W)
        
        Returns:
            logits: 分割输出 (B, n_classes, H, W)
            aux_loss: MoE 辅助损失
        """
        B = x.shape[0]
        
        # ==================== 双分支编码 ====================
        # CNN 分支
        c0 = self.cnn_stem(x)          # (B, 64, H, W) - 用于最后 skip
        c0_pool = self.cnn_pool(c0)    # (B, 64, H/2, W/2)
        c1 = self.cnn_layer1(c0_pool)  # (B, 64, H/2, W/2)
        c2 = self.cnn_layer2(c1)       # (B, 128, H/4, W/4)
        c3 = self.cnn_layer3(c2)       # (B, 256, H/8, W/8)
        c4 = self.cnn_layer4(c3)       # (B, 512, H/16, W/16)
        
        # VSS 分支
        v0 = self.vss_patch_embed(x)   # (B, H/4, W/4, 64)
        
        # 阶段 1 融合
        v1 = self.vss_stages[0](v0)    # (B, H/8, W/8, 128)
        # 调整 c1 尺寸以匹配 v0
        c1_for_fusion = F.interpolate(c1, size=(v0.shape[1], v0.shape[2]), mode='bilinear', align_corners=False)
        f1, v1_enhanced = self.cross_fusions[0](c1_for_fusion, v0)
        
        # 阶段 2 融合
        v2 = self.vss_stages[1](v1)    # (B, H/16, W/16, 256)
        c2_for_fusion = F.interpolate(c2, size=(v1.shape[1], v1.shape[2]), mode='bilinear', align_corners=False)
        f2, v2_enhanced = self.cross_fusions[1](c2_for_fusion, v1)
        
        # 阶段 3 融合
        v3 = self.vss_stages[2](v2)    # (B, H/32, W/32, 512)
        c3_for_fusion = F.interpolate(c3, size=(v2.shape[1], v2.shape[2]), mode='bilinear', align_corners=False)
        f3, v3_enhanced = self.cross_fusions[2](c3_for_fusion, v2)
        
        # 阶段 4 融合（最深层）
        v4 = self.vss_stages[3](v3)    # (B, H/32, W/32, 512)
        c4_for_fusion = F.interpolate(c4, size=(v3.shape[1], v3.shape[2]), mode='bilinear', align_corners=False)
        f4, v4_enhanced = self.cross_fusions[3](c4_for_fusion, v3)
        
        # 融合特征列表（从浅到深）
        fused_features = [f1, f2, f3, f4]
        
        # ==================== TA-MoSC 路由 ====================
        aux_loss = torch.tensor(0.0, device=x.device)
        
        if self.pretrained:
            # 调整到统一尺寸（使用 f2 的尺寸，即 H/4）
            target_size = f2.shape[2:]
            
            f1_resized = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
            f3_resized = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
            f4_resized = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
            
            # 拼接并融合
            fused_all = torch.cat([f1_resized, f2, f3_resized, f4_resized], dim=1)
            fused_all = self.fuse(fused_all)
            
            # MoE 路由
            o1, o2, o3, o4, loss = self.moe(fused_all)
            aux_loss = loss
            
            # Docker 分发
            o1 = self.docker1(o1)
            o2 = self.docker2(o2)
            o3 = self.docker3(o3)
            o4 = self.docker4(o4)
            
            # 调整到各自尺度
            o1 = F.interpolate(o1, size=f1.shape[2:], mode='bilinear', align_corners=False)
            o3 = F.interpolate(o3, size=f3.shape[2:], mode='bilinear', align_corners=False)
            o4 = F.interpolate(o4, size=f4.shape[2:], mode='bilinear', align_corners=False)
            
            skip_features = [o1, o2, o3, o4]
        else:
            skip_features = fused_features
        
        # ==================== 解码器 ====================
        # 保存原始输入尺寸，用于最终上采样
        input_size = x.shape[2:]  # (H, W)
        
        # 调整 c4 尺寸作为解码器起点
        bottleneck = F.interpolate(c4, size=skip_features[3].shape[2:], mode='bilinear', align_corners=False)
        
        d4 = self.up4(bottleneck, skip_features[2])  # (B, 256, H/8, W/8)
        d3 = self.up3(d4, skip_features[1])          # (B, 128, H/4, W/4)
        d2 = self.up2(d3, skip_features[0])          # (B, 64, H/2, W/2)
        # 最后一级与 c0_pool 对齐
        d1 = self.up1(d2, c0_pool)                   # (B, 32, H/2, W/2)
        
        # ==================== 预测 ====================
        logits = self.pred(d1)
        
        # 上采样到原始输入分辨率
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        # 当前训练框架只接收单一输出张量，这里仅返回 logits；
        # 如需显式使用 aux_loss，请在训练脚本中对 DualBranchVSSNet 特判并加权组合。
        return logits


def dualbranchvssnet(input_channel=3, num_classes=1):
    """便捷工厂函数"""
    return DualBranchVSSNet(
        n_channels=input_channel,
        n_classes=num_classes,
        pretrained=True
    )


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("DualBranchVSSNet: CNN + VSS 双分支交叉融合网络 测试")
    print("=" * 70)
    
    # 创建模型
    model = DualBranchVSSNet(
        pretrained=True,
        n_channels=3,
        n_classes=1,
        img_size=224
    )
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 模型参数统计:")
    print(f"   总参数量: {total_params / 1e6:.2f}M")
    print(f"   可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 分支参数统计
    cnn_params = sum(p.numel() for n, p in model.named_parameters() if 'cnn' in n or 'resnet' in n)
    vss_params = sum(p.numel() for n, p in model.named_parameters() if 'vss' in n)
    fusion_params = sum(p.numel() for n, p in model.named_parameters() if 'cross' in n or 'fusion' in n)
    
    print(f"\n🌳 分支参数分布:")
    print(f"   CNN 分支: {cnn_params / 1e6:.2f}M")
    print(f"   VSS 分支: {vss_params / 1e6:.2f}M")
    print(f"   融合模块: {fusion_params / 1e6:.2f}M")
    
    # 前向传播测试
    print(f"\n🚀 前向传播测试:")
    input_tensor = torch.randn(2, 3, 224, 224)
    print(f"   输入形状: {input_tensor.shape}")
    
    model.eval()
    with torch.no_grad():
        output, aux_loss = model(input_tensor)
    
    print(f"   输出形状: {output.shape}")
    print(f"   辅助损失: {aux_loss.item():.6f}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)

