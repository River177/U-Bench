"""
VSSUTANet: 融合 UTANet 与 VSSBlock 的医学图像分割模型

创新点：
1. 使用 ResNet34 作为编码器（来自 UTANet）
2. 保留 TA-MoSC 自适应跳跃连接机制（来自 UTANet）
3. 在解码器中引入 VSSBlock 进行长程依赖建模（来自 MambaUnet）
4. 结合 CNN 局部特征提取 + 状态空间模型全局建模的优势

架构示意：
    输入图像
        │
    ResNet34 编码器（多尺度特征 e1-e5）
        │
    TA-MoSC 模块（自适应跳跃连接路由）
        │
    VSS 解码器（VSSBlock 增强的上采样路径）
        │
    分割输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, Callable
from functools import partial
import math

from timm.models.layers import DropPath, trunc_normal_
from .ta_mosc import MoE  # TA-MoSC 模块

# ==================== 从 mamba_sys.py 引入的核心组件 ====================

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    selective_scan_fn = None

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    selective_scan_fn_v1 = None

from einops import rearrange, repeat


class SS2D(nn.Module):
    """
    二维选择性状态空间模块（Selective State Space 2D）
    
    核心思想：
    1. 将 2D 特征图展平成 4 个方向的序列（行正/反、列正/反）
    2. 对每个方向进行选择性扫描（类似 RNN 但更高效）
    3. 融合 4 个方向的输出，实现全局建模
    
    相比自注意力的优势：
    - 计算复杂度 O(N) vs O(N²)
    - 更适合处理长序列/高分辨率特征图
    """
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

        # 输入投影：将 d_model 映射到 2*d_inner（主分支 + 门控分支）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 深度可分离卷积：提取局部特征
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

        # 4 个扫描方向的投影权重
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # 时间步投影
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # 状态空间参数 A 和 D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # 输出层
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
        else:
            raise NotImplementedError
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
            "n -> d n",
            d=d_inner,
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
        """选择性扫描的核心前向过程"""
        if selective_scan_fn is None and selective_scan_fn_v1 is None:
            raise ImportError("请安装 mamba_ssm 或 selective_scan 库")
        
        self.selective_scan = selective_scan_fn if selective_scan_fn is not None else selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4  # 4 个扫描方向

        # 构建 4 个方向的序列：行正、列正、行反、列反
        x_hwwh = torch.stack([
            x.view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        # 投影得到 dt, B, C 参数
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

        # 选择性扫描
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # 融合 4 个方向的输出
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        
        # 双线性投影：主分支 x 和门控分支 z
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 深度卷积 + 选择性扫描
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        
        # 门控调制
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """
    VSS Block：基于状态空间的注意力块
    
    结构：LayerNorm -> SS2D -> DropPath -> 残差连接
    
    作用：在不改变张量形状的情况下，增强特征的全局建模能力
    """
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
        # 输入形状: (B, H, W, C)
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# ==================== 融合后的新模型组件 ====================

class VSSUpBlock(nn.Module):
    """
    VSS 增强的上采样块
    
    结构：转置卷积上采样 -> 跳跃连接融合 -> VSSBlock 全局建模 -> 卷积细化
    
    相比 UTANet 原始 UpBlock 的改进：
    - 在特征融合后加入 VSSBlock 进行长程依赖建模
    - 增强解码器的全局上下文理解能力
    """
    def __init__(
        self, 
        in_ch: int, 
        skip_ch: int, 
        out_ch: int,
        use_vss: bool = True,
        d_state: int = 16,
        drop_path: float = 0.0,
    ):
        """
        Args:
            in_ch: 解码器路径的输入通道数
            skip_ch: 编码器跳跃连接的通道数
            out_ch: 输出通道数
            use_vss: 是否使用 VSSBlock 进行特征增强
            d_state: 状态空间维度
            drop_path: DropPath 概率
        """
        super().__init__()
        self.use_vss = use_vss
        
        # 转置卷积上采样
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True)
        )
        
        # 跳跃连接融合后的通道数
        fused_ch = in_ch//2 + skip_ch
        
        # VSSBlock 需要 (B, H, W, C) 格式，先用 1x1 卷积调整通道
        if use_vss:
            self.pre_vss = nn.Sequential(
                nn.Conv2d(fused_ch, out_ch, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.vss_block = VSSBlock(
                hidden_dim=out_ch,
                drop_path=drop_path,
                d_state=d_state,
            )
            self.post_vss = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            # 不使用 VSS 时退化为普通卷积
            self.conv = nn.Sequential(
                nn.Conv2d(fused_ch, out_ch, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_feat: 解码器特征 (B, in_ch, H, W)
            skip_feat: 编码器跳跃连接特征 (B, skip_ch, 2H, 2W)
        
        Returns:
            融合后的特征 (B, out_ch, 2H, 2W)
        """
        # 上采样
        up_feat = self.up(decoder_feat)
        
        # 跳跃连接融合
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        
        if self.use_vss:
            # 通道调整
            x = self.pre_vss(fused_feat)
            
            # VSSBlock 处理（需要 BHWC 格式）
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            x = self.vss_block(x)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            
            # 后处理卷积
            out = self.post_vss(x)
        else:
            out = self.conv(fused_feat)
        
        return out


class VSSUTANet(nn.Module):
    """
    VSSUTANet: 融合 UTANet 与 VSSBlock 的医学图像分割模型
    
    架构特点：
    1. 编码器：ResNet34 预训练模型，提取多尺度特征
    2. 跳跃连接：TA-MoSC 模块，自适应特征路由
    3. 解码器：VSSUpBlock，结合 CNN 和状态空间模型
    
    训练策略（同 UTANet）：
    - 阶段 1 (pretrained=False)：训练基础 UNet 结构
    - 阶段 2 (pretrained=True)：训练 TA-MoSC 模块
    """
    def __init__(
        self, 
        pretrained: bool = True, 
        topk: int = 2, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        img_size: int = 224,
        use_vss: bool = True,
        d_state: int = 16,
        drop_path_rate: float = 0.1,
    ):
        """
        初始化 VSSUTANet 模型
        
        Args:
            pretrained: 训练阶段标志
                False (阶段1): 训练基础 UNet
                True  (阶段2): 训练 TA-MoSC 模块
            topk: MoE 模块中选择的专家数量
            n_channels: 输入图像通道数
            n_classes: 输出类别数
            img_size: 输入图像尺寸
            use_vss: 是否在解码器中使用 VSSBlock
            d_state: 状态空间维度
            drop_path_rate: DropPath 最大概率
        """
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size
        self.use_vss = use_vss

        # ========== 编码器：ResNet34 ==========
        self.resnet = models.resnet34(pretrained=True)
        self.filters_resnet = [64, 64, 128, 256, 512]
        self.filters_decoder = [32, 64, 128, 256, 512]

        # 自定义第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        # ResNet 各层
        self.conv2 = self.resnet.layer1  # (B, 64, H/2, W/2)
        self.conv3 = self.resnet.layer2  # (B, 128, H/4, W/4)
        self.conv4 = self.resnet.layer3  # (B, 256, H/8, W/8)
        self.conv5 = self.resnet.layer4  # (B, 512, H/16, W/16)

        # ========== TA-MoSC 模块 ==========
        if pretrained:
            self.fuse = nn.Sequential(
                nn.Conv2d(512, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            self.docker1 = self._create_docker(64, self.filters_resnet[0])
            self.docker2 = self._create_docker(64, self.filters_resnet[1])
            self.docker3 = self._create_docker(64, self.filters_resnet[2])
            self.docker4 = self._create_docker(64, self.filters_resnet[3])

        # ========== VSS 增强解码器 ==========
        # DropPath 随深度递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        
        self.up5 = VSSUpBlock(
            self.filters_resnet[4], self.filters_resnet[3], self.filters_decoder[3],
            use_vss=use_vss, d_state=d_state, drop_path=dpr[3]
        )
        self.up4 = VSSUpBlock(
            self.filters_decoder[3], self.filters_resnet[2], self.filters_decoder[2],
            use_vss=use_vss, d_state=d_state, drop_path=dpr[2]
        )
        self.up3 = VSSUpBlock(
            self.filters_decoder[2], self.filters_resnet[1], self.filters_decoder[1],
            use_vss=use_vss, d_state=d_state, drop_path=dpr[1]
        )
        self.up2 = VSSUpBlock(
            self.filters_decoder[1], self.filters_resnet[0], self.filters_decoder[0],
            use_vss=use_vss, d_state=d_state, drop_path=dpr[0]
        )

        # ========== 预测头 ==========
        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0]//2, 1),
            nn.BatchNorm2d(self.filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0]//2, n_classes, 1)
        )
        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

        # 初始化权重
        self._init_weights()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """创建特征路由器"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        """初始化新增模块的权重"""
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
        
        Args:
            x: 输入图像 (B, n_channels, H, W)
        
        Returns:
            logits: 分割输出 (B, n_classes, H, W)
            aux_loss: MoE 辅助损失（负载均衡）
        """
        # ========== 编码器 ==========
        e1 = self.conv1(x)           # (B, 64, H, W)
        e1_maxp = self.maxpool(e1)   # (B, 64, H/2, W/2)
        e2 = self.conv2(e1_maxp)     # (B, 64, H/2, W/2)
        e3 = self.conv3(e2)          # (B, 128, H/4, W/4)
        e4 = self.conv4(e3)          # (B, 256, H/8, W/8)
        e5 = self.conv5(e4)          # (B, 512, H/16, W/16)

        aux_loss = torch.tensor(0.0, device=x.device)

        # ========== TA-MoSC 模块 ==========
        if self.pretrained:
            # 多尺度特征融合到统一尺寸
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear')
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear')
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear')
            
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)
            
            # MoE 路由
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss

            # 特征分发
            o1 = self.docker1(o1)
            o2 = self.docker2(o2)
            o3 = self.docker3(o3)
            o4 = self.docker4(o4)
            
            # 调整空间尺寸
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear')
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear')
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear')
        else:
            o1, o2, o3, o4 = e1, e2, e3, e4

        # ========== VSS 解码器 ==========
        d4 = self.up5(e5, o4)  # (B, 256, H/8, W/8)
        d3 = self.up4(d4, o3)  # (B, 128, H/4, W/4)
        d2 = self.up3(d3, o2)  # (B, 64, H/2, W/2)
        d1 = self.up2(d2, o1)  # (B, 32, H, W)

        # ========== 预测输出 ==========
        logits = self.pred(d1)
        
        # 目前训练框架只接收单一输出张量，这里仅返回 logits。
        # 如需使用 aux_loss，可在训练脚本中显式加权组合。
        return logits


def vssutanet(input_channel=3, num_classes=1, use_vss=True):
    """
    便捷工厂函数
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        use_vss: 是否使用 VSSBlock
    
    Returns:
        VSSUTANet 模型实例
    """
    return VSSUTANet(
        n_channels=input_channel, 
        n_classes=num_classes,
        use_vss=use_vss
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("VSSUTANet 模型测试")
    print("=" * 60)
    
    # 创建模型
    model = VSSUTANet(
        pretrained=True,
        n_channels=3,
        n_classes=1,
        use_vss=True,
        img_size=224
    )
    
    # 打印模型结构摘要
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    print(f"\n前向传播测试:")
    input_tensor = torch.randn(2, 3, 224, 224)
    print(f"  输入形状: {input_tensor.shape}")
    
    model.eval()
    with torch.no_grad():
        output, aux_loss = model(input_tensor)
    
    print(f"  输出形状: {output.shape}")
    print(f"  辅助损失: {aux_loss.item():.6f}")
    
    # 对比不使用 VSS 的版本
    print(f"\n对比测试（不使用 VSSBlock）:")
    model_no_vss = VSSUTANet(
        pretrained=True,
        n_channels=3,
        n_classes=1,
        use_vss=False
    )
    params_no_vss = sum(p.numel() for p in model_no_vss.parameters())
    print(f"  无 VSS 参数量: {params_no_vss / 1e6:.2f}M")
    print(f"  VSS 额外参数: {(total_params - params_no_vss) / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

