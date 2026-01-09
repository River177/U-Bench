"""
UTANet++ Micro: 轻量化辅助模块
包含深度可分离卷积、轴向卷积、轻量化解码器和注意力模块

借鉴设计：
- MALUNet: 深度可分离卷积 + 门控注意力
- ULite: 轴向深度卷积
- CMUNeXt: 残差深度卷积块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：将标准卷积分解为深度卷积+逐点卷积
    参数量减少约 k²/C_out 倍（k为卷积核大小）
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False
    ):
        super().__init__()
        # 深度卷积：每个输入通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 关键：groups=in_channels
            bias=False
        )
        # 逐点卷积：1x1卷积进行通道融合
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class AxialDepthwiseConv(nn.Module):
    """
    轴向深度卷积：将2D卷积分解为两个1D卷积（垂直+水平）
    参数量从 k² 减少到 2k
    
    Reference: ULite
    """
    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        # 垂直方向卷积
        self.conv_h = nn.Conv2d(
            dim, dim, 
            kernel_size=(kernel_size, 1), 
            padding=(padding, 0),
            groups=dim, 
            dilation=dilation
        )
        # 水平方向卷积
        self.conv_w = nn.Conv2d(
            dim, dim, 
            kernel_size=(1, kernel_size), 
            padding=(0, padding),
            groups=dim, 
            dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_h(x) + self.conv_w(x)


class LightweightBlock(nn.Module):
    """
    轻量化编码器块
    结构：AxialDW + DepthwiseSeparable + Residual
    
    Reference: CMUNeXt + ULite
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 7,
        downsample: bool = True
    ):
        super().__init__()
        self.downsample = downsample
        
        # 轴向深度卷积
        self.axial_dw = AxialDepthwiseConv(in_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 深度可分离卷积进行通道转换
        self.dw_conv = DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1)
        
        # 下采样
        if downsample:
            self.pool = nn.MaxPool2d(2, 2)
        
        # 残差连接（如果通道数改变）
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 轴向卷积
        skip = self.bn1(self.axial_dw(x))
        
        # 深度可分离卷积
        out = self.dw_conv(skip)
        
        # 残差连接
        out = out + self.residual(skip)
        
        # 下采样
        if self.downsample:
            out = self.pool(out)
        
        return out, skip


class LightweightDecoder(nn.Module):
    """
    轻量化解码器块
    结构：Upsample + Concat + DepthwiseSeparable
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 深度可分离卷积处理
        self.conv = DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # 尺寸对齐
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        x = self.conv(x)
        return x


class LightFullScaleDecoder(nn.Module):
    """
    轻量化全尺度解码器：接收多个编码器层的特征
    相比原版减少了通道数，使用深度可分离卷积
    
    Reference: UNet3+ + UTANet++
    """
    def __init__(
        self, 
        filters_enc: List[int],  # 编码器各层通道数
        target_size: int,        # 目标输出尺寸
        cat_channels: int = 16,  # 统一的拼接通道数（轻量化）
        out_channels: int = 32   # 输出通道数
    ):
        super().__init__()
        self.cat_channels = cat_channels
        self.target_size = target_size
        
        # 为每个编码器层创建轻量化的特征转换
        self.transforms = nn.ModuleList()
        source_sizes = [112, 56, 28, 14, 14]  # 编码器各层的空间尺寸（下采样后）
        
        for i, (ch, src_size) in enumerate(zip(filters_enc, source_sizes)):
            transform = self._make_transform(ch, cat_channels, src_size, target_size)
            self.transforms.append(transform)
        
        # 融合模块：使用深度可分离卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(5 * cat_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 1)
        )
    
    def _make_transform(
        self, 
        in_ch: int, 
        out_ch: int, 
        src_size: int, 
        target_size: int
    ) -> nn.Module:
        """创建特征转换模块"""
        # 通道转换（使用1x1卷积，轻量）
        # 空间尺寸调整在forward中动态处理
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    
    def forward(
        self, 
        e1: torch.Tensor, 
        e2: torch.Tensor, 
        e3: torch.Tensor, 
        e4: torch.Tensor, 
        e5: torch.Tensor
    ) -> torch.Tensor:
        """融合所有编码器特征"""
        features = [e1, e2, e3, e4, e5]
        transformed = []
        
        for feat, transform in zip(features, self.transforms):
            # 先进行通道转换
            feat_transformed = transform(feat)
            
            # 然后调整空间尺寸到target_size
            if feat_transformed.shape[2] != self.target_size or feat_transformed.shape[3] != self.target_size:
                feat_transformed = F.interpolate(
                    feat_transformed, 
                    size=(self.target_size, self.target_size), 
                    mode='bilinear', 
                    align_corners=True
                )
            
            transformed.append(feat_transformed)
        
        # 拼接并融合
        concat = torch.cat(transformed, dim=1)
        out = self.fusion(concat)
        
        return out


class LightGatedAttention(nn.Module):
    """
    轻量化门控注意力模块
    简化版本：只使用空间注意力，移除通道注意力的SE模块
    
    Reference: AttU-Net (simplified)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        # 门控信号转换
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        # 跳跃连接特征转换
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        # 空间注意力
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 尺寸对齐
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        att = self.psi(psi)
        
        return x * att


class EfficientChannelAttention(nn.Module):
    """
    高效通道注意力（ECA）
    相比SE模块，使用1D卷积替代全连接层，更加轻量
    
    Reference: ECA-Net
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # 自适应计算卷积核大小
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局平均池化
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        # 1D卷积
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Sigmoid激活
        y = self.sigmoid(y)
        
        return x * y


class BottleneckASPP(nn.Module):
    """
    轻量化ASPP模块（Atrous Spatial Pyramid Pooling）
    使用深度可分离卷积实现多尺度特征提取
    """
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int] = [1, 2, 4, 8]):
        super().__init__()
        
        # 1x1卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )
        
        # 多尺度空洞卷积分支（使用深度可分离卷积）
        self.branches = nn.ModuleList()
        for dilation in dilations[1:]:
            branch = DepthwiseSeparableConv(
                in_channels, out_channels // 4, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation
            )
            self.branches.append(branch)
        
        # 全局池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels // 4 * (len(dilations) + 1), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        
        # 各分支处理
        out1 = self.conv1(x)
        
        branch_outs = [out1]
        for branch in self.branches:
            branch_outs.append(branch(x))
        
        # 全局池化
        global_out = self.global_pool(x)
        global_out = F.interpolate(global_out, size=size, mode='bilinear', align_corners=True)
        branch_outs.append(global_out)
        
        # 拼接并融合
        out = torch.cat(branch_outs, dim=1)
        out = self.fusion(out)
        
        return out


class DeepSupervisionHeadMicro(nn.Module):
    """
    微型深度监督头
    """
    def __init__(self, in_channels: int, num_classes: int, scale_factor: int, upsample: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)
        self.scale_factor = scale_factor
        self.upsample = upsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.upsample and self.scale_factor > 1:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return out
