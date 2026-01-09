"""
UTANet++ Auxiliary Modules
包含全尺度解码器、门控注意力和深度监督模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FullScaleDecoder(nn.Module):
    """
    全尺度解码器层 - 接收所有编码器层的特征
    
    借鉴UNet3+的思想，每个解码器层都能接收来自所有编码器层的信息，
    实现真正的全尺度特征融合。
    
    Args:
        filters_enc: 编码器各层的通道数 [64, 64, 128, 256, 512]
        target_size: 目标输出尺寸 (H, W)
        cat_channels: 统一的拼接通道数（默认64）
        out_channels: 输出通道数
    """
    def __init__(
        self, 
        filters_enc: List[int] = [64, 64, 128, 256, 512], 
        target_size: int = 56,
        cat_channels: int = 64,
        out_channels: int = 128
    ):
        super().__init__()
        self.cat_channels = cat_channels
        self.target_size = target_size
        
        # 为每个编码器层创建特征转换模块
        # e1: 需要下采样到目标尺寸
        scale_1 = 224 // target_size
        if scale_1 > 1:
            self.h1_transform = nn.Sequential(
                nn.MaxPool2d(scale_1, scale_1),
                nn.Conv2d(filters_enc[0], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.h1_transform = nn.Sequential(
                nn.Conv2d(filters_enc[0], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        
        # e2: 112x112
        scale_2 = 112 // target_size
        if scale_2 > 1:
            self.h2_transform = nn.Sequential(
                nn.MaxPool2d(scale_2, scale_2),
                nn.Conv2d(filters_enc[1], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        elif scale_2 < 1:
            self.h2_transform = nn.Sequential(
                nn.Upsample(scale_factor=target_size//112, mode='bilinear', align_corners=True),
                nn.Conv2d(filters_enc[1], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.h2_transform = nn.Sequential(
                nn.Conv2d(filters_enc[1], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        
        # e3: 56x56
        scale_3 = 56 // target_size
        if scale_3 > 1:
            self.h3_transform = nn.Sequential(
                nn.MaxPool2d(scale_3, scale_3),
                nn.Conv2d(filters_enc[2], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        elif scale_3 < 1:
            self.h3_transform = nn.Sequential(
                nn.Upsample(scale_factor=target_size//56, mode='bilinear', align_corners=True),
                nn.Conv2d(filters_enc[2], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.h3_transform = nn.Sequential(
                nn.Conv2d(filters_enc[2], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        
        # e4: 28x28
        scale_4 = target_size // 28
        if scale_4 > 1:
            self.h4_transform = nn.Sequential(
                nn.Upsample(scale_factor=scale_4, mode='bilinear', align_corners=True),
                nn.Conv2d(filters_enc[3], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        elif scale_4 < 1:
            self.h4_transform = nn.Sequential(
                nn.MaxPool2d(28//target_size, 28//target_size),
                nn.Conv2d(filters_enc[3], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.h4_transform = nn.Sequential(
                nn.Conv2d(filters_enc[3], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        
        # e5: 14x14
        scale_5 = target_size // 14
        if scale_5 > 1:
            self.h5_transform = nn.Sequential(
                nn.Upsample(scale_factor=scale_5, mode='bilinear', align_corners=True),
                nn.Conv2d(filters_enc[4], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        elif scale_5 < 1:
            self.h5_transform = nn.Sequential(
                nn.MaxPool2d(14//target_size, 14//target_size),
                nn.Conv2d(filters_enc[4], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.h5_transform = nn.Sequential(
                nn.Conv2d(filters_enc[4], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )
        
        # 融合模块：将5个cat_channels的特征融合为out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(5 * cat_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self, 
        e1: torch.Tensor, 
        e2: torch.Tensor, 
        e3: torch.Tensor, 
        e4: torch.Tensor, 
        e5: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：将所有编码器特征融合到目标尺寸
        
        Args:
            e1-e5: 编码器各层特征
            
        Returns:
            融合后的特征图
        """
        # 将所有特征调整到目标尺寸
        h1 = self.h1_transform(e1)  # (B, cat_channels, target_size, target_size)
        h2 = self.h2_transform(e2)  # (B, cat_channels, target_size, target_size)
        h3 = self.h3_transform(e3)  # (B, cat_channels, target_size, target_size)
        h4 = self.h4_transform(e4)  # (B, cat_channels, target_size, target_size)
        h5 = self.h5_transform(e5)  # (B, cat_channels, target_size, target_size)
        
        # 在通道维度拼接
        concat = torch.cat([h1, h2, h3, h4, h5], dim=1)  # (B, 5*cat_channels, H, W)
        
        # 融合特征
        out = self.fusion(concat)  # (B, out_channels, H, W)
        
        return out


class GatedAttention(nn.Module):
    """
    门控注意力模块 - 结合空间注意力和通道注意力
    
    改进自AttU-Net，引入门控机制和通道注意力，增强特征选择能力。
    
    Args:
        F_g: 门控信号（解码器特征）的通道数
        F_l: 跳跃连接特征的通道数
        F_int: 中间特征维度
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        # 门控信号转换（来自解码器）
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 跳跃连接特征转换
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 空间注意力生成
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 通道注意力（SE模块）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l, F_l // 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l // 16, F_l, 1, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对跳跃连接特征应用门控注意力
        
        Args:
            g: 门控信号（解码器特征）
            x: 跳跃连接特征（编码器特征）
            
        Returns:
            经过注意力加权的特征
        """
        # 空间注意力计算
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 如果尺寸不匹配，调整g1的尺寸
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        spatial_att = self.psi(psi)  # (B, 1, H, W)
        
        # 通道注意力计算
        channel_att = self.channel_att(x)  # (B, F_l, 1, 1)
        
        # 联合注意力：空间 × 通道
        out = x * spatial_att * channel_att
        
        return out


class DeepSupervisionHead(nn.Module):
    """
    深度监督头 - 为每个解码器层生成辅助输出
    
    在训练时提供额外的监督信号，帮助网络学习更好的特征表示。
    
    Args:
        in_channels: 输入特征通道数
        num_classes: 输出类别数
        scale_factor: 上采样倍数（恢复到原始图像尺寸）
    """
    def __init__(self, in_channels: int, num_classes: int, scale_factor: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)
        self.scale_factor = scale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：生成分割输出
        
        Args:
            x: 解码器特征
            
        Returns:
            上采样到原始尺寸的分割输出
        """
        out = self.conv(x)
        if self.scale_factor > 1:
            out = F.interpolate(
                out, 
                scale_factor=self.scale_factor, 
                mode='bilinear', 
                align_corners=True
            )
        return out


class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失函数
    
    结合主输出、多个辅助输出和MoE负载均衡损失。
    
    Args:
        weights: 各辅助输出的权重 [d1, d2, d3, d4]
        moe_weight: MoE损失的权重
    """
    def __init__(
        self, 
        weights: List[float] = [0.5, 0.3, 0.15, 0.05],
        moe_weight: float = 0.01
    ):
        super().__init__()
        self.weights = weights
        self.moe_weight = moe_weight
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self, 
        main_out: torch.Tensor, 
        aux_outs: List[torch.Tensor], 
        target: torch.Tensor,
        moe_loss: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算深度监督损失
        
        Args:
            main_out: 主输出
            aux_outs: 辅助输出列表 [ds1, ds2, ds3, ds4]
            target: 目标标签
            moe_loss: MoE负载均衡损失
            
        Returns:
            总损失和各项损失的字典
        """
        # 主输出损失
        main_loss = self.criterion(main_out, target)
        
        # 辅助输出损失
        aux_loss = 0
        loss_dict = {'main': main_loss.item()}
        
        for i, aux_out in enumerate(aux_outs):
            aux_loss_i = self.criterion(aux_out, target)
            aux_loss += self.weights[i] * aux_loss_i
            loss_dict[f'aux_{i+1}'] = aux_loss_i.item()
        
        # 总损失
        total_loss = main_loss + aux_loss
        
        # 添加MoE损失
        if moe_loss is not None:
            total_loss += self.moe_weight * moe_loss
            loss_dict['moe'] = moe_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict

