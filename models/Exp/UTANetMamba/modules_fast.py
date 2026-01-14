"""
UTANet+ Fast Auxiliary Modules
包含轻量化全尺度解码器（FastFullScaleDecoder）和深度监督模块
针对显存占用进行了优化：降低融合通道数，裁剪非必要连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class FastFullScaleDecoder(nn.Module):
    """
    轻量化全尺度解码器 - 动态接收特征并进行融合
    
    改进点：
    1. 动态输入：不再强制接收5个特征，而是根据初始化时提供的 in_channels_list 决定
    2. 通道减少：cat_channels 默认降低
    3. 灵活剪枝：可以通过传入较少的特征来实现连接剪枝
    
    Args:
        in_channels_list: 输入特征的通道数列表。例如 [64, 128, 256]
        in_scales_list: 输入特征相对于目标尺寸的缩放比例列表。
                       >1 代表输入比目标大 (需要 Pooling)
                       <1 代表输入比目标小 (需要 Upsample)
                       =1 代表尺寸相同
                       注意：这里简化逻辑，直接根据特征图尺寸动态调整，或者预先定义好。
                       为了简单起见，我们假设输入时已经知道每个输入的通道数。
        target_size: 目标输出尺寸 (仅用于参考，实际 forward 中动态计算或使用固定值)
        cat_channels: 统一的拼接通道数 (默认 32，比原版 64 减少一半)
        out_channels: 输出通道数
    """
    def __init__(
        self, 
        in_channels_list: List[int],
        cat_channels: int = 32,
        out_channels: int = 128
    ):
        super().__init__()
        self.cat_channels = cat_channels
        self.num_inputs = len(in_channels_list)
        
        # 为每个输入特征创建转换层
        self.transforms = nn.ModuleList()
        for in_ch in in_channels_list:
            # 这里统一使用 Conv+BN+ReLU
            # 具体的上/下采样操作在 forward 中根据尺寸动态进行
            # 这样更加灵活，不需要预先知道具体的 scale factor
            self.transforms.append(nn.Sequential(
                nn.Conv2d(in_ch, cat_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(self.num_inputs * cat_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征列表，必须与初始化时的 in_channels_list 长度一致
            
        Returns:
            融合后的特征图
        """
        if len(features) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} features, got {len(features)}")
            
        # 以第一个特征的尺寸作为目标尺寸 (通常第一个特征是主路径特征或当前层特征)
        # 或者我们可以约定第一个特征就是我们要对齐的目标
        target_h, target_w = features[0].size()[2], features[0].size()[3]
        
        processed_features = []
        for i, (feat, transform) in enumerate(zip(features, self.transforms)):
            # 1. 特征变换 (Conv 改变通道数)
            x = transform(feat)
            
            # 2. 尺寸调整 (Upsample or Pool)
            h, w = x.size()[2], x.size()[3]
            
            if h > target_h and w > target_w:
                # 输入比目标大 -> MaxPool
                # 假设是整数倍
                kernel_h = h // target_h
                kernel_w = w // target_w
                x = F.max_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
            elif h < target_h and w < target_w:
                # 输入比目标小 -> Upsample
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
            
            processed_features.append(x)
        
        # 3. 拼接
        concat = torch.cat(processed_features, dim=1)
        
        # 4. 融合
        out = self.fusion(concat)
        
        return out


class GatedAttention(nn.Module):
    """
    门控注意力模块 - 保持原版设计
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l, max(1, F_l // 16), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, F_l // 16), F_l, 1, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        spatial_att = self.psi(psi)
        channel_att = self.channel_att(x)
        
        return x * spatial_att * channel_att


class DeepSupervisionHead(nn.Module):
    """
    深度监督头
    """
    def __init__(self, in_channels: int, num_classes: int, scale_factor: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)
        self.scale_factor = scale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        main_loss = self.criterion(main_out, target)
        aux_loss = 0
        loss_dict = {'main': main_loss.item()}
        
        for i, aux_out in enumerate(aux_outs):
            if i < len(self.weights):
                aux_loss_i = self.criterion(aux_out, target)
                aux_loss += self.weights[i] * aux_loss_i
                loss_dict[f'aux_{i+1}'] = aux_loss_i.item()
        
        total_loss = main_loss + aux_loss
        
        if moe_loss is not None:
            total_loss += self.moe_weight * moe_loss
            loss_dict['moe'] = moe_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
