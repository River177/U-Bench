"""
UTANet with Edge-Aware Branch: 结合边界检测的UTANet

基于原始UTANet，添加双解码器分支：
1. 主解码器分支：标准的U-Net解码器
2. 边界解码器分支：使用2D Sobel算子增强边界特征

架构：
    编码器（共享）
    ↙         ↘
主分支      边界分支(Sobel)
    ↘         ↙
    特征融合
        ↓
    最终输出

Reference:
- UTANet: https://ojs.aaai.org/index.php/AAAI/article/view/32627
- SBCNet Sobel Branch

Usage:
    from UTANet_EdgeBranch import UTANetEdge
    model = UTANetEdge(pretrained=True, n_classes=1)
    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
from .ta_mosc import MoE
import numpy as np


class Sobel2D(nn.Module):
    """
    2D Sobel边缘检测算子
    
    在X和Y两个方向上计算图像梯度，用于提取边缘特征。
    适用于2D医学图像分割中的边界增强。
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: 输入通道数
        """
        super().__init__()
        self.channels = channels
        
        # Sobel算子：X方向（垂直边缘）
        # [[-1, 0, 1],
        #  [-2, 0, 2],
        #  [-1, 0, 1]]
        self.kernel_x = np.array([[-1, 0, 1], 
                                   [-2, 0, 2], 
                                   [-1, 0, 1]], dtype=np.float32)
        
        # Sobel算子：Y方向（水平边缘）
        # [[-1, -2, -1],
        #  [ 0,  0,  0],
        #  [ 1,  2,  1]]
        self.kernel_y = np.array([[-1, -2, -1], 
                                   [ 0,  0,  0], 
                                   [ 1,  2,  1]], dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算Sobel边缘响应
        
        Args:
            x: 输入特征图 (B, C, H, W)
        
        Returns:
            边缘特征图 (B, C, H, W)，值为梯度幅值
        """
        # 将numpy数组转换为torch tensor并扩展维度
        # (3, 3) -> (1, 1, 3, 3) -> (C, 1, 3, 3)
        sobel_kernel_x = torch.from_numpy(self.kernel_x).unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.expand(self.channels, 1, 3, 3).float().to(x.device)
        
        sobel_kernel_y = torch.from_numpy(self.kernel_y).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = sobel_kernel_y.expand(self.channels, 1, 3, 3).float().to(x.device)
        
        # 计算X和Y方向的梯度（使用分组卷积）
        G_x = F.conv2d(x, sobel_kernel_x, stride=1, padding=1, groups=self.channels)
        G_y = F.conv2d(x, sobel_kernel_y, stride=1, padding=1, groups=self.channels)
        
        # 计算梯度幅值：sqrt(Gx^2 + Gy^2)
        edge_magnitude = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + 1e-6)
        
        return edge_magnitude


class SoberConv2D(nn.Module):
    """
    Sobel增强的2D卷积块
    
    双路径设计：
    - 主路径：标准卷积
    - Sobel路径：边缘特征提取和增强
    最后融合两条路径的特征
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        
        # 主卷积路径
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Sobel边缘增强路径
        self.sobel_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//2, kernel_size=1),  # 降维
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.LeakyReLU(inplace=True),
            Sobel2D(out_channels//2),  # Sobel边缘检测
            nn.BatchNorm2d(out_channels//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1)  # 升维
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 融合层：将主路径和Sobel路径合并
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (B, in_channels, H, W)
        
        Returns:
            边缘增强特征 (B, out_channels, H, W)
        """
        # 主路径
        out = self.conv1(x)
        
        # Sobel边缘路径
        edge_features = self.sobel_branch(out)
        
        # 融合两条路径
        fused = torch.cat([out, edge_features], dim=1)
        fused = self.fusion(fused)
        
        # 第二次卷积
        out = self.conv2(fused)
        
        return out


class UpBlockEdge(nn.Module):
    """
    边界增强的上采样块（用于边界解码分支）
    
    使用SoberConv2D替代标准卷积，增强边界特征
    """
    def __init__(
        self, 
        in_ch: int, 
        skip_ch: int, 
        out_ch: int, 
        img_size: int, 
        scale_factor: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.scale_factor = scale_factor or (img_size // 14, img_size // 14)
        
        # 转置卷积上采样
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True)
        )
        
        # 使用Sobel增强卷积处理融合后的特征
        self.conv = SoberConv2D(in_ch//2 + skip_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_feat: 解码器特征 (B, in_ch, H, W)
            skip_feat: 编码器跳跃连接特征 (B, skip_ch, H, W)
        
        Returns:
            融合后的特征 (B, out_ch, 2H, 2W)
        """
        up_feat = self.up(decoder_feat)
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        return self.conv(fused_feat)


class UpBlockStandard(nn.Module):
    """标准上采样块（用于主解码分支）"""
    def __init__(
        self, 
        in_ch: int, 
        skip_ch: int, 
        out_ch: int, 
        img_size: int, 
        scale_factor: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.scale_factor = scale_factor or (img_size // 14, img_size // 14)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2 + skip_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        up_feat = self.up(decoder_feat)
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        return self.conv(fused_feat)


class Flatten(nn.Module):
    """Flatten a tensor into a 2D matrix"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class UTANetEdge(nn.Module):
    """
    UTANet with Edge-Aware Branch
    
    结合UTANet的TA-MoSC机制和SBCNet的边界检测能力的双分支架构：
    1. 编码器：ResNet34提取多尺度特征
    2. TA-MoSC：自适应特征融合和路由
    3. 双解码器：
       - 主分支：标准U-Net解码器
       - 边界分支：Sobel增强的边界解码器
    4. 特征融合：融合两个分支的输出
    """
    def __init__(
        self, 
        pretrained: bool = True, 
        topk: int = 2, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        img_size: int = 224,
        edge_weight: float = 0.5
    ):
        """
        Args:
            pretrained: 是否使用TA-MoSC模块
            topk: MoE中top-k专家数量
            n_channels: 输入通道数
            n_classes: 输出类别数
            img_size: 输入图像尺寸
            edge_weight: 边界分支的权重（用于加权融合）
        """
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size
        self.edge_weight = edge_weight

        # ========== 编码器：ResNet34 ==========
        try:
            # 新版本 torchvision (0.13+) 使用 weights 参数
            from torchvision.models import ResNet34_Weights
            self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # 旧版本 torchvision 使用 pretrained 参数
            self.resnet = models.resnet34(pretrained=True)
        
        self.filters_resnet = [64, 64, 128, 256, 512]
        self.filters_decoder = [32, 64, 128, 256, 512]

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        # ResNet各层
        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4

        # ========== TA-MoSC模块 ==========
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

        # ========== 主解码器分支 ==========
        self.up5_main = UpBlockStandard(self.filters_resnet[4], self.filters_resnet[3], 
                                        self.filters_decoder[3], 28)
        self.up4_main = UpBlockStandard(self.filters_decoder[3], self.filters_resnet[2], 
                                        self.filters_decoder[2], 56)
        self.up3_main = UpBlockStandard(self.filters_decoder[2], self.filters_resnet[1], 
                                        self.filters_decoder[1], 112)
        self.up2_main = UpBlockStandard(self.filters_decoder[1], self.filters_resnet[0], 
                                        self.filters_decoder[0], 224)

        # ========== 边界解码器分支（使用Sobel增强）==========
        self.up5_edge = UpBlockEdge(self.filters_resnet[4], self.filters_resnet[3], 
                                    self.filters_decoder[3], 28)
        self.up4_edge = UpBlockEdge(self.filters_decoder[3], self.filters_resnet[2], 
                                    self.filters_decoder[2], 56)
        self.up3_edge = UpBlockEdge(self.filters_decoder[2], self.filters_resnet[1], 
                                    self.filters_decoder[1], 112)
        self.up2_edge = UpBlockEdge(self.filters_decoder[1], self.filters_resnet[0], 
                                    self.filters_decoder[0], 224)

        # ========== 预测头 ==========
        # 主分支输出
        self.pred_main = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0]//2, 1),
            nn.BatchNorm2d(self.filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0]//2, n_classes, 1)
        )
        
        # 边界分支输出
        self.pred_edge = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0]//2, 1),
            nn.BatchNorm2d(self.filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0]//2, n_classes, 1)
        )
        
        # 融合输出
        self.pred_final = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0]*2, self.filters_decoder[0], 1),
            nn.BatchNorm2d(self.filters_decoder[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0], n_classes, 1)
        )

        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """创建特征路由器"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, n_channels, H, W)
        
        Returns:
            最终融合的输出 (B, n_classes, H, W)
        
        Note:
            训练框架只接收单一输出张量，因此只返回最终融合输出。
            辅助输出（主分支、边界分支）在内部计算但不返回。
        """
        # ========== 编码器 ==========
        e1 = self.conv1(x)           # (B, 64, 224, 224)
        e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112)
        e2 = self.conv2(e1_maxp)     # (B, 64, 112, 112)
        e3 = self.conv3(e2)          # (B, 128, 56, 56)
        e4 = self.conv4(e3)          # (B, 256, 28, 28)
        e5 = self.conv5(e4)          # (B, 512, 14, 14)

        aux_loss = torch.tensor(0.0, device=x.device)

        # ========== TA-MoSC模块 ==========
        if self.pretrained:
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear')
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear')
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear')
            
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)
            
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss
            
            o1 = self.docker1(o1)
            o2 = self.docker2(o2)
            o3 = self.docker3(o3)
            o4 = self.docker4(o4)
            
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear')
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear')
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear')
        else:
            o1, o2, o3, o4 = e1, e2, e3, e4

        # ========== 主解码器分支 ==========
        d4_main = self.up5_main(e5, o4)  # (B, 256, 28, 28)
        d3_main = self.up4_main(d4_main, o3)  # (B, 128, 56, 56)
        d2_main = self.up3_main(d3_main, o2)  # (B, 64, 112, 112)
        d1_main = self.up2_main(d2_main, o1)  # (B, 32, 224, 224)

        # ========== 边界解码器分支 ==========
        d4_edge = self.up5_edge(e5, o4)  # (B, 256, 28, 28)
        d3_edge = self.up4_edge(d4_edge, o3)  # (B, 128, 56, 56)
        d2_edge = self.up3_edge(d3_edge, o2)  # (B, 64, 112, 112)
        d1_edge = self.up2_edge(d2_edge, o1)  # (B, 32, 224, 224)

        # ========== 预测输出 ==========
        out_main = self.pred_main(d1_main)  # 主分支输出
        out_edge = self.pred_edge(d1_edge)  # 边界分支输出
        
        # 融合两个分支的特征
        d1_fused = torch.cat([d1_main, d1_edge], dim=1)
        out_final = self.pred_final(d1_fused)  # 最终融合输出

        # 训练框架只接收单一输出张量，这里仅返回最终融合输出
        # 辅助输出（out_main, out_edge）和辅助损失（aux_loss）在内部计算但不返回
        return out_final


def utanet_edge(input_channel=3, num_classes=1, edge_weight=0.5):
    """
    创建UTANetEdge模型的便捷函数
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        edge_weight: 边界分支权重
    
    Returns:
        UTANetEdge模型实例
    """
    return UTANetEdge(n_channels=input_channel, n_classes=num_classes, edge_weight=edge_weight)


if __name__ == "__main__":
    print("=" * 80)
    print("UTANet with Edge-Aware Branch 模型测试")
    print("=" * 80)
    
    input_tensor = torch.randn(2, 3, 224, 224)
    
    # 测试阶段1（不使用TA-MoSC）
    print("\n阶段1（pretrained=False）:")
    model_stage1 = UTANetEdge(pretrained=False, n_classes=1)
    model_stage1.train()
    
    outputs1 = model_stage1(input_tensor)
    if isinstance(outputs1, tuple):
        final, main, edge = outputs1
        print(f"✓ 输入形状: {input_tensor.shape}")
        print(f"✓ 最终输出: {final.shape}")
        print(f"✓ 主分支输出: {main.shape}")
        print(f"✓ 边界分支输出: {edge.shape}")
    
    total_params = sum(p.numel() for p in model_stage1.parameters())
    print(f"✓ 总参数量: {total_params:,}")
    
    # 测试阶段2（使用TA-MoSC）
    print("\n阶段2（pretrained=True）:")
    model_stage2 = UTANetEdge(pretrained=True, n_classes=1)
    model_stage2.train()
    
    outputs2 = model_stage2(input_tensor)
    if isinstance(outputs2, tuple):
        final, main, edge, aux_loss = outputs2
        print(f"✓ 输入形状: {input_tensor.shape}")
        print(f"✓ 最终输出: {final.shape}")
        print(f"✓ 主分支输出: {main.shape}")
        print(f"✓ 边界分支输出: {edge.shape}")
        print(f"✓ 辅助损失: {aux_loss.item():.6f}")
    
    total_params = sum(p.numel() for p in model_stage2.parameters())
    print(f"✓ 总参数量: {total_params:,}")
    
    # 测试推理模式
    print("\n推理模式:")
    model_stage2.eval()
    with torch.no_grad():
        output_infer = model_stage2(input_tensor)
    print(f"✓ 推理输出: {output_infer.shape}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    print("\n架构特点:")
    print("✓ 双解码器分支：主分支 + 边界增强分支")
    print("✓ Sobel边缘检测：2D Sobel算子（X和Y方向）")
    print("✓ TA-MoSC自适应路由：动态选择top-k专家")
    print("✓ 特征融合：两个分支的特征互补融合")
    print("✓ 多输出监督：最终输出 + 主分支 + 边界分支")

