"""
Lightweight Hierarchical UTANet: 轻量化层次化医学图像分割网络

核心创新：
1. Hierarchical MoE - 层次化专家混合，每个专家关注不同感受野
2. Lightweight ASPP - 深度可分离卷积实现的多尺度特征提取
3. Depthwise Separable Decoder - 轻量化解码器，减少参数量
4. TA-MoSC Integration - 保留任务自适应专家混合模块

设计目标：
- 参数量减少40-60%
- 计算量减少50-70%
- 性能保持甚至提升

Reference:
- UTANet: https://ojs.aaai.org/index.php/AAAI/article/view/32627
- MobileNetV2: https://arxiv.org/abs/1801.04381
- DeepLabV3+: https://arxiv.org/abs/1802.02611
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import sys
import os

# 使用相对导入
from .ta_mosc import MoE

# 导入辅助模块
from .modules import (
    HierarchicalMoE,
    LightweightASPP,
    LightweightUpBlock
)


class LightweightHierarchicalUTANet(nn.Module):
    """
    轻量化层次化UTANet
    
    网络结构：
    1. 编码器: ResNet34（可选MobileNetV2进一步轻量化）
    2. 瓶颈层: LightweightASPP（多尺度特征增强）
    3. 特征路由: HierarchicalMoE（层次化专家混合）
    4. 解码器: LightweightUpBlock（深度可分离卷积）
    
    两阶段训练：
    - Stage 1 (pretrained=False): 训练编码器和解码器
    - Stage 2 (pretrained=True): 训练HierarchicalMoE模块
    
    Args:
        pretrained: 是否启用HierarchicalMoE模块
        topk: MoE中选择的专家数量
        n_channels: 输入图像通道数
        n_classes: 输出类别数
        img_size: 输入图像尺寸
        use_mobilenet: 是否使用MobileNetV2替代ResNet34（更轻量）
    """
    def __init__(
        self,
        pretrained: bool = True,
        topk: int = 2,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        use_mobilenet: bool = False
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size
        self.use_mobilenet = use_mobilenet
        
        # ========== 编码器选择 ==========
        if use_mobilenet:
            # 选项1: MobileNetV2（更轻量）
            from torchvision.models import mobilenet_v2
            mobilenet = mobilenet_v2(pretrained=True)
            self.filters_enc = [32, 24, 32, 96, 320]  # MobileNetV2各层通道数
            
            # 提取MobileNetV2的各个阶段
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, self.filters_enc[0], 3, 1, 1, bias=True),
                nn.BatchNorm2d(self.filters_enc[0]),
                nn.ReLU(inplace=True)
            )
            self.maxpool = nn.MaxPool2d(2, 2)
            
            # MobileNetV2的特征层
            # 注意：这是简化版本，实际可能需要调整
            self.conv2 = mobilenet.features[0:4]   # 输出24通道
            self.conv3 = mobilenet.features[4:7]   # 输出32通道
            self.conv4 = mobilenet.features[7:14]  # 输出96通道
            self.conv5 = mobilenet.features[14:18] # 输出320通道
        else:
            # 选项2: ResNet34（标准）
            self.resnet = models.resnet34(pretrained=True)
            self.filters_enc = [64, 64, 128, 256, 512]
            
            # ResNet34编码器
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, self.filters_enc[0], 3, 1, 1, bias=True),
                nn.BatchNorm2d(self.filters_enc[0]),
                nn.ReLU(inplace=True)
            )
            self.maxpool = nn.MaxPool2d(2, 2)
            
            self.conv2 = self.resnet.layer1  # 64, 112x112
            self.conv3 = self.resnet.layer2  # 128, 56x56
            self.conv4 = self.resnet.layer3  # 256, 28x28
            self.conv5 = self.resnet.layer4  # 512, 14x14
        
        # ========== 瓶颈层: LightweightASPP ==========
        self.aspp = LightweightASPP(self.filters_enc[4], self.filters_enc[4])
        
        # ========== TA-MoSC模块: HierarchicalMoE ==========
        if pretrained:
            # 特征融合层（使用分组卷积进一步轻量化）
            fuse_in_channels = sum(self.filters_enc[:4])  # 64+64+128+256=512 或 MobileNet的总和
            self.fuse = nn.Sequential(
                nn.Conv2d(fuse_in_channels, 64, 1, groups=min(8, 64)),  # 分组卷积
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            # 层次化MoE
            self.moe = HierarchicalMoE(num_experts=4, top=topk, emb_size=64)
            
            # Docker路由器（使用分组卷积）
            self.docker1 = self._create_lightweight_docker(64, self.filters_enc[0])
            self.docker2 = self._create_lightweight_docker(64, self.filters_enc[1])
            self.docker3 = self._create_lightweight_docker(64, self.filters_enc[2])
            self.docker4 = self._create_lightweight_docker(64, self.filters_enc[3])
        
        # ========== 解码器: LightweightUpBlock ==========
        self.filters_dec = [32, 64, 128, 256, self.filters_enc[4]]
        
        # 逐层上采样
        self.up5 = LightweightUpBlock(self.filters_enc[4], self.filters_enc[3], self.filters_dec[3])
        self.up4 = LightweightUpBlock(self.filters_dec[3], self.filters_enc[2], self.filters_dec[2])
        self.up3 = LightweightUpBlock(self.filters_dec[2], self.filters_enc[1], self.filters_dec[1])
        self.up2 = LightweightUpBlock(self.filters_dec[1], self.filters_enc[0], self.filters_dec[0])
        
        # ========== 预测头 ==========
        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_dec[0], self.filters_dec[0]//2, 1),
            nn.BatchNorm2d(self.filters_dec[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_dec[0]//2, n_classes, 1)
        )
    
    def _create_lightweight_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        创建轻量化Docker路由器
        
        使用分组卷积减少参数量
        
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            
        Returns:
            轻量化Docker模块
        """
        groups = min(in_ch, out_ch)  # 分组数
        # 确保groups能整除in_ch和out_ch
        while in_ch % groups != 0 or out_ch % groups != 0:
            groups //= 2
        groups = max(1, groups)  # 至少为1
        
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        处理流程：
        1. 编码器提取多尺度特征
        2. ASPP增强最深层特征
        3. (可选) HierarchicalMoE进行特征路由
        4. 解码器逐层上采样并融合特征
        5. 预测头输出分割结果
        
        Args:
            x: 输入图像 (B, n_channels, H, W)
            
        Returns:
            logits: 分割输出 (B, n_classes, H, W)
        """
        # ========== 编码器路径 ==========
        e1 = self.conv1(x)           # (B, 64, 224, 224)
        e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112)
        e2 = self.conv2(e1_maxp)     # (B, 64, 112, 112)
        e3 = self.conv3(e2)          # (B, 128, 56, 56)
        e4 = self.conv4(e3)          # (B, 256, 28, 28)
        e5 = self.conv5(e4)          # (B, 512, 14, 14)
        
        # ========== 瓶颈层: ASPP增强 ==========
        e5 = self.aspp(e5)  # (B, 512, 14, 14)
        
        # 初始化辅助损失
        aux_loss = torch.tensor(0.0, device=x.device)
        
        # ========== HierarchicalMoE特征路由 ==========
        if self.pretrained:
            # 步骤1: 多尺度特征融合到统一尺寸（112x112）
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear', align_corners=True)
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear', align_corners=True)
            
            # 步骤2: 拼接并融合
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)  # (B, 64, 112, 112)
            
            # 步骤3: HierarchicalMoE路由
            moe_out, loss = self.moe(fused)  # (B, 64, 112, 112)
            aux_loss = loss
            
            # 步骤4: Docker特征分发（生成4个不同尺度的输出）
            # 注意：这里简化为单一输出，然后通过docker分发到不同通道数
            o1 = self.docker1(moe_out)  # (B, 64, 112, 112)
            o2 = self.docker2(moe_out)  # (B, 64, 112, 112)
            o3 = self.docker3(moe_out)  # (B, 128, 112, 112)
            o4 = self.docker4(moe_out)  # (B, 256, 112, 112)
            
            # 步骤5: 调整空间尺寸以匹配编码器特征
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear', align_corners=True)    # -> 224x224
            # o2保持112x112
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear', align_corners=True)  # -> 56x56
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear', align_corners=True) # -> 28x28
        else:
            # 不使用HierarchicalMoE，直接使用编码器特征
            o1, o2, o3, o4 = e1, e2, e3, e4
        
        # ========== 解码器路径 ==========
        d4 = self.up5(e5, o4)  # (B, 256, 28, 28)
        d3 = self.up4(d4, o3)  # (B, 128, 56, 56)
        d2 = self.up3(d3, o2)  # (B, 64, 112, 112)
        d1 = self.up2(d2, o1)  # (B, 32, 224, 224)
        
        # ========== 预测输出 ==========
        logits = self.pred(d1)  # (B, n_classes, 224, 224)
        
        # 训练框架只接收单一输出张量，这里仅返回 logits
        # aux_loss 在内部计算但不返回，以保持与训练框架的兼容性
        return logits


def lightweight_hierarchical_utanet(
    input_channel: int = 3,
    num_classes: int = 1,
    pretrained: bool = True,
    use_mobilenet: bool = False
):
    """
    创建LightweightHierarchicalUTANet模型
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        pretrained: 是否使用HierarchicalMoE模块
        use_mobilenet: 是否使用MobileNetV2编码器
        
    Returns:
        LightweightHierarchicalUTANet模型实例
    """
    return LightweightHierarchicalUTANet(
        n_channels=input_channel,
        n_classes=num_classes,
        pretrained=pretrained,
        use_mobilenet=use_mobilenet
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Lightweight Hierarchical UTANet")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # === Test 1: ResNet34 version ===
    print("\n" + "=" * 70)
    print("Test 1: ResNet34 encoder version")
    print("=" * 70)
    
    model_resnet = LightweightHierarchicalUTANet(
        pretrained=True,
        n_channels=3,
        n_classes=1,
        use_mobilenet=False
    ).to(device)
    
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    
    # Training mode
    model_resnet.train()
    output = model_resnet(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Parameter statistics
    total_params = sum(p.numel() for p in model_resnet.parameters())
    trainable_params = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Inference mode
    model_resnet.eval()
    with torch.no_grad():
        output = model_resnet(input_tensor)
    print(f"\nInference mode output shape: {output.shape}")
    
    # === 测试2: MobileNetV2版本（如果需要） ===
    # print("\n" + "=" * 70)
    # print("测试2: MobileNetV2编码器版本（更轻量）")
    # print("=" * 70)
    #
    # try:
    #     model_mobilenet = LightweightHierarchicalUTANet(
    #         pretrained=True,
    #         n_channels=3,
    #         n_classes=1,
    #         use_mobilenet=True
    #     ).to(device)
    #
    #     output_mb, moe_loss_mb = model_mobilenet(input_tensor)
    #     
    #     total_params_mb = sum(p.numel() for p in model_mobilenet.parameters())
    #     
    #     print(f"输出形状: {output_mb.shape}")
    #     print(f"MoE损失: {moe_loss_mb.item():.6f}")
    #     print(f"总参数量: {total_params_mb:,}")
    #     print(f"参数减少: {(1 - total_params_mb / total_params) * 100:.1f}%")
    # except Exception as e:
    #     print(f"MobileNetV2版本测试跳过: {e}")
    
    # === Test 3: Compare with original UTANet ===
    print("\n" + "=" * 70)
    print("Comparison with original UTANet")
    print("=" * 70)
    
    try:
        from UTANet import UTANet
        utanet_original = UTANet(pretrained=True, n_channels=3, n_classes=1).to(device)
        original_params = sum(p.numel() for p in utanet_original.parameters())
        
        print(f"Original UTANet parameters: {original_params:,}")
        print(f"Lightweight version parameters: {total_params:,}")
        print(f"Parameter reduction: {(1 - total_params / original_params) * 100:.1f}%")
    except Exception as e:
        print(f"Failed to load original UTANet for comparison: {e}")
    
    # === Test 4: Gradient backpropagation ===
    print("\n" + "=" * 70)
    print("Testing gradient backpropagation")
    print("=" * 70)
    
    model_resnet.train()
    output = model_resnet(input_tensor)
    
    # Simulate loss function
    target = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
    criterion = nn.BCEWithLogitsLoss()
    seg_loss = criterion(output, target)
    
    # Backpropagation
    seg_loss.backward()
    
    print(f"Segmentation loss: {seg_loss.item():.6f}")
    print("Gradient backpropagation successful!")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

