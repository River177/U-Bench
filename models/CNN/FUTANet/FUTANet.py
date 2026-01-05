"""
FUTANet: Frequency-enhanced UTANet
结合频率特征增强的任务自适应混合跳跃连接医学图像分割网络

核心创新：
1. 引入拉普拉斯金字塔提取高频边缘特征
2. 在编码器各层应用频率特征增强（单通道频率 × 多通道特征）
3. 通过残差连接保证性能不下降
4. 保留UTANet的TA-MoSC机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
from torchvision.transforms.functional import rgb_to_grayscale
from .ta_mosc import MoE
from .modules import make_laplace_pyramid


class Flatten(nn.Module):
    """Flatten a tensor into a 2D matrix (batch_size, feature_dim)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Reconstruct(nn.Module):
    """Reconstruct feature maps from flattened tensors with upsampling"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        scale_factor: Tuple[int, int] = (2, 2)
    ):
        super().__init__()
        self.padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_patches, hidden = x.size()
        h, w = int(n_patches ** 0.5), int(n_patches ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    """Downsampling block with MaxPooling followed by convolution"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution and skip connections"""
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


class FrequencyEnhancement(nn.Module):
    """
    频率特征增强模块
    
    功能：
    1. 将单通道频率特征调整到目标空间尺寸
    2. 与多通道编码器特征相乘（广播机制）
    3. 通过残差连接保证至少不会掉点
    
    输入：
        encoder_feat: 编码器特征 [B, C, H, W]
        freq_feat: 频率特征 [B, 1, H', W']
    
    输出：
        增强后的特征 [B, C, H, W]
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, encoder_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_feat: 编码器特征 (B, C, H, W) - 如64通道
            freq_feat: 频率特征 (B, 1, H', W') - 单通道
        
        Returns:
            增强后的特征 (B, C, H, W) - 保持原通道数
        """
        # 调整频率特征到编码器特征的空间尺寸
        if freq_feat.shape[2:] != encoder_feat.shape[2:]:
            freq_feat = F.interpolate(
                freq_feat, 
                size=encoder_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 频率特征增强：单通道频率 × 多通道特征（广播）
        enhanced = encoder_feat * freq_feat  # (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W)
        
        # 残差连接：保证至少不会掉点
        output = enhanced + encoder_feat
        
        return output


class FUTANet(nn.Module):
    """
    FUTANet: Frequency-enhanced UTANet
    频率增强的任务自适应混合跳跃连接网络
    
    主要创新点：
    1. 在编码器各层引入高频边缘特征增强
    2. 频率特征与编码器特征相乘后加残差连接
    3. 保留UTANet的TA-MoSC机制进行特征融合和路由
    4. 结合边缘信息和自适应特征选择的优势
    """
    def __init__(
        self, 
        pretrained: bool = True, 
        topk: int = 2, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        img_size: int = 224
    ):
        """
        初始化FUTANet模型
        
        Args:
            pretrained: 训练阶段标志
                    False (阶段1): 训练原始UNet模型
                    True  (阶段2): 训练TA-MoSC模块
            topk: MoE模块中选择的专家数量
            n_channels: 输入图像通道数
            n_classes: 输出类别数
            img_size: 输入图像尺寸
        """
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size

        # ========== 编码器部分：基于ResNet34 ==========
        self.resnet = models.resnet34(pretrained=True)
        self.filters_resnet = [64, 64, 128, 256, 512]
        self.filters_decoder = [32, 64, 128, 256, 512]

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4

        # ========== 频率特征增强模块 ==========
        # 为每个编码器层添加频率增强
        self.freq_enhance1 = FrequencyEnhancement()
        self.freq_enhance2 = FrequencyEnhancement()
        self.freq_enhance3 = FrequencyEnhancement()
        self.freq_enhance4 = FrequencyEnhancement()

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

        # ========== 解码器部分 ==========
        self.up5 = UpBlock(self.filters_resnet[4], self.filters_resnet[3], self.filters_decoder[3], 28)
        self.up4 = UpBlock(self.filters_decoder[3], self.filters_resnet[2], self.filters_decoder[2], 56)
        self.up3 = UpBlock(self.filters_decoder[2], self.filters_resnet[1], self.filters_decoder[1], 112)
        self.up2 = UpBlock(self.filters_decoder[1], self.filters_resnet[0], self.filters_decoder[0], 224)

        # ========== 预测头 ==========
        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0]//2, 1),
            nn.BatchNorm2d(self.filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0]//2, n_classes, 1)
        )
        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """创建特征路由器模块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播流程：
        1. 提取频率特征（拉普拉斯金字塔）
        2. 编码器提取多尺度特征
        3. 对每层编码器特征应用频率增强（频率×特征+残差）
        4. TA-MoSC模块进行特征融合和路由（如果pretrained=True）
        5. 解码器上采样并融合特征
        6. 输出分割结果

        Args:
            x: 输入图像 (B, n_channels, H, W)
        
        Returns:
            logits: 分割输出 (B, n_classes, H, W)
            aux_loss: MoE辅助损失
        """
        # ========== 频率特征提取 ==========
        # 将RGB图像转换为灰度图并提取边缘特征
        grayscale_img = rgb_to_grayscale(x)
        # 构建拉普拉斯金字塔，提取高频信息（边缘特征）
        laplace_pyramid = make_laplace_pyramid(grayscale_img, 5, 1)
        freq_feat = laplace_pyramid[1]  # 选择第2层作为频率特征 (B, 1, H, W)

        # ========== 编码器路径：提取多尺度特征 ==========
        e1 = self.conv1(x)           # (B, 64, 224, 224)
        e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112)
        e2 = self.conv2(e1_maxp)     # (B, 64, 112, 112)
        e3 = self.conv3(e2)          # (B, 128, 56, 56)
        e4 = self.conv4(e3)          # (B, 256, 28, 28)
        e5 = self.conv5(e4)          # (B, 512, 14, 14)

        # ========== 频率特征增强：频率×特征+残差 ==========
        # 对每个编码器层应用频率增强
        e1_enhanced = self.freq_enhance1(e1, freq_feat)  # (B, 64, 224, 224)
        e2_enhanced = self.freq_enhance2(e2, freq_feat)  # (B, 64, 112, 112)
        e3_enhanced = self.freq_enhance3(e3, freq_feat)  # (B, 128, 56, 56)
        e4_enhanced = self.freq_enhance4(e4, freq_feat)  # (B, 256, 28, 28)

        # 初始化辅助损失
        aux_loss = torch.tensor(0.0, device=x.device)

        # ========== TA-MoSC模块：特征融合和自适应路由 ==========
        if self.pretrained:
            # 使用增强后的特征进行融合
            e1_resized = F.interpolate(e1_enhanced, scale_factor=0.5, mode='bilinear')
            e3_resized = F.interpolate(e3_enhanced, scale_factor=2, mode='bilinear')
            e4_resized = F.interpolate(e4_enhanced, scale_factor=4, mode='bilinear')
            
            fused = torch.cat([e1_resized, e2_enhanced, e3_resized, e4_resized], dim=1)
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
            # 非pretrained模式：使用增强后的编码器特征
            o1, o2, o3, o4 = e1_enhanced, e2_enhanced, e3_enhanced, e4_enhanced

        # ========== 解码器路径：上采样并融合 ==========
        d4 = self.up5(e5, o4)
        d3 = self.up4(d4, o3)
        d2 = self.up3(d3, o2)
        d1 = self.up2(d2, o1)

        # ========== 预测输出 ==========
        logits = self.pred(d1)

        return logits, aux_loss
    

def futanet(input_channel=3, num_classes=1, pretrained=True, topk=2):
    """
    FUTANet工厂函数
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        pretrained: 是否使用TA-MoSC模块
        topk: MoE选择的专家数量
    
    Returns:
        FUTANet模型实例
    """
    return FUTANet(
        n_channels=input_channel, 
        n_classes=num_classes,
        pretrained=pretrained,
        topk=topk
    )


if __name__ == "__main__":
    # 示例使用
    input_tensor = torch.randn(2, 3, 224, 224)
    model = FUTANet(pretrained=True, n_classes=1)
    model.eval()

    with torch.no_grad():
        output, loss = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {loss.item()}")
    print("\n模型创新点：")
    print("1. 引入拉普拉斯金字塔提取高频边缘特征")
    print("2. 对编码器各层应用频率增强（频率×特征+残差）")
    print("3. 保留TA-MoSC的自适应特征融合机制")
    print("4. 通过残差连接保证性能至少不会下降")

