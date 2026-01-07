"""
UTANet + EAG: UTANet with Entropy-based Attention Gate Module
在编码器和解码器之间添加EAG模块（CoronaryIdentificationModule）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
from .ta_mosc import MoE


# ========== EAG模块相关组件 ==========
class ConvolutionalSelfAttention(nn.Module):
    """卷积自注意力模块"""
    def __init__(self, in_channels):
        super(ConvolutionalSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        attention = query * key
        attention = F.softmax(attention, dim=1)
        value = self.value_conv(x)
        out = value * attention
        return out + x


class EntropyBasedFusion(nn.Module):
    """基于熵的融合模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, high_freq_output, low_freq_output):
        norm_high = torch.sigmoid(high_freq_output)
        norm_low = torch.sigmoid(low_freq_output)
        entropy_high = -torch.sum(norm_high * torch.log(norm_high + 1e-8), dim=1, keepdim=True)
        entropy_low = -torch.sum(norm_low * torch.log(norm_low + 1e-8), dim=1, keepdim=True)

        w_h = 1 - entropy_high
        w_l = 1 - entropy_low
        w_total = w_h + w_l
        weight_high = w_h / (w_total + 1e-8)
        weight_low = w_l / (w_total + 1e-8)

        fused_output = weight_high * high_freq_output + weight_low * low_freq_output + high_freq_output + low_freq_output
        return self.conv(fused_output)


class DoubleConv(nn.Module):
    """双卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, kernel_size), padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CoronaryIdentificationModule(nn.Module):
    """EAG模块：冠状动脉识别模块（Entropy-based Attention Gate）"""
    def __init__(self, in_channels, out_channels):
        super(CoronaryIdentificationModule, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.high_freq_path = nn.Sequential(
            DoubleConv(self.in_channels, self.in_channels)
        )
        self.low_freq_path = nn.Sequential(
            ConvolutionalSelfAttention(self.in_channels)
        )
        self.fusion = EntropyBasedFusion(self.in_channels, self.out_channels)

    def forward(self, x):
        high_freq = self.high_freq_path(x)
        low_freq = self.low_freq_path(x)
        fused = self.fusion(high_freq, low_freq)
        return fused


# ========== UTANet原始组件 ==========
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


class UTANet_EAG(nn.Module):
    """
    UTANet + EAG: 在编码器和解码器之间添加EAG模块
    
    架构：
    - 编码器：ResNet34提取多尺度特征
    - Bottleneck：EAG模块处理最深层特征
    - 解码器：上采样并融合编码器特征
    """
    def __init__(
        self, 
        pretrained: bool = True, 
        topk: int = 2, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        img_size: int = 224
    ):
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

        # ========== Bottleneck：EAG模块 ==========
        self.eag = CoronaryIdentificationModule(self.filters_resnet[4], self.filters_resnet[4])

        # ========== TA-MoSC模块（可选） ==========
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
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ========== 编码器路径 ==========
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        # ========== Bottleneck：EAG处理 ==========
        e5_enhanced = self.eag(e5)  # 通过EAG模块增强特征

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

        # ========== 解码器路径（使用EAG增强的e5） ==========
        d4 = self.up5(e5_enhanced, o4)  # 使用EAG增强后的特征
        d3 = self.up4(d4, o3)
        d2 = self.up3(d3, o2)
        d1 = self.up2(d2, o1)

        # ========== 预测输出 ==========
        logits = self.pred(d1)
        # 训练框架只接收单一输出张量，这里仅返回 logits
        return logits


def utanet_eag(input_channel=3, num_classes=1):
    return UTANet_EAG(n_channels=input_channel, n_classes=num_classes)


if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 224, 224)
    model = UTANet_EAG(pretrained=True, n_classes=1)
    model.eval()

    with torch.no_grad():
        output, loss = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {loss.item()}")
    print("UTANet_EAG模型创建成功！")

