"""
UTANet with ResNet50 Encoder: Task-Adaptive Mixture of Skip Connections for Enhanced Medical Image Segmentation

基于原始UTANet，将编码器从ResNet34升级到ResNet50

主要改进：
    - ResNet50使用Bottleneck块，具有更强的特征表示能力
    - 通道数：[64, 256, 512, 1024, 2048]（比ResNet34的[64, 64, 128, 256, 512]更多）
    - 参数量：~25M（ResNet34约21M）
    - 在ImageNet上性能更优

Reference Paper: https://ojs.aaai.org/index.php/AAAI/article/view/32627

Usage:
    from UTANet_ResNet50 import UTANetResNet50
    
    # 使用ResNet50编码器
    model = UTANetResNet50(pretrained=True, n_classes=1)
    
    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
from .ta_mosc import MoE  # 从ta_mosc模块导入MoE（专家混合）类


class Flatten(nn.Module):
    """Flatten a tensor into a 2D matrix (batch_size, feature_dim)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Flattened tensor (B, C*H*W)
        """
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
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size (default: 3 with padding 1)
            scale_factor: Upsampling factor (default: 2x2)
        """
        super().__init__()
        self.padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, n_patches, hidden_dim)
        
        Returns:
            Reconstructed feature map (B, out_channels, H, W)
        """
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
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
        """
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            Downsampled feature map (B, out_ch, H/2, W/2)
        """
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
        """
        Args:
            in_ch: Number of input channels from decoder path
            skip_ch: Number of channels from encoder skip connection
            out_ch: Number of output channels
            img_size: Input image size (used for default scale factor)
            scale_factor: Upsampling factor (default: calculated based on img_size)
        """
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
        """
        Args:
            decoder_feat: Decoder feature map (B, in_ch, H, W)
            skip_feat: Encoder skip connection feature map (B, skip_ch, H, W)
        
        Returns:
            Fused and upsampled feature map (B, out_ch, 2H, 2W)
        """
        up_feat = self.up(decoder_feat)
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        return self.conv(fused_feat)


class UTANetResNet50(nn.Module):
    """
    UTANet with ResNet50 Encoder: U型Transformer注意力网络，结合专家混合模型(Mixture of Experts)
    
    这是一个用于医学图像分割的U型网络架构，主要创新点在于：
    1. 使用ResNet50作为编码器提取多尺度特征（相比ResNet34更强）
    2. 引入TA-MoSC（Task-Adaptive Mixture of Skip Connections）模块
    3. 通过MoE机制自适应地选择和融合不同尺度的跳跃连接特征
    4. 采用两阶段训练策略：先训练基础UNet，再训练TA-MoSC模块
    
    ResNet50 vs ResNet34:
    - ResNet34: BasicBlock, 通道数 [64, 64, 128, 256, 512]
    - ResNet50: Bottleneck, 通道数 [64, 256, 512, 1024, 2048]
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
        初始化UTANetResNet50模型
        
        Args:
            pretrained: 训练阶段标志
                    False (阶段1): 训练原始UNet模型（编码器和解码器）
                    True  (阶段2): 针对性地训练TA-MoSC模块（特征融合和路由）
            topk: MoE模块中选择的专家数量（默认2，即每次选择top-2专家）
            n_channels: 输入图像通道数（默认3，RGB图像）
            n_classes: 输出类别数（默认1，二值分割）
            img_size: 输入图像尺寸（默认224x224）
        """
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained  # 控制是否使用TA-MoSC模块
        self.img_size = img_size

        # ========== 编码器部分：基于ResNet50 ==========
        # 加载预训练的ResNet50作为特征提取器
        print("加载预训练的ResNet50模型...")
        try:
            # 新版本 torchvision (0.13+) 使用 weights 参数
            from torchvision.models import ResNet50_Weights
            self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # 旧版本 torchvision 使用 pretrained 参数
            self.resnet = models.resnet50(pretrained=True)
        
        # ResNet50各层的输出通道数：[conv1, layer1, layer2, layer3, layer4]
        # 注意：ResNet50使用Bottleneck块，通道数扩展4倍
        # layer1: 64 -> 256 (Bottleneck: 64->64->256)
        # layer2: 256 -> 512
        # layer3: 512 -> 1024
        # layer4: 1024 -> 2048
        self.filters_resnet = [64, 256, 512, 1024, 2048]
        
        # 解码器各层的输出通道数（逐步减少，最终输出32维特征）
        self.filters_decoder = [32, 64, 128, 256, 512]

        # 自定义第一层卷积（支持任意输入通道数，如灰度图、多光谱图像等）
        # 使用3x3卷积，padding=1保持空间尺寸不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),  # 批归一化，加速训练
            nn.ReLU(inplace=True)  # 激活函数，inplace=True节省内存
        )
        # 最大池化层，将特征图尺寸减半（224x224 -> 112x112）
        self.maxpool = nn.MaxPool2d(2, 2)

        # 提取ResNet的各个残差块层
        # 这些层会逐步下采样并增加通道数
        self.conv2 = self.resnet.layer1   # 输出: (B, 256, 112, 112) - 3个Bottleneck块
        self.conv3 = self.resnet.layer2   # 输出: (B, 512, 56, 56)  - 4个Bottleneck块
        self.conv4 = self.resnet.layer3   # 输出: (B, 1024, 28, 28) - 6个Bottleneck块
        self.conv5 = self.resnet.layer4   # 输出: (B, 2048, 14, 14) - 3个Bottleneck块

        # ========== TA-MoSC模块：特征融合和路由（仅在pretrained=True时使用）==========
        if pretrained:
            # 特征融合模块：将多尺度特征融合为统一维度
            # 输入：拼接后的1856维特征（64+256+512+1024）
            # 输出：64维统一特征表示
            self.fuse = nn.Sequential(
                nn.Conv2d(64+256+512+1024, 64, 1, 1),  # 1x1卷积，降维到64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # MoE（专家混合）模块：4个专家，每次选择topk个专家
            # 每个专家通过4个门控机制（gate1-4）进行路由
            # emb_size=64：特征嵌入维度
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            # 特征路由器（Docker）：将MoE输出的64维特征路由到不同尺度
            # 每个docker负责将64维特征转换为对应编码器层的通道数
            self.docker1 = self._create_docker(64, self.filters_resnet[0])  # 64 -> 64 (对应e1)
            self.docker2 = self._create_docker(64, self.filters_resnet[1])  # 64 -> 256 (对应e2)
            self.docker3 = self._create_docker(64, self.filters_resnet[2])  # 64 -> 512 (对应e3)
            self.docker4 = self._create_docker(64, self.filters_resnet[3])  # 64 -> 1024 (对应e4)

        # ========== 解码器部分：上采样和特征融合 ==========
        # 每个UpBlock包含：转置卷积上采样 + 跳跃连接融合 + 卷积细化
        # up5: 从最深层开始，融合e5和o4特征
        self.up5 = UpBlock(self.filters_resnet[4], self.filters_resnet[3], self.filters_decoder[4], 28)
        # up4: 融合d4和o3特征
        self.up4 = UpBlock(self.filters_decoder[4], self.filters_resnet[2], self.filters_decoder[3], 56)
        # up3: 融合d3和o2特征
        self.up3 = UpBlock(self.filters_decoder[3], self.filters_resnet[1], self.filters_decoder[2], 112)
        # up2: 融合d2和o1特征，输出最终32维特征
        self.up2 = UpBlock(self.filters_decoder[2], self.filters_resnet[0], self.filters_decoder[1], 224)

        # ========== 预测头：将特征图转换为分割掩码 ==========
        # 两层1x1卷积，逐步降维到类别数
        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_decoder[1], self.filters_decoder[1]//2, 1),  # 64 -> 32
            nn.BatchNorm2d(self.filters_decoder[1]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[1]//2, n_classes, 1)  # 32 -> n_classes
        )
        # 二值分割使用Sigmoid激活，多类分割直接输出logits
        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        创建特征路由器模块（Docker）
        
        Docker的作用是将MoE输出的统一特征表示（64维）转换为不同尺度编码器层所需的通道数。
        使用1x1卷积进行通道变换，不改变空间尺寸。
        
        Args:
            in_ch: 输入通道数（MoE输出，固定为64）
            out_ch: 输出通道数（对应编码器层的通道数：64/256/512/1024）
        
        Returns:
            特征路由器模块：1x1卷积 + 批归一化 + ReLU激活
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),  # 1x1卷积，仅改变通道数
            nn.BatchNorm2d(out_ch),  # 批归一化，稳定训练
            nn.ReLU(inplace=True)  # 非线性激活
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播流程：
        1. 编码器提取多尺度特征（e1-e5）
        2. 如果pretrained=True：特征融合 -> MoE路由 -> 特征分发到不同尺度
        3. 解码器上采样并融合编码器特征（跳跃连接）
        4. 输出分割掩码

        Args:
            x: 输入图像张量 (B, n_channels, H, W)
                B: batch size（批次大小）
                n_channels: 输入通道数（如3表示RGB）
                H, W: 图像高度和宽度（默认224x224）
        
        Returns:
            logits: 分割输出logits (B, n_classes, H, W)
        
        Note:
            MoE的辅助损失（aux_loss）在内部计算但不返回，以保持与训练框架的兼容性
        """
        # ========== 编码器路径：逐步下采样提取多尺度特征 ==========
        # e1: 第一层卷积输出，保持原始分辨率
        e1 = self.conv1(x)           # (B, 64, 224, 224) - 初始特征提取
        # 最大池化，尺寸减半
        e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112) - 下采样到112x112
        # e2-e5: ResNet50残差块，逐步下采样并增加通道数
        e2 = self.conv2(e1_maxp)     # (B, 256, 112, 112)  - ResNet50 layer1
        e3 = self.conv3(e2)          # (B, 512, 56, 56)    - ResNet50 layer2，下采样到56x56
        e4 = self.conv4(e3)          # (B, 1024, 28, 28)   - ResNet50 layer3，下采样到28x28
        e5 = self.conv5(e4)          # (B, 2048, 14, 14)   - ResNet50 layer4，下采样到14x14（最深层特征）

        # 初始化辅助损失（MoE的负载均衡损失）
        aux_loss = torch.tensor(0.0, device=x.device)

        # ========== TA-MoSC模块：特征融合和自适应路由 ==========
        if self.pretrained:
            # 步骤1：将不同尺度的特征调整到统一尺寸（112x112）以便融合
            # e1从224x224下采样到112x112
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear')   # (B, 64, 112, 112)
            # e2已经是112x112，不需要调整
            # e3从56x56上采样到112x112
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear')     # (B, 512, 112, 112)
            # e4从28x28上采样到112x112
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear')     # (B, 1024, 112, 112)
            
            # 步骤2：特征融合 - 在通道维度拼接多尺度特征
            # 拼接后：64+256+512+1024=1856维特征
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)  # (B, 1856, 112, 112)
            # 通过融合模块降维到64维统一表示
            fused = self.fuse(fused)                                           # (B, 64, 112, 112)
            
            # 步骤3：MoE路由 - 通过4个门控机制选择专家并生成4个输出
            # o1-o4: MoE输出的4个特征表示（每个都是64维，112x112）
            # loss: 负载均衡损失，鼓励均匀使用所有专家
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss  # 保存辅助损失用于反向传播

            # 步骤4：特征路由 - 将MoE输出分发到不同尺度
            # 通过Docker将64维特征转换为对应编码器层的通道数
            o1 = self.docker1(o1)     # (B, 64, 112, 112) -> (B, 64, 112, 112) 对应e1
            o2 = self.docker2(o2)     # (B, 64, 112, 112) -> (B, 256, 112, 112) 对应e2
            o3 = self.docker3(o3)     # (B, 64, 112, 112) -> (B, 512, 112, 112) 对应e3
            o4 = self.docker4(o4)     # (B, 64, 112, 112) -> (B, 1024, 112, 112) 对应e4
            
            # 步骤5：调整空间尺寸以匹配解码器的跳跃连接
            # o4需要下采样到28x28（对应e4的尺寸）
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear')  # (B, 1024, 28, 28)
            # o3需要下采样到56x56（对应e3的尺寸）
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear')   # (B, 512, 56, 56)
            # o1需要上采样回224x224（对应e1的尺寸）
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear')     # (B, 64, 224, 224)
        else:
            # 非pretrained模式：直接使用编码器特征，不使用TA-MoSC模块
            # 这是阶段1训练时的行为，训练基础的UNet结构
            o1, o2, o3, o4 = e1, e2, e3, e4

        # ========== 解码器路径：逐步上采样并融合编码器特征 ==========
        # 解码器通过跳跃连接融合编码器特征，逐步恢复空间分辨率
        # up5: 从最深层e5开始，融合o4特征，上采样到28x28
        d4 = self.up5(e5, o4)  # (B, 2048, 14, 14) + (B, 1024, 28, 28) -> (B, 512, 28, 28)
        # up4: 融合d4和o3，上采样到56x56
        d3 = self.up4(d4, o3)  # (B, 512, 28, 28) + (B, 512, 56, 56) -> (B, 256, 56, 56)
        # up3: 融合d3和o2，上采样到112x112
        d2 = self.up3(d3, o2)  # (B, 256, 56, 56) + (B, 256, 112, 112) -> (B, 128, 112, 112)
        # up2: 融合d2和o1，上采样回224x224，输出64维特征
        d1 = self.up2(d2, o1)  # (B, 128, 112, 112) + (B, 64, 224, 224) -> (B, 64, 224, 224)

        # ========== 预测输出：将特征图转换为分割掩码 ==========
        # 通过预测头将64维特征转换为类别数
        logits = self.pred(d1)       # (B, 64, 224, 224) -> (B, n_classes, 224, 224)
        
        # 训练框架只接收单一输出张量，这里仅返回 logits
        # aux_loss 在内部计算但不返回，以保持与训练框架的兼容性
        return logits
    

def utanet_resnet50(input_channel=3, num_classes=1):
    """
    创建UTANetResNet50模型的便捷函数
    
    Args:
        input_channel: 输入通道数（默认3）
        num_classes: 输出类别数（默认1）
    
    Returns:
        UTANetResNet50模型实例
    """
    return UTANetResNet50(n_channels=input_channel, n_classes=num_classes)


if __name__ == "__main__":
    # 示例用法
    print("=" * 80)
    print("UTANetResNet50 模型测试")
    print("=" * 80)
    
    input_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 channels, 224x224 images
    
    # 测试阶段1（不使用TA-MoSC）
    print("\n阶段1（pretrained=False）:")
    model_stage1 = UTANetResNet50(pretrained=False, n_classes=1)
    model_stage1.eval()
    
    with torch.no_grad():
        output1 = model_stage1(input_tensor)
    
    print(f"✓ 输入形状: {input_tensor.shape}")
    print(f"✓ 输出形状: {output1.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model_stage1.parameters())
    trainable_params = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ 可训练参数量: {trainable_params:,}")
    
    # 测试阶段2（使用TA-MoSC）
    print("\n阶段2（pretrained=True）:")
    model_stage2 = UTANetResNet50(pretrained=True, n_classes=1)
    model_stage2.eval()
    
    with torch.no_grad():
        output2 = model_stage2(input_tensor)
    
    print(f"✓ 输入形状: {input_tensor.shape}")
    print(f"✓ 输出形状: {output2.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model_stage2.parameters())
    trainable_params = sum(p.numel() for p in model_stage2.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ 可训练参数量: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    print("\nResNet50 vs ResNet34 对比:")
    print("- ResNet34: BasicBlock, 通道数 [64, 64, 128, 256, 512], ~21M参数")
    print("- ResNet50: Bottleneck, 通道数 [64, 256, 512, 1024, 2048], ~25M参数")
    print("- ResNet50在ImageNet上性能更优，适合追求更高精度的场景")

