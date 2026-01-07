"""
SegFormerUTANet: UTANet with SegFormer-B4 as Encoder

This model combines SegFormer-B4 encoder with UTANet's TA-MoSC module for enhanced medical image segmentation.
The encoder part is replaced with SegFormer-B4 while keeping the decoder and TA-MoSC modules unchanged.

Usage:
    from segformer_utanet import SegFormerUTANet
    model = SegFormerUTANet(pretrained=True, n_classes=1)
    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .ta_mosc import MoE  # 从ta_mosc模块导入MoE（专家混合）类

# 使用项目中的自定义SegFormer实现（不依赖timm）
try:
    # 使用相对导入从DAEFormer模块导入MiT
    import sys
    import os
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    daeformer_segformer_path = os.path.join(project_root, 'models', 'Hybrid', 'DAEFormer', 'segformer.py')
    
    if os.path.exists(daeformer_segformer_path):
        # 动态导入segformer模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("segformer", daeformer_segformer_path)
        segformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(segformer_module)
        MiT = segformer_module.MiT
        USE_CUSTOM_SEGFORMER = True
        print("✓ 成功导入自定义SegFormer实现 (MiT)")
    else:
        raise FileNotFoundError(f"找不到segformer.py: {daeformer_segformer_path}")
except Exception as e:
    USE_CUSTOM_SEGFORMER = False
    # 如果导入失败，尝试使用timm作为备选
    try:
        import timm
        print(f"警告: 无法导入自定义SegFormer ({e})，将尝试使用timm库")
    except ImportError:
        raise ImportError(f"无法导入SegFormer实现: {e}\n请确保DAEFormer/segformer.py存在或安装timm库")


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution and skip connections"""
    def __init__(
        self, 
        in_ch: int, 
        skip_ch: int, 
        out_ch: int, 
        img_size: int = None, 
        scale_factor: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            in_ch: Number of input channels from decoder path
            skip_ch: Number of channels from encoder skip connection
            out_ch: Number of output channels
            img_size: Input image size (unused, kept for compatibility)
            scale_factor: Upsampling factor (default: 2x2)
        """
        super().__init__()
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
            skip_feat: Encoder skip connection feature map (B, skip_ch, H_skip, W_skip)
        
        Returns:
            Fused and upsampled feature map (B, out_ch, 2H, 2W)
        """
        up_feat = self.up(decoder_feat)  # (B, in_ch//2, 2H, 2W)
        
        # 确保 skip_feat 的尺寸与 up_feat 匹配
        if skip_feat.shape[2:] != up_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat, 
                size=up_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        return self.conv(fused_feat)


class SegFormerUTANet(nn.Module):
    """
    SegFormerUTANet: UTANet with SegFormer-B4 Encoder
    
    主要改进：
    1. 使用SegFormer-B4作为编码器提取多尺度特征（替代ResNet34）
    2. 保持TA-MoSC（Task-Adaptive Mixture of Skip Connections）模块
    3. 通过MoE机制自适应地选择和融合不同尺度的跳跃连接特征
    4. 采用两阶段训练策略：先训练基础UNet，再训练TA-MoSC模块
    
    SegFormer-B4特征：
    - Stage 1: H/4 × W/4, 64通道
    - Stage 2: H/8 × W/8, 128通道
    - Stage 3: H/16 × W/16, 320通道
    - Stage 4: H/32 × W/32, 512通道
    """
    def __init__(
        self, 
        pretrained_encoder: bool = True,
        pretrained: bool = True, 
        topk: int = 2, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        img_size: int = 224
    ):
        """
        初始化SegFormerUTANet模型
        
        Args:
            pretrained_encoder: 是否使用预训练的SegFormer-B4权重
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

        # ========== 编码器部分：基于SegFormer-B4 ==========
        # 优先使用项目中的自定义SegFormer实现（MiT）
        # SegFormer-B4配置：[64, 128, 320, 512]通道，[3, 8, 27, 3]层数
        if USE_CUSTOM_SEGFORMER:
            # 使用自定义的MiT实现（来自DAEFormer/segformer.py）
            segformer_b4_dims = [64, 128, 320, 512]
            segformer_b4_layers = [3, 8, 27, 3]
            self.segformer = MiT(
                image_size=img_size,
                dims=segformer_b4_dims,
                layers=segformer_b4_layers,
                token_mlp="mix_skip"
            )
            print(f"✓ 使用自定义SegFormer-B4编码器 (MiT)")
            # 注意：自定义实现不支持预训练权重加载，pretrained_encoder参数会被忽略
            if pretrained_encoder:
                print("  注意: 自定义SegFormer实现暂不支持预训练权重，将使用随机初始化")
        else:
            # 备选方案：使用timm库
            segformer_model_names = [
                'mit_b4',           # Mix Transformer B4
                'segformer_b4',     # SegFormer-B4
                'mit_b3',           # 降级选项
            ]
            
            self.segformer = None
            last_error = None
            
            for model_name in segformer_model_names:
                try:
                    self.segformer = timm.create_model(
                        model_name, 
                        pretrained=pretrained_encoder,
                        features_only=True,
                        out_indices=(0, 1, 2, 3)
                    )
                    print(f"✓ 使用timm库加载SegFormer编码器: {model_name}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if self.segformer is None:
                error_msg = (
                    f"无法加载SegFormer-B4模型。尝试的模型名称: {segformer_model_names}\n"
                    f"最后错误: {last_error}\n"
                    f"请尝试: pip install --upgrade timm>=0.6.0"
                )
                raise RuntimeError(error_msg)
        
        # SegFormer-B4各层的输出通道数：[64, 128, 320, 512]
        # 对应下采样倍率：[4, 8, 16, 32]
        # 注意：mit_b3的通道数可能不同，但这里假设使用B4
        self.filters_encoder = [64, 128, 320, 512]
        
        # 解码器各层的输出通道数（需要调整以匹配SegFormer的通道数）
        self.filters_decoder = [64, 128, 256, 512]

        # 通道适配层：将SegFormer的输出通道调整为与原UTANet兼容的通道数
        # 这样可以更好地与解码器配合
        self.adapt1 = nn.Sequential(
            nn.Conv2d(self.filters_encoder[0], 64, 1, 1),  # 64 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.adapt2 = nn.Sequential(
            nn.Conv2d(self.filters_encoder[1], 64, 1, 1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.adapt3 = nn.Sequential(
            nn.Conv2d(self.filters_encoder[2], 128, 1, 1),  # 320 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.adapt4 = nn.Sequential(
            nn.Conv2d(self.filters_encoder[3], 256, 1, 1),  # 512 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 调整后的编码器通道数（用于TA-MoSC和解码器）
        self.filters_adapted = [64, 64, 128, 256]

        # ========== TA-MoSC模块：特征融合和路由（仅在pretrained=True时使用）==========
        if pretrained:
            # 特征融合模块：将多尺度特征融合为统一维度
            # 输入：拼接后的384维特征（64+64+128+256）
            # 输出：64维统一特征表示
            fusion_channels = sum(self.filters_adapted)  # 64+64+128+256 = 512
            self.fuse = nn.Sequential(
                nn.Conv2d(fusion_channels, 64, 1, 1),  # 512 -> 64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # MoE（专家混合）模块：4个专家，每次选择topk个专家
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            # 特征路由器（Docker）：将MoE输出的64维特征路由到不同尺度
            self.docker1 = self._create_docker(64, self.filters_adapted[0])  # 64 -> 64
            self.docker2 = self._create_docker(64, self.filters_adapted[1])  # 64 -> 64
            self.docker3 = self._create_docker(64, self.filters_adapted[2])  # 64 -> 128
            self.docker4 = self._create_docker(64, self.filters_adapted[3])  # 64 -> 256

        # ========== 解码器部分：上采样和特征融合 ==========
        # 由于SegFormer输出4个尺度，我们需要调整解码器结构
        # up4: 从最深层开始，融合e4(H/32)特征
        self.up4 = UpBlock(512, self.filters_adapted[3], self.filters_decoder[2])  # 512+256 -> 256
        # up3: 融合d3和o3特征
        self.up3 = UpBlock(self.filters_decoder[2], self.filters_adapted[2], self.filters_decoder[1])  # 256+128 -> 128
        # up2: 融合d2和o2特征
        self.up2 = UpBlock(self.filters_decoder[1], self.filters_adapted[1], self.filters_decoder[0])  # 128+64 -> 64
        # up1: 融合d1和o1特征，上采样到原始分辨率
        self.up1 = UpBlock(self.filters_decoder[0], self.filters_adapted[0], self.filters_decoder[0])  # 64+64 -> 64

        # ========== 预测头：将特征图转换为分割掩码 ==========
        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0]//2, 1),  # 64 -> 32
            nn.BatchNorm2d(self.filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0]//2, n_classes, 1)  # 32 -> n_classes
        )
        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        创建特征路由器模块（Docker）
        
        Args:
            in_ch: 输入通道数（MoE输出，固定为64）
            out_ch: 输出通道数
        
        Returns:
            特征路由器模块
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播流程：
        1. SegFormer编码器提取多尺度特征（e1-e4）
        2. 通道适配层调整特征通道数
        3. 如果pretrained=True：特征融合 -> MoE路由 -> 特征分发到不同尺度
        4. 解码器上采样并融合编码器特征（跳跃连接）
        5. 输出分割掩码

        Args:
            x: 输入图像张量 (B, n_channels, H, W)
        
        Returns:
            logits: 分割输出logits (B, n_classes, H, W)
        """
        B, C, H, W = x.shape
        
        # ========== 编码器路径：SegFormer多尺度特征提取 ==========
        # SegFormer输出4个尺度的特征
        features = self.segformer(x)  # 返回list: [e1, e2, e3, e4]
        
        # 提取各层特征
        # e1: (B, 64, H/4, W/4)   - Stage 1输出
        # e2: (B, 128, H/8, W/8)  - Stage 2输出
        # e3: (B, 320, H/16, W/16) - Stage 3输出
        # e4: (B, 512, H/32, W/32) - Stage 4输出（最深层）
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]
        
        # 通道适配：将SegFormer特征调整为适合解码器的通道数
        e1 = self.adapt1(e1)  # (B, 64, H/4, W/4)
        e2 = self.adapt2(e2)  # (B, 64, H/8, W/8)
        e3 = self.adapt3(e3)  # (B, 128, H/16, W/16)
        e4 = self.adapt4(e4)  # (B, 256, H/32, W/32)
        
        # 初始化辅助损失
        aux_loss = torch.tensor(0.0, device=x.device)

        # ========== TA-MoSC模块：特征融合和自适应路由 ==========
        if self.pretrained:
            # 获取实际的特征尺寸
            e1_h, e1_w = e1.shape[2:]
            e2_h, e2_w = e2.shape[2:]
            e3_h, e3_w = e3.shape[2:]
            e4_h, e4_w = e4.shape[2:]
            
            # 确定融合的目标尺寸（使用e2的尺寸作为中间尺寸）
            target_size = (e2_h, e2_w)
            
            # 步骤1：将不同尺度的特征调整到统一尺寸以便融合
            e1_resized = F.interpolate(e1, size=target_size, mode='bilinear', align_corners=False)
            e2_resized = e2  # 已经是目标尺寸
            e3_resized = F.interpolate(e3, size=target_size, mode='bilinear', align_corners=False)
            e4_resized = F.interpolate(e4, size=target_size, mode='bilinear', align_corners=False)
            
            # 步骤2：特征融合 - 在通道维度拼接多尺度特征
            fused = torch.cat([e1_resized, e2_resized, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)  # 降维到64维统一表示
            
            # 步骤3：MoE路由 - 通过4个门控机制选择专家并生成4个输出
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss
            
            # 步骤4：特征路由 - 将MoE输出分发到不同尺度
            o1 = self.docker1(o1)  # (B, 64, e2_h, e2_w)
            o2 = self.docker2(o2)  # (B, 64, e2_h, e2_w)
            o3 = self.docker3(o3)  # (B, 128, e2_h, e2_w)
            o4 = self.docker4(o4)  # (B, 256, e2_h, e2_w)
            
            # 步骤5：调整空间尺寸以匹配解码器的跳跃连接（使用实际尺寸）
            o1 = F.interpolate(o1, size=(e1_h, e1_w), mode='bilinear', align_corners=False)
            o2 = F.interpolate(o2, size=(e2_h, e2_w), mode='bilinear', align_corners=False)  # 保持原尺寸
            o3 = F.interpolate(o3, size=(e3_h, e3_w), mode='bilinear', align_corners=False)
            o4 = F.interpolate(o4, size=(e4_h, e4_w), mode='bilinear', align_corners=False)
        else:
            # 非pretrained模式：直接使用编码器特征
            o1, o2, o3, o4 = e1, e2, e3, e4

        # ========== 解码器路径：逐步上采样并融合编码器特征 ==========
        # 使用实际的特征尺寸（而不是假设的 H//32 等）
        # 创建一个512通道的深层特征（用于up4的输入）
        # 将e4通道数从256扩展到512
        deep_feat = torch.cat([e4, e4], dim=1)  # (B, 512, e4_h, e4_w)
        
        # 解码器上采样路径（UpBlock内部会自动处理尺寸对齐）
        # up4: deep_feat (B,512,e4_h,e4_w) 上采样2倍 -> (B,256,2*e4_h,2*e4_w)，然后与o4融合
        d3 = self.up4(deep_feat, o4)  # UpBlock会自动对齐o4的尺寸
        d2 = self.up3(d3, o3)          # UpBlock会自动对齐o3的尺寸
        d1 = self.up2(d2, o2)          # UpBlock会自动对齐o2的尺寸
        d0 = self.up1(d1, o1)          # UpBlock会自动对齐o1的尺寸
        
        # 最后上采样到原始分辨率
        d0 = F.interpolate(d0, size=(H, W), mode='bilinear', align_corners=False)

        # ========== 预测输出：将特征图转换为分割掩码 ==========
        logits = self.pred(d0)  # (B, 64, H, W) -> (B, n_classes, H, W)
        
        return logits


def segformer_utanet(input_channel=3, num_classes=1, pretrained=True, pretrained_encoder=True):
    """
    创建SegFormerUTANet模型的工厂函数
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        pretrained: 是否使用TA-MoSC模块
        pretrained_encoder: 是否使用预训练的SegFormer-B4权重
    
    Returns:
        SegFormerUTANet模型实例
    """
    return SegFormerUTANet(
        pretrained_encoder=pretrained_encoder,
        pretrained=pretrained,
        n_channels=input_channel,
        n_classes=num_classes
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("SegFormerUTANet Model Test")
    print("=" * 60)
    
    try:
        # 创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {device}")
        
        # 测试不同配置
        for pretrained in [False, True]:
            print(f"\n{'='*60}")
            print(f"测试模式: pretrained={'启用TA-MoSC' if pretrained else '基础UNet'}")
            print(f"{'='*60}")
            
            model = SegFormerUTANet(
                pretrained_encoder=False,  # 测试时不下载预训练权重
                pretrained=pretrained,
                n_classes=1,
                img_size=224
            ).to(device)
            model.eval()
            
            # 生成随机输入
            input_tensor = torch.randn(2, 3, 224, 224).to(device)
            
            # 前向传播
            with torch.no_grad():
                output = model(input_tensor)
            
            # 打印结果
            print(f"输入形状: {input_tensor.shape}")
            print(f"输出形状: {output.shape}")
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            
            # 验证输出形状
            assert output.shape == (2, 1, 224, 224), f"输出形状错误: {output.shape}"
            print("✓ 模型测试通过!")
        
        print(f"\n{'='*60}")
        print("所有测试完成!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

