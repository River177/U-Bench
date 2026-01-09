"""
UTANet: Task-Adaptive Mixture of Skip Connections for Enhanced Medical Image Segmentation

Reference Paper: https://ojs.aaai.org/index.php/AAAI/article/view/32627

Usage:
    from utanet import UTANet
    model = UTANet(pretrained=True, n_classes=1)
    inputs = torch.randn(2, 3, 224, 224)
    outputs, loss = model(inputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision is used for the ConvNeXt-Base backbone (ImageNet pretrained).
# If your environment has a torchvision binary mismatch (e.g., missing compiled ops),
# this import may fail. In that case, reinstall a matching torchvision build.
import torchvision.models as models
from typing import Optional, Tuple, List
from .ta_mosc import MoE  # Assuming MoE module is in ta_mosc.py


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


class UTANet(nn.Module):
    """U-shaped Transformer Attention Network with Mixture of Experts"""
    def __init__(
        self, 
        pretrained: bool = True, 
        topk: int = 2, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        img_size: int = 224
    ):
        """
        Args:
            pretrained: 
                    False (stage 1) : Training origin Unet model(Encoder and Decoder)
                    True  (stage 2) : Targeted training TA-MoSC module
            topk: Number of experts to select in MoE module
            n_channels: Number of input channels (default: 3 for RGB)
            n_classes: Number of output classes (default: 1 for binary segmentation)
            img_size: Input image size (default: 224x224)
        """
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size

        # Encoder backbone: ConvNeXt-Base (ImageNet pretrained)
        # Keep the same U-Net channel plan expected by the original decoder/MoE.
        # (e1/e2: 64ch, e3: 128ch, e4: 256ch, e5: 512ch)
        self.filters_resnet = [64, 64, 128, 256, 512]
        self.filters_decoder = [32, 64, 128, 256, 512]

        # ConvNeXt expects 3-channel input for pretrained weights. If your input
        # has a different channel count, we adapt it with a lightweight 1x1 conv.
        self.convnext_in = nn.Identity() if n_channels == 3 else nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)

        # ConvNeXt-Base backbone (pretrained on ImageNet-1K)
        self.convnext = self._create_convnext_base(pretrained=True)

        # Custom first convolution layer (supports arbitrary input channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        # Encoder stage-2 at 1/2 resolution (112x112): keep a small conv block to
        # match the original e2 interface (64ch @ 112x112).
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.filters_resnet[0], self.filters_resnet[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.filters_resnet[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_resnet[1], self.filters_resnet[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.filters_resnet[1]),
            nn.ReLU(inplace=True),
        )

        # MoE routing and feature fusion (only used in pretrained mode)
        if pretrained:
            # Feature fusion module
            self.fuse = nn.Sequential(
                nn.Conv2d(512, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # MoE module (4 experts, topk selection)
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            # Feature routers (Dokers)
            self.docker1 = self._create_docker(64, self.filters_resnet[0])
            self.docker2 = self._create_docker(64, self.filters_resnet[1])
            self.docker3 = self._create_docker(64, self.filters_resnet[2])
            self.docker4 = self._create_docker(64, self.filters_resnet[3])

        # Decoder modules
        self.up5 = UpBlock(self.filters_resnet[4], self.filters_resnet[3], self.filters_decoder[3], 28)
        self.up4 = UpBlock(self.filters_decoder[3], self.filters_resnet[2], self.filters_decoder[2], 56)
        self.up3 = UpBlock(self.filters_decoder[2], self.filters_resnet[1], self.filters_decoder[1], 112)
        self.up2 = UpBlock(self.filters_decoder[1], self.filters_resnet[0], self.filters_decoder[0], 224)

        # Output prediction head
        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0]//2, 1),
            nn.BatchNorm2d(self.filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0]//2, n_classes, 1)
        )
        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create a feature router module (1x1 convolution + BatchNorm + ReLU)"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _create_convnext_base(self, pretrained: bool = True) -> nn.Module:
        """Create a ConvNeXt-Base backbone and load ImageNet pretrained weights."""
        # Newer torchvision versions: use `weights=` enums; older versions still accept `pretrained=`.
        if hasattr(models, "ConvNeXt_Base_Weights"):
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            return models.convnext_base(weights=weights)
        # Fallback for older torchvision
        return models.convnext_base(pretrained=pretrained)

    @torch.no_grad()
    def _convnext_stage_shapes(self, x: torch.Tensor) -> List[Tuple[int, int]]:
        """(Optional) Debug helper to inspect ConvNeXt feature resolutions."""
        shapes: List[Tuple[int, int]] = []
        out = x
        for layer in self.convnext.features:
            out = layer(out)
            shapes.append((out.shape[-2], out.shape[-1]))
        return shapes

    def _forward_convnext_stages(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward ConvNeXt features and return stage outputs at progressively
        downsampled resolutions.

        Returns:
            s1: stage-1 output (~1/4 res, e.g., 56x56, 128ch)
            s2: stage-2 output (~1/8 res, e.g., 28x28, 256ch)
            s3: stage-3 output (~1/16 res, e.g., 14x14, 512ch)
        """
        out = x
        stage_outs: List[torch.Tensor] = []
        prev_hw = None
        prev_out = None

        for layer in self.convnext.features:
            out = layer(out)
            hw = out.shape[-2:]
            if prev_hw is None:
                prev_hw = hw
                prev_out = out
                continue
            # Resolution changed => downsample happened; save the previous resolution's final output.
            if hw != prev_hw:
                stage_outs.append(prev_out)
            prev_hw = hw
            prev_out = out

        # Append the last output (final stage)
        stage_outs.append(out)

        if len(stage_outs) < 3:
            raise RuntimeError(
                f"Unexpected ConvNeXt feature structure: got {len(stage_outs)} stages; "
                "cannot extract 3 stage outputs."
            )

        # For ConvNeXt-Base, stage_outs is typically [56, 28, 14, 7].
        return stage_outs[0], stage_outs[1], stage_outs[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass workflow:
        1. Encoder extracts multi-scale features
        2. In pretrained mode, features are fused and routed through MoE
        3. Decoder upsamples and fuses encoder features
        4. Output segmentation mask and auxiliary loss (from MoE)

        Args:
            x: Input image (B, n_channels, H, W)
        
        Returns:
            out: Segmentation output (B, n_classes, H, W)
            aux_loss: Auxiliary loss from MoE routing
        """
        # Encoder path
        e1 = self.conv1(x)           # (B, 64, 224, 224)
        e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112)
        e2 = self.conv2(e1_maxp)     # (B, 64, 112, 112)
        # ConvNeXt feature pyramid (base): stage outputs ~[1/4, 1/8, 1/16] resolution
        convnext_x = self.convnext_in(x)
        e3, e4, e5 = self._forward_convnext_stages(convnext_x)

        aux_loss = torch.tensor(0.0, device=x.device)

        if self.pretrained:
            # Resize features for fusion
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear')   # (B, 64, 112, 112)
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear')     # (B, 128, 112, 112)
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear')     # (B, 256, 112, 112)
            
            # Feature fusion
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)  # (B, 512, 112, 112)
            fused = self.fuse(fused)                                           # (B, 64, 112, 112)
            
            # MoE routing
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss

            # Route to respective scales
            o1 = self.docker1(o1)     # (B, 64, 112, 112)
            o2 = self.docker2(o2)     # (B, 64, 112, 112)
            o3 = self.docker3(o3)     # (B, 128, 56, 56)
            o4 = self.docker4(o4)     # (B, 256, 28, 28)
            
            # Resize to match decoder skip connections
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear')  # (B, 256, 7, 7)
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear')   # (B, 128, 14, 14)
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear')     # (B, 64, 224, 224)
        else:
            # Use encoder features directly in non-pretrained mode
            o1, o2, o3, o4 = e1, e2, e3, e4

        # Decoder path
        d4 = self.up5(e5, o4)  # (B, 256, 14, 14) -> (B, 256, 28, 28)
        d3 = self.up4(d4, o3)  # (B, 256, 28, 28) -> (B, 128, 56, 56)
        d2 = self.up3(d3, o2)  # (B, 128, 56, 56) -> (B, 64, 112, 112)
        d1 = self.up2(d2, o1)  # (B, 64, 112, 112) -> (B, 32, 224, 224)

        # Output prediction
        logits = self.pred(d1)       # (B, n_classes, 224, 224)
        # 训练框架只接收单一输出张量，这里仅返回 logits
        # aux_loss 在内部计算但不返回，以保持与训练框架的兼容性
        return logits
    
def utanet(input_channel=3,num_classes=1):
    return UTANet(n_channels=input_channel,n_classes=num_classes)


if __name__ == "__main__":
    # Example usage
    input_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 channels, 224x224 images
    model = UTANet(pretrained=True, n_classes=1)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("UTANet with ConvNeXt-Base backbone created successfully!")

def utanet_convnext(input_channel=3, num_classes=1, pretrained=True, img_size=224, topk=2, **kwargs):
    return UTANet(pretrained=pretrained, topk=topk, n_channels=input_channel, n_classes=num_classes, img_size=img_size)

# 兼容旧名字（可选）
def utanet(input_channel=3, num_classes=1, **kwargs):
    return utanet_convnext(input_channel=input_channel, num_classes=num_classes, **kwargs)
