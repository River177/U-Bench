"""
UTANet + ESTF: UTANet with Enhanced Spatial-Temporal Fusion Module
在编码器和解码器之间添加ESTF模块（LongDistanceDependencyModule）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import math
import re
from .ta_mosc import MoE
from timm.models.layers import DropPath, trunc_normal_, Mlp


# ========== ESTF模块相关组件 ==========
class S_GCN(nn.Module):
    """空间图卷积网络 - 动态适应输入尺寸的版本"""
    def __init__(self, channel):
        super(S_GCN, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        # 不再在初始化时创建固定尺寸的参数，而是在forward中动态创建或使用缓存

    def forward(self, x):
        b, c, H, W = x.size()
        n = H * W
        
        # 动态创建或获取参数
        para_key = f'para_{H}_{W}'
        adj_key = f'adj_{H}_{W}'
        
        # 使用buffer而不是parameter，以便能够动态调整
        # 确保buffer在正确的设备上（与输入x相同的设备）
        if not hasattr(self, para_key):
            para_buffer = torch.ones((1, c, H, W), dtype=torch.float32, device=x.device)
            self.register_buffer(para_key, para_buffer)
        if not hasattr(self, adj_key):
            adj_buffer = torch.ones((n, n), dtype=torch.float32, device=x.device)
            self.register_buffer(adj_key, adj_buffer)
        
        para = getattr(self, para_key).to(x.device)
        adj = getattr(self, adj_key).to(x.device)
        
        fea_matrix = x.view(b, c, n)
        c_adj = torch.mean(fea_matrix, dim=1)
        m = torch.zeros((b, c, H, W), dtype=torch.float32, device=x.device)
        
        for i in range(b):
            t1 = c_adj[i].unsqueeze(0)
            t2 = t1.t()
            
            c_adj_ = torch.abs(torch.abs(torch.sigmoid(t1 - t2) - 0.5) - 0.5) * 2
            c_adj_s = (c_adj_.t() + c_adj_) / 2
            output0 = torch.mul(torch.mm(fea_matrix[i], adj * c_adj_s).view(1, c, H, W), para)
            m[i] = output0

        output = torch.nn.functional.relu(m)
        return output


class Attention(nn.Module):
    """自注意力机制 - 动态适应输入尺寸的版本"""
    def __init__(self, dim, sa_num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., expand_ratio=2):
        super().__init__()
        self.dim = dim
        self.sa_num_heads = sa_num_heads
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."
        self.bn = nn.BatchNorm2d(dim*expand_ratio)
        head_dim = dim // sa_num_heads
        self.scale = qk_scale or (1+1e-6) / (math.sqrt(head_dim)+1e-6)
        self.q_sgcn = S_GCN(dim)  # 移除H, W参数
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_q = x.view(B, H, W, C).permute(0, 3, 1, 2)
        q_sgcn = self.q_sgcn(x_q).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        q_gcn = q_sgcn
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q_gcn @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class TransBlock(nn.Module):
    """Transformer块 - 动态适应输入尺寸的版本"""
    def __init__(self, dim, sa_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, expand_ratio=expand_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class LongDistanceDependencyModule_onlytrans(nn.Module):
    """ESTF模块：长距离依赖模块（Enhanced Spatial-Temporal Fusion）- 完整版本"""
    def __init__(self, in_channels, num_heads=4, hidden_dim=256):
        super(LongDistanceDependencyModule_onlytrans, self).__init__()
        self.transformer = TransBlock(dim=in_channels, sa_num_heads=num_heads)
        self.s_gcn = S_GCN(in_channels)
        self.fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x_trans = self.transformer(x_seq)
        x_trans = x_trans.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        x_gcn = self.s_gcn(x)

        x_fused = torch.cat([x_trans, x_gcn], dim=1)  # (B, 2*C, H, W)
        x_out = self.fusion(x_fused)  # (B, C, H, W)
        return x_out


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
        # 确保 skip_feat 和 up_feat 的空间尺寸匹配
        if skip_feat.shape[2:] != up_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat,
                size=up_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        return self.conv(fused_feat)


class UTANet_ESTF(nn.Module):
    """
    UTANet + ESTF: 在编码器和解码器之间添加ESTF模块
    
    架构：
    - 编码器：ResNet34提取多尺度特征
    - Bottleneck：ESTF模块处理最深层特征
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

        # ========== Bottleneck：ESTF模块 ==========
        # ESTF模块会自动适应输入特征图的尺寸
        self.estf = LongDistanceDependencyModule_onlytrans(self.filters_resnet[4])

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

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict以过滤动态创建的buffer"""
        # 过滤掉动态创建的buffer（para_{H}_{W} 和 adj_{H}_{W}）
        # 这些buffer是在forward中动态创建的，不应该从checkpoint中加载
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # 检查是否是动态创建的buffer（格式：para_数字_数字 或 adj_数字_数字）
            last_key = k.split('.')[-1]
            if re.match(r'^(para|adj)_\d+_\d+$', last_key):
                continue  # 跳过动态创建的buffer
            filtered_state_dict[k] = v
        # 调用父类方法加载过滤后的state_dict
        return super().load_state_dict(filtered_state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ========== 编码器路径 ==========
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        # ========== Bottleneck：ESTF处理 ==========
        e5_enhanced = self.estf(e5)  # 通过ESTF模块增强特征

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

        # ========== 解码器路径（使用ESTF增强的e5） ==========
        d4 = self.up5(e5_enhanced, o4)  # 使用ESTF增强后的特征
        d3 = self.up4(d4, o3)
        d2 = self.up3(d3, o2)
        d1 = self.up2(d2, o1)

        # ========== 预测输出 ==========
        logits = self.pred(d1)
        # 训练框架只接收单一输出张量，这里仅返回 logits
        return logits


def utanet_estf(input_channel=3, num_classes=1):
    return UTANet_ESTF(n_channels=input_channel, n_classes=num_classes)


if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 224, 224)
    model = UTANet_ESTF(pretrained=True, n_classes=1)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("UTANet_ESTF模型创建成功！")

