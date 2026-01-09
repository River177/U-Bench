"""
Lightweight Hierarchical UTANet Modules
包含层次化MoE、轻量ASPP、深度可分离解码器等模块

核心思路：
- HierarchicalExpert: 不同感受野的专家网络
- HierarchicalMoE: 层次化专家混合，top-k路由
- LightweightASPP: 深度可分离卷积实现的ASPP
- LightweightUpBlock: 深度可分离卷积实现的上采样块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HierarchicalExpert(nn.Module):
    """
    层次化专家网络 - 每个专家专注于不同的感受野
    
    四种专家类型：
    1. small: 小感受野专家（关注细节，dilation=1）
    2. medium: 中感受野专家（平衡，dilation=2）
    3. large: 大感受野专家（关注上下文，dilation=4）
    4. global: 全局专家（全局池化）
    
    Args:
        emb_size: 特征嵌入维度
        scale: 专家类型 ['small', 'medium', 'large', 'global']
    """
    def __init__(self, emb_size: int, scale: str = 'small'):
        super().__init__()
        self.scale = scale
        
        if scale == 'small':
            # 小感受野专家（关注细节）- 标准深度可分离卷积
            self.conv = nn.Sequential(
                # 深度卷积 (Depthwise Convolution)
                nn.Conv2d(emb_size, emb_size*2, 3, padding=1, groups=emb_size),
                nn.BatchNorm2d(emb_size*2),
                nn.ReLU(inplace=True),
                # 逐点卷积 (Pointwise Convolution)
                nn.Conv2d(emb_size*2, emb_size, 1)
            )
        elif scale == 'medium':
            # 中感受野专家（平衡）- 空洞卷积 dilation=2
            self.conv = nn.Sequential(
                nn.Conv2d(emb_size, emb_size*2, 3, padding=2, dilation=2, groups=emb_size),
                nn.BatchNorm2d(emb_size*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(emb_size*2, emb_size, 1)
            )
        elif scale == 'large':
            # 大感受野专家（关注上下文）- 空洞卷积 dilation=4
            self.conv = nn.Sequential(
                nn.Conv2d(emb_size, emb_size*2, 3, padding=4, dilation=4, groups=emb_size),
                nn.BatchNorm2d(emb_size*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(emb_size*2, emb_size, 1)
            )
        else:  # 'global'
            # 全局专家（全局上下文）- 全局池化 + 1x1卷积
            self.conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(emb_size, emb_size*2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(emb_size*2, emb_size, 1),
                nn.Sigmoid()  # 生成注意力权重
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            处理后的特征 (B, C, H, W)
        """
        if self.scale == 'global':
            # 全局专家：生成注意力权重并应用
            attention = self.conv(x)  # (B, C, 1, 1)
            return x * attention  # 广播乘法
        else:
            return self.conv(x)


class HierarchicalMoE(nn.Module):
    """
    层次化专家混合模块
    
    核心改进：
    1. 4个专家对应4种感受野（small/medium/large/global）
    2. 轻量化门控：使用全局平均池化 + 1x1卷积
    3. Top-k专家选择，减少计算量
    4. 负载均衡损失，确保专家均匀使用
    
    Args:
        num_experts: 专家数量（默认4）
        top: top-k选择的k值（默认2）
        emb_size: 特征嵌入维度（默认64）
    """
    def __init__(self, num_experts: int = 4, top: int = 2, emb_size: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.top = top
        self.emb_size = emb_size
        
        # 创建4个不同感受野的专家
        scales = ['small', 'medium', 'large', 'global']
        self.experts = nn.ModuleList([
            HierarchicalExpert(emb_size, scale) for scale in scales
        ])
        
        # 轻量化门控网络：全局池化 + 1x1卷积
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(emb_size, num_experts, 1)  # 生成专家权重
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：选择top-k专家并加权聚合
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            output: 聚合后的输出 (B, C, H, W)
            balance_loss: 负载均衡损失
        """
        B, C, H, W = x.shape
        
        # === 步骤1：计算门控权重 ===
        gate_score = self.gate_conv(x).view(B, -1)  # (B, num_experts)
        gate_score = F.softmax(gate_score, dim=1)  # softmax归一化
        
        # === 步骤2：选择top-k专家 ===
        top_weights, top_idx = torch.topk(gate_score, self.top, dim=1)  # (B, top)
        top_weights = F.softmax(top_weights, dim=1)  # 重新归一化
        
        # === 步骤3：聚合专家输出 ===
        output = torch.zeros_like(x)
        for i in range(B):
            for j, weight in enumerate(top_weights[i]):
                expert_idx = top_idx[i, j]
                expert_out = self.experts[expert_idx](x[i:i+1])
                output[i:i+1] += weight * expert_out
        
        # === 步骤4：计算负载均衡损失 ===
        # 使用变异系数：CV² = Var(X) / Mean(X)²
        usage = gate_score.sum(0)  # 每个专家的总使用次数
        balance_loss = usage.var() / (usage.mean()**2 + 1e-10)
        
        return output, balance_loss


class LightweightASPP(nn.Module):
    """
    轻量空洞空间金字塔池化
    
    使用深度可分离卷积实现ASPP，大幅减少参数量和计算量。
    
    结构：
    1. 1x1卷积分支
    2. 3x3空洞卷积分支（dilation=6）
    3. 3x3空洞卷积分支（dilation=12）
    4. 全局池化分支
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 分支1: 1x1卷积（局部特征）
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支2: 深度可分离空洞卷积（dilation=6）
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=6, dilation=6, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支3: 深度可分离空洞卷积（dilation=12）
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=12, dilation=12, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支4: 全局池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        # 融合层：4个分支的特征拼接后降维
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：多尺度特征融合
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            融合后的多尺度特征 (B, out_channels, H, W)
        """
        size = x.shape[2:]
        
        # 4个分支并行处理
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)
        
        # 拼接并融合
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(out)


class LightweightUpBlock(nn.Module):
    """
    轻量化上采样块
    
    使用深度可分离卷积替代标准卷积，减少参数量。
    
    结构：
    1. 转置卷积上采样
    2. 跳跃连接拼接
    3. 深度可分离卷积融合
    
    Args:
        in_ch: 解码器输入通道数
        skip_ch: 跳跃连接通道数
        out_ch: 输出通道数
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        
        # 转置卷积上采样（2倍）
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        
        # 深度可分离卷积（融合跳跃连接）
        total_ch = in_ch//2 + skip_ch
        self.dw = nn.Conv2d(total_ch, total_ch, 3, padding=1, groups=total_ch)  # 深度卷积
        self.pw = nn.Conv2d(total_ch, out_ch, 1)  # 逐点卷积
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, dec_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播：上采样并融合跳跃连接
        
        Args:
            dec_feat: 解码器特征 (B, in_ch, H, W)
            skip_feat: 跳跃连接特征 (B, skip_ch, 2H, 2W)
            
        Returns:
            融合后的特征 (B, out_ch, 2H, 2W)
        """
        # 上采样
        up = self.up(dec_feat)
        
        # 确保 skip_feat 和 up 的空间尺寸匹配
        if skip_feat.shape[2:] != up.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat,
                size=up.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 拼接跳跃连接
        fused = torch.cat([skip_feat, up], dim=1)
        
        # 深度可分离卷积融合
        out = self.dw(fused)
        out = self.pw(out)
        out = self.relu(self.bn(out))
        
        return out


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Lightweight Modules")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # === Test 1: HierarchicalExpert ===
    print("\nTesting HierarchicalExpert")
    x = torch.randn(2, 64, 56, 56).to(device)
    
    for scale in ['small', 'medium', 'large', 'global']:
        expert = HierarchicalExpert(64, scale).to(device)
        out = expert(x)
        params = sum(p.numel() for p in expert.parameters())
        print(f"  {scale}: output shape={out.shape}, parameters={params:,}")
    
    # === Test 2: HierarchicalMoE ===
    print("\nTesting HierarchicalMoE")
    moe = HierarchicalMoE(num_experts=4, top=2, emb_size=64).to(device)
    out, loss = moe(x)
    params = sum(p.numel() for p in moe.parameters())
    print(f"  Output shape: {out.shape}")
    print(f"  Load balancing loss: {loss.item():.6f}")
    print(f"  Parameters: {params:,}")
    
    # === Test 3: LightweightASPP ===
    print("\nTesting LightweightASPP")
    x_aspp = torch.randn(2, 512, 14, 14).to(device)
    aspp = LightweightASPP(512, 512).to(device)
    out = aspp(x_aspp)
    params = sum(p.numel() for p in aspp.parameters())
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {params:,}")
    
    # === Test 4: LightweightUpBlock ===
    print("\nTesting LightweightUpBlock")
    dec_feat = torch.randn(2, 512, 14, 14).to(device)
    skip_feat = torch.randn(2, 256, 28, 28).to(device)
    upblock = LightweightUpBlock(512, 256, 256).to(device)
    out = upblock(dec_feat, skip_feat)
    params = sum(p.numel() for p in upblock.parameters())
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {params:,}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

