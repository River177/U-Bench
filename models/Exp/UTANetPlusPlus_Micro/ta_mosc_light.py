"""
轻量化 TA-MoSC 模块
简化版专家混合机制，减少专家数量和计算复杂度

改进：
1. 专家网络使用深度可分离卷积
2. 减少专家数量（默认2个）
3. 简化门控机制
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple


class LightExpert(nn.Module):
    """
    轻量化专家网络：使用深度可分离卷积
    参数量相比原版减少约 6-8 倍
    """
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        
        self.seq = nn.Sequential(
            # 深度卷积
            nn.Conv2d(emb_size, emb_size, 3, 1, 1, groups=emb_size, bias=False),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
            # 扩展
            nn.Conv2d(emb_size, hidden_emb, 1, bias=False),
            nn.BatchNorm2d(hidden_emb),
            nn.GELU(),
            # 压缩
            nn.Conv2d(hidden_emb, emb_size, 1, bias=False),
            nn.BatchNorm2d(emb_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MoELight(nn.Module):
    """
    轻量化专家混合模块
    
    改进：
    1. 减少专家数量（2-3个专家）
    2. 简化门控机制（共享门控）
    3. 使用轻量化专家网络
    
    Args:
        num_experts: 专家数量（默认2）
        top: top-k选择（默认1）
        emb_size: 特征嵌入维度
        num_gates: 门控数量（默认4）
    """
    def __init__(
        self, 
        num_experts: int = 2, 
        top: int = 1, 
        emb_size: int = 32,
        num_gates: int = 4
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_gates = num_gates
        self.top = top
        self.emb_size = emb_size
        
        # 轻量化专家网络
        self.experts = nn.ModuleList([
            LightExpert(emb_size, hidden_rate=2) 
            for _ in range(num_experts)
        ])
        
        # 共享门控网络（更轻量）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(emb_size, num_experts * num_gates),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            o1, o2, o3, o4: 4个门控输出
            loss: 负载均衡损失
        """
        B, C, H, W = x.shape
        
        # 计算门控概率
        gate_logits = self.gate(x)  # (B, num_experts * num_gates)
        gate_logits = gate_logits.view(B, self.num_gates, self.num_experts)  # (B, 4, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, 4, num_experts)
        
        # 计算所有专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, C, H, W)
        
        # Top-k 选择
        topk = min(self.top, self.num_experts)
        topk_probs, topk_indices = torch.topk(gate_probs, topk, dim=-1)  # (B, 4, topk)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # 计算4个门控的输出
        outputs = []
        for gate_idx in range(self.num_gates):
            out = torch.zeros_like(x)
            for b in range(B):
                for k in range(topk):
                    expert_idx = topk_indices[b, gate_idx, k]
                    weight = topk_probs[b, gate_idx, k]
                    out[b] += weight * expert_outputs[b, expert_idx]
            outputs.append(out)
        
        # 负载均衡损失
        expert_usage = gate_probs.mean(dim=1).mean(dim=0)  # (num_experts,)
        loss = self._cv_squared(expert_usage)
        
        return outputs[0], outputs[1], outputs[2], outputs[3], loss
    
    def _cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """计算变异系数的平方"""
        eps = 1e-10
        mean = x.mean()
        var = x.var()
        return var / (mean ** 2 + eps)


class MoEMicro(nn.Module):
    """
    微型专家混合模块（极简版本）
    
    特点：
    1. 只使用2个专家
    2. 单一门控，输出复制到4路
    3. 最小化计算开销
    """
    def __init__(self, emb_size: int = 32):
        super().__init__()
        self.emb_size = emb_size
        
        # 2个轻量专家
        self.expert1 = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, 3, 1, 1, groups=emb_size, bias=False),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
            nn.Conv2d(emb_size, emb_size, 1, bias=False),
            nn.BatchNorm2d(emb_size)
        )
        
        self.expert2 = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, 5, 1, 2, groups=emb_size, bias=False),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
            nn.Conv2d(emb_size, emb_size, 1, bias=False),
            nn.BatchNorm2d(emb_size)
        )
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(emb_size, 2),
            nn.Softmax(dim=-1)
        )
        
        # 4路分化卷积
        self.split_convs = nn.ModuleList([
            nn.Conv2d(emb_size, emb_size, 1) for _ in range(4)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 门控权重
        gate_weights = self.gate(x)  # (B, 2)
        w1 = gate_weights[:, 0:1, None, None]  # (B, 1, 1, 1)
        w2 = gate_weights[:, 1:2, None, None]  # (B, 1, 1, 1)
        
        # 专家输出
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        
        # 混合
        mixed = w1 * e1 + w2 * e2 + x  # 残差连接
        
        # 4路分化输出
        o1 = self.split_convs[0](mixed)
        o2 = self.split_convs[1](mixed)
        o3 = self.split_convs[2](mixed)
        o4 = self.split_convs[3](mixed)
        
        # 负载均衡损失（简化）
        loss = ((w1.mean() - 0.5) ** 2 + (w2.mean() - 0.5) ** 2)
        
        return o1, o2, o3, o4, loss.squeeze()
