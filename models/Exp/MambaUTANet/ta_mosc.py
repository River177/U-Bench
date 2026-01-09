"""
TA-MoSC with Mamba Experts

将原始的1x1卷积专家网络升级为Mamba增强专家，结合：
- 状态空间模型（SS2D）：建模长程依赖
- 前馈网络（FFN）：保留局部特征变换能力

核心改进：
1. MambaExpert: SS2D + FFN 双分支结构
2. 保留原始MoE的门控机制和负载均衡损失
3. 轻量化设计：expand=1.5 而不是2，减少参数量
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Set
from functools import partial

from timm.models.layers import DropPath
from .mamba_sys import SS2D


class MambaExpert(nn.Module):
    """
    Mamba增强的专家网络
    
    相比原始Expert（3层1x1卷积）的改进：
    1. 引入SS2D进行长程依赖建模
    2. FFN保留局部特征变换能力
    3. 双分支结构，相互补充
    
    结构：
        输入 -> [SS2D分支] + [FFN分支] -> 输出
    """
    def __init__(self, emb_size: int = 64, d_state: int = 16, drop_path: float = 0.0):
        """
        Args:
            emb_size: 特征嵌入维度
            d_state: 状态空间维度（默认16）
            drop_path: DropPath概率（默认0.0）
        """
        super().__init__()
        self.emb_size = emb_size
        
        # === Mamba分支：长程依赖建模 ===
        # 注意：SS2D需要BHWC格式输入
        self.norm1 = nn.LayerNorm(emb_size)
        self.ss2d = SS2D(
            d_model=emb_size,
            d_state=d_state,
            d_conv=3,
            expand=1.5,  # 轻量化：1.5倍扩展而不是2倍
            dropout=0.0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # === FFN分支：局部特征变换 ===
        # 保持与原始Expert类似的结构
        self.norm2 = nn.LayerNorm(emb_size)
        ffn_hidden = int(emb_size * 2)  # 2倍扩展
        self.ffn = nn.Sequential(
            nn.Conv2d(emb_size, ffn_hidden, 1),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, emb_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (B, C, H, W) 注意是BCHW格式
        
        Returns:
            处理后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # === Mamba分支（长程依赖）===
        # 转换为BHWC格式
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_norm = self.norm1(x_bhwc)
        x_mamba = self.ss2d(x_norm)  # (B, H, W, C)
        x_mamba = self.drop_path(x_mamba)
        # 转回BCHW格式
        x_mamba = x_mamba.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = x + x_mamba  # 残差连接
        
        # === FFN分支（局部特征）===
        # 先转为BHWC做LayerNorm，再转回BCHW做卷积
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x = x + self.ffn(x_norm2)  # 残差连接
        
        return x


class LightweightExpert(nn.Module):
    """
    轻量级专家网络（作为对比和备用）
    
    保持原始Expert的结构，但减少参数量
    """
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        self.seq = nn.Sequential(
            nn.Conv2d(emb_size, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(hidden_emb, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_emb),
            nn.ReLU(),
            nn.Conv2d(hidden_emb, emb_size, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MambaMoE(nn.Module):
    """
    Mamba增强的专家混合模块
    
    核心改进：
    1. 将4个专家从简单的1x1卷积升级为MambaExpert
    2. 保留原始MoE的门控机制和top-k选择
    3. 保留负载均衡损失
    
    相比原始MoE的优势：
    - 专家具有长程依赖建模能力
    - 更强的特征表达能力
    - 参数量增加有限（通过轻量化设计控制）
    """
    def __init__(
        self, 
        num_experts: int = 4, 
        top: int = 2, 
        emb_size: int = 64,
        use_mamba_expert: bool = True,
        d_state: int = 16,
        drop_path: float = 0.0,
    ):
        """
        Args:
            num_experts: 专家数量（默认4）
            top: top-k选择的k值（默认2）
            emb_size: 特征嵌入维度（默认64）
            use_mamba_expert: 是否使用Mamba专家（默认True，False时退化为轻量级专家）
            d_state: 状态空间维度（默认16）
            drop_path: DropPath概率（默认0.0）
        """
        super().__init__()
        self.num_experts = num_experts
        self.top = top
        self.emb_size = emb_size
        self.use_mamba_expert = use_mamba_expert
        
        # === 创建专家网络 ===
        if use_mamba_expert:
            print(f"[MambaMoE] Using MambaExpert with {num_experts} experts")
            self.experts = nn.ModuleList([
                MambaExpert(emb_size, d_state=d_state, drop_path=drop_path) 
                for _ in range(num_experts)
            ])
        else:
            print(f"[MambaMoE] Using LightweightExpert with {num_experts} experts")
            self.experts = nn.ModuleList([
                LightweightExpert(emb_size) 
                for _ in range(num_experts)
            ])
        
        # === 4个门控权重矩阵 ===
        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        
        # 初始化门控权重
        self._initialize_weights()
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
    def _initialize_weights(self) -> None:
        """Xavier均匀初始化门控权重"""
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)
        
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算变异系数的平方（负载均衡损失）
        
        CV² = Var(X) / Mean(X)²
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
        
    def _process_gate(
        self, 
        x: torch.Tensor, 
        gate_weights: nn.Parameter
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过单个门控机制处理输入特征
        
        处理流程：
        1. 全局平均池化 -> 门控概率计算
        2. top-k专家选择
        3. 专家并行处理（考虑到Mamba专家的计算特性）
        4. 加权聚合
        5. 负载均衡损失计算
        
        Args:
            x: 输入特征 (B, C, H, W)
            gate_weights: 门控权重 (C, num_experts)
        
        Returns:
            y: 输出特征 (B, C, H, W)
            loss: 负载均衡损失
        """
        batch_size, emb_size, H, W = x.shape
        
        # === 步骤1：计算门控概率 ===
        x0 = self.gap(x).view(batch_size, emb_size)  # (B, C)
        gate_out = F.softmax(x0 @ gate_weights, dim=1)  # (B, num_experts)
        
        # === 步骤2：负载均衡统计 ===
        expert_usage = gate_out.sum(0)  # (num_experts,)
        
        # === 步骤3：top-k选择 ===
        top_weights, top_index = torch.topk(gate_out, self.top, dim=1)  # (B, top), (B, top)
        used_experts = torch.unique(top_index)
        unused_experts = set(range(self.num_experts)) - set(used_experts.tolist())
        top_weights = F.softmax(top_weights, dim=1)  # 归一化
        
        # === 步骤4：专家处理 ===
        # 扩展输入
        x_expanded = x.unsqueeze(1).expand(batch_size, self.top, emb_size, H, W).reshape(-1, emb_size, H, W)
        y = torch.zeros_like(x_expanded)
        
        for expert_i, expert_model in enumerate(self.experts):
            # 找出当前专家被选中的样本
            expert_mask = (top_index == expert_i).view(-1)
            expert_indices = expert_mask.nonzero().flatten()
            
            if expert_indices.numel() > 0:
                x_expert = x_expanded[expert_indices]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=expert_indices, source=y_expert)
            elif expert_i in unused_experts and self.training:
                # 训练时强制使用未选中的专家（负载均衡）
                random_sample = torch.randint(0, batch_size * self.top, (1,), device=x.device)
                x_expert = x_expanded[random_sample]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=random_sample, source=y_expert)
        
        # === 步骤5：加权聚合 ===
        top_weights_expanded = top_weights.view(-1, 1, 1, 1).expand_as(y)
        y = y * top_weights_expanded
        y = y.view(batch_size, self.top, emb_size, H, W).sum(dim=1)
        
        # === 步骤6：负载均衡损失 ===
        return y, self.cv_squared(expert_usage)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：通过4个门控处理输入
        
        Args:
            x: 输入特征 (B, C, H, W)
        
        Returns:
            y1, y2, y3, y4: 4个门控的输出特征
            loss: 负载均衡损失总和
        """
        y1, loss1 = self._process_gate(x, self.gate1)
        y2, loss2 = self._process_gate(x, self.gate2)
        y3, loss3 = self._process_gate(x, self.gate3)
        y4, loss4 = self._process_gate(x, self.gate4)
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return y1, y2, y3, y4, loss


# 保持向后兼容：MoE别名
MoE = MambaMoE


def count_parameters(model: nn.Module) -> str:
    """统计模型参数量"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params >= 1e6:
        return f"{params / 1e6:.2f}M parameters"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K parameters"
    else:
        return f"{params} parameters"


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("MambaMoE Module Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test parameters
    batch_size = 2
    emb_size = 64
    H, W = 112, 112
    
    # === Test 1: Mamba expert version ===
    print("\n" + "=" * 60)
    print("Testing MambaMoE (use_mamba_expert=True)")
    print("=" * 60)
    
    moe_mamba = MambaMoE(
        num_experts=4, 
        top=2, 
        emb_size=emb_size,
        use_mamba_expert=True,
        d_state=16
    ).to(device)
    
    x = torch.randn(batch_size, emb_size, H, W).to(device)
    print(f"Input shape: {x.shape}")
    
    y1, y2, y3, y4, loss = moe_mamba(x)
    
    print(f"Output shapes: {y1.shape}, {y2.shape}, {y3.shape}, {y4.shape}")
    print(f"Load balancing loss: {loss.item():.6f}")
    print(f"Parameters: {count_parameters(moe_mamba)}")
    
    # Verify output shapes
    assert y1.shape == x.shape, f"Output shape mismatch: {y1.shape} vs {x.shape}"
    assert y2.shape == x.shape, f"Output shape mismatch: {y2.shape} vs {x.shape}"
    assert y3.shape == x.shape, f"Output shape mismatch: {y3.shape} vs {x.shape}"
    assert y4.shape == x.shape, f"Output shape mismatch: {y4.shape} vs {x.shape}"
    
    # === Test 2: Lightweight expert version (comparison) ===
    print("\n" + "=" * 60)
    print("Testing MambaMoE (use_mamba_expert=False, lightweight experts)")
    print("=" * 60)
    
    moe_light = MambaMoE(
        num_experts=4, 
        top=2, 
        emb_size=emb_size,
        use_mamba_expert=False
    ).to(device)
    
    y1_light, y2_light, y3_light, y4_light, loss_light = moe_light(x)
    
    print(f"Output shape: {y1_light.shape}")
    print(f"Load balancing loss: {loss_light.item():.6f}")
    print(f"Parameters: {count_parameters(moe_light)}")
    
    # === Parameter comparison ===
    print("\n" + "=" * 60)
    print("Parameter Comparison")
    print("=" * 60)
    
    params_mamba = sum(p.numel() for p in moe_mamba.parameters())
    params_light = sum(p.numel() for p in moe_light.parameters())
    
    print(f"Mamba expert version: {params_mamba / 1e6:.2f}M")
    print(f"Lightweight expert version: {params_light / 1e3:.2f}K")
    print(f"Parameter increase: {(params_mamba - params_light) / 1e6:.2f}M ({(params_mamba / params_light - 1) * 100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

