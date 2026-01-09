
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Set

class Expert(nn.Module):
    """
    专家网络模块：由多个1x1卷积层组成的轻量级特征变换网络
    
    每个专家是一个独立的特征处理单元，采用"扩展-压缩"的瓶颈结构：
    - 输入维度 -> 扩展维度（hidden_rate倍）-> 压缩回输入维度
    - 使用1x1卷积，不改变空间尺寸，只进行通道间的特征变换
    - 这种设计使得每个专家可以学习不同的特征表示模式
    
    Args:
        emb_size (int): 输入/输出嵌入维度（特征通道数）
        hidden_rate (int, optional): 隐藏层扩展倍数，默认2（即隐藏层维度是输入的2倍）
    """
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        # 计算隐藏层维度：输入维度的hidden_rate倍
        hidden_emb = hidden_rate * emb_size
        # 构建专家网络：瓶颈结构（Bottleneck Architecture）
        self.seq = nn.Sequential(
            # 第一层：扩展维度 emb_size -> hidden_emb
            nn.Conv2d(emb_size, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            # 第二层：在扩展维度上进行特征变换 hidden_emb -> hidden_emb
            nn.Conv2d(hidden_emb, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            # 批归一化：加速训练，稳定梯度
            nn.BatchNorm2d(hidden_emb),
            # ReLU激活：引入非线性
            nn.ReLU(),
            # 第三层：压缩回原始维度 hidden_emb -> emb_size
            nn.Conv2d(hidden_emb, emb_size, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：通过专家网络处理输入特征
        
        Args:
            x: 输入特征张量 (B, emb_size, H, W)
        
        Returns:
            处理后的特征张量 (B, emb_size, H, W)，空间尺寸不变
        """
        return self.seq(x)

class MoE(nn.Module):
    """
    专家混合（Mixture of Experts, MoE）模块，包含多个门控机制
    
    MoE的核心思想：
    1. 维护多个专家网络，每个专家学习不同的特征变换模式
    2. 使用门控网络（Gating Network）根据输入特征动态选择最相关的专家
    3. 只激活top-k个专家，提高计算效率（稀疏激活）
    4. 通过负载均衡损失确保所有专家都被充分利用
    
    本实现使用4个独立的门控机制（gate1-4），每个门控产生一个输出特征。
    这种设计允许模型同时学习多种特征表示模式。
    
    Args:
        num_experts (int): 专家网络的数量（默认4个）
        top (int, optional): 每次选择top-k个专家（默认2，即稀疏激活）
        emb_size (int, optional): 特征嵌入维度（默认128）
        H (int, optional): 输入特征图高度（默认224，实际运行时动态确定）
        W (int, optional): 输入特征图宽度（默认224，实际运行时动态确定）
    """
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 128, H: int = 224, W: int = 224):
        super().__init__()
        # 创建多个专家网络：每个专家是独立的特征变换模块
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(num_experts)])
        
        # 4个独立的门控权重矩阵：每个门控学习不同的专家选择策略
        # 形状：(emb_size, num_experts) - 每个特征维度对应每个专家的权重
        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        
        # 初始化门控权重
        self._initialize_weights()
        
        # 全局平均池化：将空间特征图压缩为全局特征向量
        # 用于计算门控概率（基于全局特征选择专家）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # top-k选择：每次激活的专家数量
        self.top = top
        
    def _initialize_weights(self) -> None:
        """
        初始化门控权重：使用Xavier均匀初始化
        
        Xavier初始化有助于保持前向和反向传播中的梯度方差，
        使得训练更加稳定，特别适合全连接层和卷积层。
        """
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)
        
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算变异系数的平方（Coefficient of Variation Squared）
        
        用于负载均衡损失的计算，衡量专家使用分布的均匀性。
        cv² = (std/mean)² = (var/mean²)
        
        Args:
            x: 输入张量
        
        Returns:
            变异系数的平方
        """
        eps = 1e-10
        mean = x.mean()
        var = x.var()
        return var / (mean ** 2 + eps)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：通过MoE模块处理输入特征
        
        流程：
        1. 全局平均池化得到全局特征向量
        2. 通过4个门控网络计算专家选择概率
        3. 选择top-k个专家
        4. 加权组合专家输出
        5. 计算负载均衡损失
        
        Args:
            x: 输入特征 (B, emb_size, H, W)
        
        Returns:
            o1, o2, o3, o4: 4个门控的输出特征 (B, emb_size, H, W)
            loss: 负载均衡损失（标量）
        """
        B, C, H, W = x.shape
        
        # 步骤1：全局平均池化，得到全局特征向量
        x_gap = self.gap(x)  # (B, C, 1, 1)
        x_gap = x_gap.view(B, C)  # (B, C)
        
        # 步骤2：计算4个门控的专家选择概率
        # 每个门控：通过矩阵乘法计算每个专家的得分
        gate1_logits = torch.matmul(x_gap, self.gate1)  # (B, num_experts)
        gate2_logits = torch.matmul(x_gap, self.gate2)  # (B, num_experts)
        gate3_logits = torch.matmul(x_gap, self.gate3)  # (B, num_experts)
        gate4_logits = torch.matmul(x_gap, self.gate4)  # (B, num_experts)
        
        # Softmax归一化，得到概率分布
        gate1_probs = F.softmax(gate1_logits, dim=-1)  # (B, num_experts)
        gate2_probs = F.softmax(gate2_logits, dim=-1)  # (B, num_experts)
        gate3_probs = F.softmax(gate3_logits, dim=-1)  # (B, num_experts)
        gate4_probs = F.softmax(gate4_logits, dim=-1)  # (B, num_experts)
        
        # 步骤3：Top-k专家选择（稀疏激活）
        # 只选择概率最高的top个专家
        topk = min(self.top, len(self.experts))
        gate1_topk_probs, gate1_topk_indices = torch.topk(gate1_probs, topk, dim=-1)  # (B, topk)
        gate2_topk_probs, gate2_topk_indices = torch.topk(gate2_probs, topk, dim=-1)
        gate3_topk_probs, gate3_topk_indices = torch.topk(gate3_probs, topk, dim=-1)
        gate4_topk_probs, gate4_topk_indices = torch.topk(gate4_probs, topk, dim=-1)
        
        # 归一化top-k概率（使其和为1）
        gate1_topk_probs = gate1_topk_probs / (gate1_topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        gate2_topk_probs = gate2_topk_probs / (gate2_topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        gate3_topk_probs = gate3_topk_probs / (gate3_topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        gate4_topk_probs = gate4_topk_probs / (gate4_topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # 步骤4：通过专家网络处理输入，并加权组合
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个输出: (B, C, H, W)
        
        # 将专家输出堆叠
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, C, H, W)
        
        # 为每个门控计算加权输出
        o1 = torch.zeros_like(x)
        o2 = torch.zeros_like(x)
        o3 = torch.zeros_like(x)
        o4 = torch.zeros_like(x)
        
        for i in range(B):
            # Gate 1
            for j, idx in enumerate(gate1_topk_indices[i]):
                o1[i] += gate1_topk_probs[i, j] * expert_outputs[i, idx]
            
            # Gate 2
            for j, idx in enumerate(gate2_topk_indices[i]):
                o2[i] += gate2_topk_probs[i, j] * expert_outputs[i, idx]
            
            # Gate 3
            for j, idx in enumerate(gate3_topk_indices[i]):
                o3[i] += gate3_topk_probs[i, j] * expert_outputs[i, idx]
            
            # Gate 4
            for j, idx in enumerate(gate4_topk_indices[i]):
                o4[i] += gate4_topk_probs[i, j] * expert_outputs[i, idx]
        
        # 步骤5：计算负载均衡损失
        # 目标：确保所有专家都被均匀使用
        # 方法：最小化专家使用概率的变异系数
        
        # 计算每个专家的平均使用概率（跨所有门控和batch）
        expert_usage = (gate1_probs + gate2_probs + gate3_probs + gate4_probs) / 4.0  # (B, num_experts)
        expert_usage_mean = expert_usage.mean(dim=0)  # (num_experts,)
        
        # 计算负载均衡损失：变异系数的平方
        loss = self.cv_squared(expert_usage_mean)
        
        return o1, o2, o3, o4, loss

