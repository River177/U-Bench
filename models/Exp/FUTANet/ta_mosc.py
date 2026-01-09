
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
        
        用于负载均衡损失（Load Balancing Loss）：
        - 鼓励均匀使用所有专家，避免某些专家被过度使用而其他专家被忽略
        - CV² = Var(X) / Mean(X)²，衡量数据分布的离散程度
        - CV²越小，表示专家使用越均匀；CV²=0表示完全均匀
        
        Args:
            x (torch.Tensor): 专家使用量张量，形状为(num_experts,)
                每个元素表示对应专家被选中的总次数或总权重
            
        Returns:
            torch.Tensor: 变异系数的平方，标量值
        """
        eps = 1e-10  # 防止除零的小常数
        # 如果只有一个样本，无法计算方差，返回0
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        # CV² = 方差 / 均值的平方
        return x.float().var() / (x.float().mean()**2 + eps)
        
    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过单个门控机制处理输入特征
        
        处理流程：
        1. 全局平均池化获取全局特征表示
        2. 计算每个专家的选择概率（门控输出）
        3. 选择top-k个专家（稀疏激活）
        4. 将输入分配给选中的专家处理
        5. 加权聚合专家输出
        6. 计算负载均衡损失
        
        Args:
            x (torch.Tensor): 输入特征张量 (batch_size, emb_size, H, W)
            gate_weights (nn.Parameter): 当前门控的权重矩阵 (emb_size, num_experts)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 输出特征张量 (batch_size, emb_size, H, W)
                - 负载均衡损失（标量）
        """
        batch_size, emb_size, H, W = x.shape
        
        # ========== 步骤1：计算门控概率 ==========
        # 全局平均池化：将空间特征图压缩为全局特征向量
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x0 = self.gap(x).view(batch_size, emb_size)
        # 门控计算：全局特征 × 门控权重 -> (B, num_experts)
        # 然后softmax归一化得到每个专家的选择概率
        gate_out = F.softmax(x0 @ gate_weights, dim=1)  # (B, num_experts)
        
        # ========== 步骤2：计算专家使用量（用于负载均衡） ==========
        # 统计每个专家被选中的总概率（跨批次求和）
        expert_usage = gate_out.sum(0)  # (num_experts,)
        
        # ========== 步骤3：选择top-k专家（稀疏激活） ==========
        # 对每个样本，选择概率最高的top-k个专家
        top_weights, top_index = torch.topk(gate_out, self.top, dim=1)  # (B, top), (B, top)
        # 找出实际被使用的专家索引
        used_experts = torch.unique(top_index)  # 被选中的专家ID
        # 找出未使用的专家（用于训练时的负载均衡）
        unused_experts = set(range(len(self.experts))) - set(used_experts.tolist())
        
        # 对top-k权重再次softmax归一化，确保权重和为1
        top_weights = F.softmax(top_weights, dim=1)  # (B, top)
        
        # ========== 步骤4：准备输入数据用于并行专家处理 ==========
        # 扩展输入：为每个样本的top-k选择准备k份输入
        # (B, C, H, W) -> (B, 1, C, H, W) -> (B, top, C, H, W) -> (B*top, C, H, W)
        x_expanded = x.unsqueeze(1).expand(batch_size, self.top, emb_size, H, W).reshape(-1, emb_size, H, W)
        # 初始化输出张量
        y = torch.zeros_like(x_expanded)  # (B*top, C, H, W)
        
        # ========== 步骤5：处理每个专家 ==========
        for expert_i, expert_model in enumerate(self.experts):
            # 找出当前专家被选中的样本索引
            expert_mask = (top_index == expert_i).view(-1)  # (B*top,)
            expert_indices = expert_mask.nonzero().flatten()  # 被选中样本的全局索引
            
            if expert_indices.numel() > 0:
                # 如果该专家被选中，处理对应的输入
                x_expert = x_expanded[expert_indices]  # (N, C, H, W)，N为选中样本数
                y_expert = expert_model(x_expert)  # 通过专家网络处理
                # 将专家输出累加到对应位置
                y = y.index_add(dim=0, index=expert_indices, source=y_expert)
            elif expert_i in unused_experts and self.training:
                # 训练时：如果某个专家未被使用，强制使用一次（负载均衡）
                # 随机选择一个样本让该专家处理
                random_sample = torch.randint(0, x.size(0), (1,), device=x.device)
                x_expert = x_expanded[random_sample]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=random_sample, source=y_expert)
        
        # ========== 步骤6：加权聚合专家输出 ==========
        # 将权重扩展到空间维度：(B, top) -> (B*top, 1, 1, 1)
        top_weights = top_weights.view(-1, 1, 1, 1).expand_as(y)
        # 对每个专家输出应用对应权重
        y = y * top_weights  # (B*top, C, H, W)
        # 重塑并求和：将top-k个输出聚合为单个输出
        # (B*top, C, H, W) -> (B, top, C, H, W) -> (B, C, H, W)
        y = y.view(batch_size, self.top, emb_size, H, W).sum(dim=1)
        
        # ========== 步骤7：计算负载均衡损失 ==========
        return y, self.cv_squared(expert_usage)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：通过所有4个门控机制处理输入
        
        每个门控机制独立地：
        1. 根据输入特征选择top-k专家
        2. 通过选中的专家处理特征
        3. 生成一个输出特征表示
        
        4个门控的输出（y1-y4）会被UTANet用于不同尺度的特征路由。
        
        Args:
            x (torch.Tensor): 输入特征张量 (batch_size, emb_size, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - y1: 门控1的输出特征 (batch_size, emb_size, H, W)
                - y2: 门控2的输出特征 (batch_size, emb_size, H, W)
                - y3: 门控3的输出特征 (batch_size, emb_size, H, W)
                - y4: 门控4的输出特征 (batch_size, emb_size, H, W)
                - loss: 4个门控的负载均衡损失之和（标量）
        """
        # 通过4个独立的门控机制处理输入，每个门控产生一个输出
        y1, loss1 = self._process_gate(x, self.gate1)  # 门控1：用于路由到e1/e2尺度
        y2, loss2 = self._process_gate(x, self.gate2)  # 门控2：用于路由到e2尺度
        y3, loss3 = self._process_gate(x, self.gate3)  # 门控3：用于路由到e3尺度
        y4, loss4 = self._process_gate(x, self.gate4)  # 门控4：用于路由到e4尺度
        
        # 合并所有门控的负载均衡损失
        # 鼓励所有门控都均匀使用所有专家
        loss = loss1 + loss2 + loss3 + loss4
        
        # 训练时可以打印专家使用情况（调试用）
        #if self.training:
            #print(f"Expert Usage - Gate1: {self._format_usage([loss1,loss])}")
            #print(f"Expert Usage - Gate2: {self._format_usage([loss2,loss])}")
            #print(f"Expert Usage - Gate3: {self._format_usage([loss3,loss])}")
            #print(f"Expert Usage - Gate4: {self._format_usage([loss4,loss])}")
        
        return y1, y2, y3, y4, loss
    
    def _format_usage(self, usage: torch.Tensor) -> str:
        """Format expert usage statistics for logging."""
        return f"Min: {usage.min():.4f}, Max: {usage.max():.4f}, CV²: {self.cv_squared(usage):.4f}"

def count_parameters(model: nn.Module) -> str:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model.
        
    Returns:
        str: Formatted string with parameter count.
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params >= 1e6:
        return f"{params / 1e6:.2f}M parameters"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K parameters"
    else:
        return f"{params} parameters"

if __name__ == '__main__':
    """Unit test for MoE module."""
    try:
        # Initialize model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MoE(num_experts=4, top=2, emb_size=128, H=224, W=224).to(device)
        model.train()
        
        # Generate random input
        emb = torch.randn(6, 128, 224, 224).to(device)
        
        # Forward pass
        out1, out2, out3, out4, loss = model(emb)
        
        # Verify output shapes
        assert out1.shape == emb.shape, f"Output shape mismatch: {out1.shape} vs {emb.shape}"
        assert out2.shape == emb.shape, f"Output shape mismatch: {out2.shape} vs {emb.shape}"
        assert out3.shape == emb.shape, f"Output shape mismatch: {out3.shape} vs {emb.shape}"
        assert out4.shape == emb.shape, f"Output shape mismatch: {out4.shape} vs {emb.shape}"
        
        print("\n=== MoE Module Test Passed ===")

        print(f"Input Shape: {emb.shape}")
        print(f"Output Shapes: {out1.shape}, {out2.shape}, {out3.shape}, {out4.shape}")
        print(f"Load Balancing Loss: {loss.item():.4f}")
        print(f"Model Parameters: {count_parameters(model)}")
        
    except Exception as e:
        print(f"Test failed: {e}")