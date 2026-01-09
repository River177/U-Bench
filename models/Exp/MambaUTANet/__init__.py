"""
Mamba-UTANet: Long-Range Dependency Enhanced UTANet

将Mamba的状态空间模型(SSM)引入UTANet，增强长程依赖建模能力。

核心创新：
1. SS2D-Enhanced Bottleneck: 在瓶颈层使用状态空间模型替代普通卷积
2. Mamba-Expert: 将MoE专家网络从1x1卷积升级为Mamba块
3. 保留TA-MoSC的自适应特征路由机制
"""

from .MambaUTANet import MambaUTANet, mambautanet

__all__ = ['MambaUTANet', 'mambautanet']

