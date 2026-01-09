"""
Lightweight Hierarchical UTANet
轻量化层次化医学图像分割网络

作者：基于UTANet改进
日期：2026-01-09

核心模块：
- HierarchicalMoE: 层次化专家混合
- LightweightASPP: 轻量空洞空间金字塔池化
- LightweightUpBlock: 深度可分离解码器
- LightweightHierarchicalUTANet: 完整模型
"""

from .LightweightHierarchicalUTANet import (
    LightweightHierarchicalUTANet,
    lightweight_hierarchical_utanet
)

from .modules import (
    HierarchicalExpert,
    HierarchicalMoE,
    LightweightASPP,
    LightweightUpBlock
)

__all__ = [
    'LightweightHierarchicalUTANet',
    'lightweight_hierarchical_utanet',
    'HierarchicalExpert',
    'HierarchicalMoE',
    'LightweightASPP',
    'LightweightUpBlock'
]

__version__ = '1.0.0'

