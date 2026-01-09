"""
Mamba State Space Model 核心组件

基于 VMamba/Vision Mamba 的 SS2D 实现，用于医学图像分割。

核心思想：
- 将 2D 特征图沿 4 个方向（行正/反、列正/反）进行选择性扫描
- 使用状态空间模型建模长程依赖，计算复杂度 O(N) vs 自注意力的 O(N²)
- 适合处理高分辨率医学图像

参考文献：
- VMamba: Visual State Space Model
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
"""

import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

# 尝试导入 mamba_ssm 官方实现
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    selective_scan_fn = None

# 尝试导入备用实现
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    selective_scan_fn_v1 = None


class SS2D(nn.Module):
    """
    二维选择性状态空间模块（Selective State Space 2D）
    
    核心思想：
    1. 将 2D 特征图展平成 4 个方向的序列（行正/反、列正/反）
    2. 对每个方向进行选择性扫描（类似 RNN 但更高效）
    3. 融合 4 个方向的输出，实现全局建模
    
    相比自注意力的优势：
    - 计算复杂度 O(N) vs O(N²)
    - 更适合处理长序列/高分辨率特征图
    - 参数量更少
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        Args:
            d_model: 模型维度（输入/输出通道数）
            d_state: 状态空间维度（默认16）
            d_conv: 深度卷积核大小（默认3）
            expand: 扩展因子（默认2，即内部维度是输入的2倍）
            dt_rank: 时间步投影的秩（默认"auto"，自动计算）
            dropout: Dropout 概率
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影：将 d_model 映射到 2*d_inner（主分支 + 门控分支）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 深度可分离卷积：提取局部特征
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 4 个扫描方向的投影权重
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # 时间步投影
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # 状态空间参数 A 和 D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # 输出层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """初始化时间步投影"""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """初始化状态转移矩阵 A"""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """初始化跳跃连接参数 D"""
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        """
        选择性扫描的核心前向过程
        
        Args:
            x: 输入特征 (B, C, H, W)
        
        Returns:
            y: 输出特征 (B, H, W, C)
        """
        if selective_scan_fn is None and selective_scan_fn_v1 is None:
            raise ImportError("请安装 mamba_ssm 或 selective_scan 库！\n"
                            "pip install mamba-ssm 或参考 https://github.com/state-spaces/mamba")
        
        self.selective_scan = selective_scan_fn if selective_scan_fn is not None else selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4  # 4 个扫描方向

        # 构建 4 个方向的序列：行正、列正、行反、列反
        x_hwwh = torch.stack([
            x.view(B, -1, L),  # 行扫描
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)  # 列扫描
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # 添加反向扫描

        # 投影得到 dt, B, C 参数
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 选择性扫描
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # 融合 4 个方向的输出
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, H, W, C) 注意是 BHWC 格式！
        
        Returns:
            out: 输出特征 (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # 双线性投影：主分支 x 和门控分支 z
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 深度卷积 + 选择性扫描
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)  # (B, H, W, C)
        
        # 门控调制
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """
    VSS Block：基于状态空间的注意力块
    
    结构：LayerNorm -> SS2D -> DropPath -> 残差连接
    
    作用：在不改变张量形状的情况下，增强特征的全局建模能力
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            drop_path: DropPath 概率
            norm_layer: 归一化层
            attn_drop_rate: 注意力 Dropout 概率
            d_state: 状态空间维度
        """
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: 输入特征 (B, H, W, C)
        
        Returns:
            输出特征 (B, H, W, C)
        """
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("SS2D 模块测试")
    print("=" * 60)
    
    # 测试 SS2D
    print("\n测试 SS2D:")
    d_model = 64
    batch_size = 2
    H, W = 56, 56
    
    ss2d = SS2D(d_model=d_model, d_state=16, expand=2)
    x = torch.randn(batch_size, H, W, d_model)  # 注意是 BHWC 格式
    
    print(f"  输入形状: {x.shape}")
    y = ss2d(x)
    print(f"  输出形状: {y.shape}")
    assert y.shape == x.shape, "输出形状不匹配！"
    
    # 测试 VSSBlock
    print("\n测试 VSSBlock:")
    vss_block = VSSBlock(hidden_dim=d_model, drop_path=0.1, d_state=16)
    y = vss_block(x)
    print(f"  输出形状: {y.shape}")
    assert y.shape == x.shape, "输出形状不匹配！"
    
    # 参数统计
    ss2d_params = sum(p.numel() for p in ss2d.parameters())
    vss_block_params = sum(p.numel() for p in vss_block.parameters())
    print(f"\nSS2D 参数量: {ss2d_params / 1e3:.2f}K")
    print(f"VSSBlock 参数量: {vss_block_params / 1e3:.2f}K")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

