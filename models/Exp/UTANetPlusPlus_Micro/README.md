# UTANet++ Micro

## 概述

UTANet++ Micro 是 UTANet++ 的超轻量化版本，在保持核心设计思想的同时，大幅减少参数量和计算复杂度。

### 设计目标

- **参数量**: < 3M（相比 UTANet++ 的 ~25M 减少 80%+）
- **计算量**: 显著降低 FLOPs
- **性能**: 保持接近原版的分割性能

## 核心技术

### 1. 轻量化编码器

放弃预训练的 ResNet34 backbone，使用自定义的轻量化编码器：

```python
# 深度可分离卷积：参数量减少约 k²/C_out 倍
DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3)

# 轴向深度卷积：参数量从 k² 减少到 2k
AxialDepthwiseConv(dim, kernel_size=7)
```

### 2. 简化全尺度解码器

- 减少拼接通道数：64 → 16
- 使用 1x1 卷积进行通道转换
- 深度可分离卷积进行特征处理

### 3. 轻量化注意力机制

- 移除 SE 模块中的全连接层
- 使用 ECA（Efficient Channel Attention）替代
- 简化门控机制

### 4. 微型 MoE

- 专家数量：4 → 2
- 使用深度可分离卷积实现专家网络
- 简化门控机制

## 模型变体

| 变体 | 基础通道 | MoE | 参数量 | 适用场景 |
|------|----------|-----|--------|----------|
| **Micro** | 16 | ✓ | ~2.5M | 标准轻量化 |
| **Nano** | 8 | ✗ | ~0.5M | 极致轻量化 |

## 使用方法

### U-Bench 训练框架集成

模型已集成到 U-Bench 训练框架，可直接通过命令行使用：

```bash
python main.py --model UTANetPlusPlus_Micro \
    --base_dir ./data/xca_dataset \
    --dataset_name xca \
    --train_file_dir train.txt \
    --val_file_dir val.txt \
    --batch_size 8 \
    --max_epochs 100 \
    --base_lr 0.01 \
    --img_size 224 \
    --num_classes 1 \
    --input_channel 3 \
    --pretrained  # 启用 MoE 模块
```

### 基本使用

```python
from models.Exp.UTANetPlusPlus_Micro import utanet_plusplus_micro, utanet_plusplus_nano

# 创建 Micro 版本（U-Bench 兼容接口）
model = utanet_plusplus_micro(
    input_channel=3,
    num_classes=1,
    pretrained=True,  # 控制是否使用 MoE 模块
    base_channels=16,
    deep_supervision=True
)

# 创建 Nano 版本（更轻量）
model_nano = utanet_plusplus_nano(input_channel=3, num_classes=1)
```

### 自定义配置

```python
from models.Exp.UTANetPlusPlus_Micro import UTANetPlusPlus_Micro

model = UTANetPlusPlus_Micro(
    n_channels=3,           # 输入通道数
    n_classes=1,            # 输出类别数
    base_channels=16,       # 基础通道数（越小越轻量）
    use_moe=True,           # 是否使用 MoE
    deep_supervision=True,  # 是否启用深度监督
    cat_channels=16         # 全尺度连接的拼接通道数
)
```

### 参数说明

- **input_channel/n_channels**: 输入图像通道数（RGB=3，灰度=1）
- **num_classes/n_classes**: 分割类别数（二分类=1，多分类>1）
- **pretrained**: U-Bench 框架参数，控制是否使用 MoE 模块
- **base_channels**: 基础通道数，越小模型越轻量（推荐：16=标准，8=极轻量）
- **deep_supervision**: 是否启用深度监督，提升训练效果
- **cat_channels**: 全尺度连接的通道数，影响解码器复杂度

## 架构图

```
输入 (3, 224, 224)
     │
     ▼
┌─────────────────────────────────────┐
│  轻量化编码器 (DepthwiseSeparable)   │
│  E1: (16, 224) → E2: (16, 112)      │
│  E3: (32, 56) → E4: (64, 28)        │
│  E5: (256, 14) + ASPP               │
└─────────────────────────────────────┘
     │
     ▼ (可选)
┌─────────────────────────────────────┐
│  微型 MoE (2 experts)                │
│  特征路由与混合                       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  轻量化全尺度解码器                   │
│  D4: (64, 28) ← E1-E5               │
│  D3: (32, 56) ← E1-E5 + D4          │
│  D2: (16, 112) ← E1-E5 + D3-D4      │
│  D1: (8, 224) ← E1-E5 + D2-D4       │
│  + LightGatedAttention + ECA        │
└─────────────────────────────────────┘
     │
     ▼
输出 (n_classes, 224, 224)
```

## 性能参考

在典型医学图像分割任务上的预期性能：

| 模型 | 参数量 | Dice | IoU | FPS |
|------|--------|------|-----|-----|
| UTANet++ | ~25M | 基准 | 基准 | 基准 |
| **Micro** | ~2.5M | -2~3% | -2~3% | +50~100% |
| **Nano** | ~0.5M | -5~7% | -5~7% | +100~200% |

## 轻量化技术详解

### 深度可分离卷积

标准卷积参数量：`C_in × C_out × K × K`

深度可分离卷积参数量：`C_in × K × K + C_in × C_out`

节省比例：约 `K²/C_out` 倍

### 轴向深度卷积

将 2D 卷积分解为两个 1D 卷积：
- 垂直方向：`(K, 1)` 
- 水平方向：`(1, K)`

参数量从 `K²` 减少到 `2K`

### ECA (Efficient Channel Attention)

相比 SE 模块：
- 移除全连接层的参数
- 使用 1D 卷积实现通道间交互
- 自适应计算卷积核大小

## 文件结构

```
UTANetPlusPlus_Micro/
├── __init__.py              # 模块导出
├── UTANetPlusPlus_Micro.py  # 主模型
├── modules_micro.py         # 轻量化模块
├── ta_mosc_light.py         # 轻量化 MoE
└── README.md                # 文档
```

## 参考文献

1. **UTANet++**: Full-scale skip connections + Gated Attention
2. **MALUNet**: Depthwise separable convolution, Gated attention units
3. **ULite**: Axial depthwise convolution
4. **CMUNeXt**: Residual depthwise convolution blocks
5. **ECA-Net**: Efficient Channel Attention

## License

与主项目保持一致。
