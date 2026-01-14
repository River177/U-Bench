# UTANet+ Mamba

## Overview

UTANet+ Mamba is an enhanced medical image segmentation model that combines the efficiency of UTANet+ architecture with **Sparse Mamba Mixture of Experts (MoE)**. This implementation features **TRUE sparse Top-K computation**, where only selected experts are computed, significantly improving computational efficiency.

## Key Features

### 1. Sparse Mamba MoE
- **Mamba Experts Only**: Uses only Mamba (State Space Model) experts for capturing long-range dependencies
- **TRUE Sparse Top-K**: Only computes forward pass for selected Top-K experts (not all experts)
- **4 Independent Gates**: Multiple gating mechanisms for different feature scales
- **Load Balancing**: Automatic load balancing loss to encourage uniform expert usage

### 2. Architecture Components
- **Encoder**: ResNet34 backbone for feature extraction
- **TA-MoSC Module**: Multi-scale feature fusion with sparse Mamba MoE
- **Fast Full-Scale Decoder**: Memory-efficient decoder with multi-scale feature aggregation
- **Gated Attention**: Attention mechanism for feature refinement
- **Deep Supervision**: Multiple auxiliary outputs for better gradient flow

## Sparse Computation Implementation

### Key Difference from Dense MoE

**Dense MoE (Inefficient)**:
```python
# Computes ALL experts regardless of selection
expert_outputs = []
for expert in self.experts:
    expert_outputs.append(expert(x))  # All experts computed!
```

**Sparse MoE (Efficient - Our Implementation)**:
```python
# Only computes selected Top-K experts
selected_experts = torch.unique(topk_indices).tolist()
expert_outputs = {}
for expert_idx in selected_experts:
    expert_outputs[expert_idx] = self.experts[expert_idx](x)  # Only selected!
```

### Benefits
- **Reduced Computation**: Only 50% of experts computed when Top-K=2 out of 4 experts
- **Lower Memory**: No need to store outputs from unused experts
- **Scalability**: Can use more experts without proportional cost increase

## Model Architecture

```
Input (3, 224, 224)
    ↓
ResNet34 Encoder
    ├─ e1: (64, 224, 224)
    ├─ e2: (64, 112, 112)
    ├─ e3: (128, 56, 56)
    ├─ e4: (256, 28, 28)
    └─ e5: (512, 14, 14)
    ↓
Multi-Scale Fusion + Sparse Mamba MoE
    ├─ Gate 1 → o1 (64, 224, 224)
    ├─ Gate 2 → o2 (64, 112, 112)
    ├─ Gate 3 → o3 (128, 56, 56)
    └─ Gate 4 → o4 (256, 28, 28)
    ↓
Fast Full-Scale Decoder + Gated Attention
    ├─ d4: (256, 28, 28)
    ├─ d3: (128, 56, 56)
    ├─ d2: (64, 112, 112)
    └─ d1: (32, 224, 224)
    ↓
Output (1, 224, 224)
```

## Usage

### Basic Usage

```python
from models.Exp.UTANetMamba.UTANetMamba import utanet_mamba

# Create model
model = utanet_mamba(
    input_channel=3,
    num_classes=1,
    pretrained=True
)

# Forward pass
import torch
x = torch.randn(2, 3, 224, 224)
output = model(x)

# With deep supervision (returns list of 5 outputs)
if isinstance(output, list):
    ds4, ds3, ds2, ds1, final = output
```

### Custom Configuration

```python
from models.Exp.UTANetMamba.UTANetMamba import UTANetMamba

model = UTANetMamba(
    pretrained=True,
    topk=2,                    # Number of experts to select
    n_channels=3,
    n_classes=1,
    img_size=224,
    deep_supervision=True,
    cat_channels=32            # Decoder fusion channels
)
```

## Training with U-Bench Framework

```bash
python main.py --model UTANetMamba --dataset BUSI --batch_size 8
```

## Requirements

- PyTorch >= 1.10
- torchvision
- mamba-ssm (for Mamba experts)
- CUDA (recommended for Mamba)

Install Mamba:
```bash
pip install mamba-ssm
```

## Model Registration

The model is registered in the U-Bench framework:
- **Model Name**: `UTANetMamba`
- **Model ID**: 134
- **Deep Supervision**: Enabled
- **Factory Function**: `utanet_mamba()`

## Performance Characteristics

### Computational Efficiency
- **Sparse Top-K**: ~50% reduction in expert computation (Top-2 out of 4)
- **Memory Efficient**: Fast decoder design with reduced memory footprint
- **Mamba Experts**: Linear complexity O(N) for sequence modeling

### Model Capacity
- **Parameters**: ~25-30M (depends on configuration)
- **Experts**: 4 Mamba experts with independent gating
- **Receptive Field**: Global (thanks to Mamba SSM)

## Citation

If you use this model, please cite:

```bibtex
@article{utanet_mamba,
  title={UTANet+ Mamba: Sparse Mixture of Experts for Medical Image Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Comparison with Other Variants

| Model | Expert Type | Sparse Computation | Global Modeling |
|-------|-------------|-------------------|-----------------|
| UTANet | CNN | ✓ | ✗ |
| UTANetPlusHybrid | Mixed (CNN+LK+Mamba) | ✗ | ✓ |
| **UTANetMamba** | **Mamba Only** | **✓** | **✓** |

## References

1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
2. UTANet: Uncertainty-aware Transformer Attention Network
3. Mixture of Experts: Efficient Sparse Learning
