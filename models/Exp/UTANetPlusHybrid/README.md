# UTANet+ Hybrid

## Overview

UTANet+ Hybrid is an enhanced version of UTANet+ that incorporates Hybrid Mixture-of-Experts (MoE) with support for multiple expert types including CNN, Large Kernel, and **Mamba** (default). This model has been adapted to work seamlessly with the U-Bench framework.

## Key Features

- **Hybrid MoE Architecture**: Supports multiple expert types for flexible feature learning
  - **Mamba Experts** (default): Captures long-range dependencies using State Space Models (SSM)
  - **Large Kernel Experts**: Improves local connectivity with 7×7 depthwise convolutions
  - **CNN Experts**: Efficient standard channel mixing
  - **Hybrid Mode**: Combines all expert types for maximum flexibility
- **Memory Efficient**: Inherits the optimized design from UTANet+ Fast
- **Deep Supervision**: Built-in multi-scale supervision for better training
- **Dynamic Input Size**: Supports any input resolution (224×224, 256×256, etc.)

## Architecture

### Encoder
- **Backbone**: ResNet34 (pretrained on ImageNet)
- **Feature Extraction**: Multi-scale features at 5 levels

### TA-MoSC with Hybrid MoE
- **Fusion Module**: Combines multi-scale encoder features
- **MoE Routing**: Top-k expert selection (default k=2)
- **Expert Types**: Configurable via `expert_type` parameter
- **Docker Modules**: Route expert outputs to decoder levels

### Decoder
- **Fast Full-Scale Decoder**: Memory-efficient multi-scale fusion
- **Gated Attention**: Spatial and channel attention mechanisms
- **Connection Pruning**: Optimized skip connections to reduce memory

## U-Bench Integration

### Model Registration

The model is registered in the U-Bench framework with:
- **Model ID**: 132
- **Deep Supervision**: Enabled by default
- **Default Expert Type**: Mamba
- **Factory Function**: `utanet_plus_hybrid(input_channel, num_classes, pretrained)`

### Usage Example

```bash
# Training with U-Bench framework (Mamba experts by default)
python main.py \
    --model UTANetPlusHybrid \
    --base_dir ./data/busi \
    --dataset_name busi \
    --train_file_dir train.txt \
    --val_file_dir val.txt \
    --batch_size 8 \
    --max_epochs 100 \
    --base_lr 0.01 \
    --img_size 256 \
    --num_classes 1 \
    --input_channel 3 \
    --pretrained \
    --exp_name utanet_hybrid_mamba
```

### Parameters

- `input_channel` (int): Number of input channels (default: 3)
- `num_classes` (int): Number of output classes (default: 1)
- `pretrained` (bool): Whether to use TA-MoSC with MoE (default: True)

### Advanced Usage (Custom Expert Type)

To use different expert types, you can directly instantiate the model:

```python
from models.Exp.UTANetPlusHybrid.UTANetPlusHybrid import UTANetPlusHybrid

# Mamba experts (default)
model_mamba = UTANetPlusHybrid(
    n_channels=3,
    n_classes=1,
    pretrained=True,
    expert_type='mamba'
)

# Large Kernel experts
model_lk = UTANetPlusHybrid(
    n_channels=3,
    n_classes=1,
    pretrained=True,
    expert_type='large_kernel'
)

# CNN experts
model_cnn = UTANetPlusHybrid(
    n_channels=3,
    n_classes=1,
    pretrained=True,
    expert_type='cnn'
)

# Hybrid (all expert types)
model_hybrid = UTANetPlusHybrid(
    n_channels=3,
    n_classes=1,
    pretrained=True,
    expert_type='hybrid'
)
```

## Output Format

When `deep_supervision=True` (default):
- Returns a list: `[ds4_out, ds3_out, ds2_out, ds1_out, logits]`
- Each output is upsampled to the original input resolution
- Main output is `logits` (last element)

When `deep_supervision=False`:
- Returns single tensor: `logits`

## Expert Types Comparison

| Expert Type | Strengths | Use Case |
|------------|-----------|----------|
| **Mamba** (default) | Long-range dependencies, global context | Medical images with large structures |
| **Large Kernel** | Local connectivity, texture details | Images with fine-grained patterns |
| **CNN** | Efficient, fast inference | Resource-constrained environments |
| **Hybrid** | Maximum flexibility, best performance | When computational resources allow |

## Model Statistics

- **Parameters**: ~24M (with ResNet34 encoder + MoE)
- **Input Size**: Flexible (224×224, 256×256, etc.)
- **Output**: Binary or multi-class segmentation masks
- **Memory**: Optimized with connection pruning and reduced fusion channels

## Dependencies

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- mamba-ssm (for Mamba experts)
- Local modules: `ta_mosc.py`, `modules_fast.py`

## Notes

- **Default expert type is Mamba** for optimal long-range dependency modeling
- The model automatically enables deep supervision for U-Bench compatibility
- TA-MoSC module can be disabled by setting `pretrained=False`
- Mamba experts require `mamba-ssm` package and CUDA support
- All interpolation operations use `align_corners=True` for consistency

## Citation

If you use this model, please cite the original UTANet work and acknowledge the U-Bench framework.

## Troubleshooting

**Issue**: ImportError for mamba_ssm
- **Solution**: Install with `pip install mamba-ssm` or use `expert_type='cnn'` or `expert_type='large_kernel'`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use `expert_type='cnn'` for lower memory usage

**Issue**: Model not registered
- **Solution**: Ensure model is imported in `models/__init__.py` and listed in `model_id.json`
