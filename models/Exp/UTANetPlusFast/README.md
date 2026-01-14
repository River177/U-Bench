# UTANet+ Fast

## Overview

UTANet+ Fast is an optimized version of UTANet+ designed for low VRAM usage while maintaining high segmentation performance. This model has been adapted to work seamlessly with the U-Bench framework.

## Key Features

- **Low VRAM Usage**: Reduced fusion channels (32 vs 64) and pruned long-range skip connections
- **Deep Supervision**: Built-in multi-scale supervision for better training
- **TA-MoSC Module**: Optional Mixture-of-Experts routing for enhanced feature learning
- **ResNet34 Encoder**: Pretrained backbone for robust feature extraction

## Architecture Optimizations

### Connection Pruning Strategy

1. **Decoder 4 (28×28)**: Fuses `o4`, `e5`, `o3` (pruned: `o1`, `o2`)
2. **Decoder 3 (56×56)**: Fuses `o3`, `d4_att`, `e5`, `o2` (pruned: `o1`)
3. **Decoder 2 (112×112)**: Fuses `o2`, `d3_att`, `d4_att`, `o1` (pruned: `e5`)
4. **Decoder 1 (224×224)**: Fuses `o1`, `d2_att`, `d3_att` (pruned: `d4_att`, `e5`)

### Memory Efficiency

- Fusion channels: 32 (reduced from 64)
- Eliminates expensive upsampling of deep features to high resolutions
- Maintains accuracy while significantly reducing memory footprint

## U-Bench Integration

### Model Registration

The model is registered in the U-Bench framework with:
- **Model ID**: 131
- **Deep Supervision**: Enabled by default
- **Factory Function**: `utanet_plus_fast(input_channel, num_classes, pretrained)`

### Usage Example

```bash
# Training with U-Bench framework
python main.py \
    --model UTANetPlusFast \
    --base_dir ./data/busi \
    --dataset_name busi \
    --train_file_dir train.txt \
    --val_file_dir val.txt \
    --batch_size 8 \
    --max_epochs 100 \
    --base_lr 0.01 \
    --img_size 224 \
    --num_classes 1 \
    --input_channel 3 \
    --pretrained \
    --exp_name utanet_fast_exp
```

### Parameters

- `input_channel` (int): Number of input channels (default: 3)
- `num_classes` (int): Number of output classes (default: 1)
- `pretrained` (bool): Whether to use TA-MoSC module (default: True)

## Output Format

When `deep_supervision=True` (default):
- Returns a list: `[ds4_out, ds3_out, ds2_out, ds1_out, logits]`
- Each output is at the original input resolution
- Main output is `logits` (last element)

When `deep_supervision=False`:
- Returns single tensor: `logits`

## Model Statistics

- **Parameters**: ~24M (with ResNet34 encoder)
- **Input Size**: 224×224 (configurable)
- **Output**: Binary or multi-class segmentation masks

## Dependencies

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- Local modules: `ta_mosc.py`, `modules_fast.py`

## Citation

If you use this model, please cite the original UTANet work and acknowledge the U-Bench framework.

## Notes

- The model automatically enables deep supervision for U-Bench compatibility
- TA-MoSC module can be disabled by setting `pretrained=False`
- Designed for medical image segmentation tasks
