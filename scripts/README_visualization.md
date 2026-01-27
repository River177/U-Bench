# 模型对比可视化脚本使用说明

## 功能说明

`visualize_model_comparison.py` 脚本用于生成多个模型在同一数据集上的预测结果对比图，包括：
- 原始图像
- Ground Truth
- 各个模型的预测结果

支持的数据集：XCAD, busi, arcade, xca_dataset

## 使用方法

### 基本用法

#### 处理指定图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name XCAD \
    --base_dir ./data/XCAD \
    --image_list "00018_33,00026_38,00035_45" \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp
```

#### 处理所有测试集图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name XCAD \
    --base_dir ./data/XCAD \
    --all_images \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp
```

### 参数说明

- `--dataset_name`: 数据集名称，可选值：XCAD, busi, arcade, xca_dataset
- `--base_dir`: 数据集根目录路径
- `--image_list`: 要可视化的图片名称列表，用逗号分隔（不含扩展名）。与--all_images二选一
- `--all_images`: 处理所有测试集图片（从val.txt或test.txt加载）。与--image_list二选一
- `--val_file_dir`: 验证集文件列表（默认：val.txt）
- `--max_images`: 最多处理的图片数量，用于测试（默认：None，处理全部）
- `--models`: 要对比的模型名称列表，用逗号分隔
- `--output_dir`: 输出目录（默认：./visualizations）
- `--exp_name`: 实验名称（默认：default_exp）
- `--img_size`: 图像大小（默认：256）
- `--gpu`: GPU设备（默认：0）
- `--checkpoint_dir`: 模型checkpoint基础目录（默认：./output）
- `--use_best`: 使用best模型而非final模型（默认：True）
- `--threshold`: 二值化阈值（默认：0.5）

## 使用示例

### 1. XCAD数据集

#### 指定图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name XCAD \
    --base_dir ./data/XCAD \
    --image_list "00018_33,00026_38" \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

#### 所有测试集图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name XCAD \
    --base_dir ./data/XCAD \
    --all_images \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

### 2. BUSI数据集

#### 指定图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name busi \
    --base_dir ./data/busi \
    --image_list "benign_001,benign_002,malignant_001" \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

#### 所有测试集图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name busi \
    --base_dir ./data/busi \
    --all_images \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

### 3. ARCADE数据集

#### 指定图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name arcade \
    --base_dir ./data/arcade \
    --image_list "image_001,image_002,image_003" \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

#### 所有测试集图片
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name arcade \
    --base_dir ./data/arcade \
    --all_images \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2,UTANetMamba_Ablation3,UTANetMamba_Ablation4" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

#### 限制处理数量（用于测试）
```bash
python scripts/visualize_model_comparison.py \
    --dataset_name arcade \
    --base_dir ./data/arcade \
    --all_images \
    --max_images 10 \
    --models "UTANetMamba,UTANetMamba_Ablation1" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

### 4. xca_dataset（旧格式）

```bash
python scripts/visualize_model_comparison.py \
    --dataset_name xca_dataset \
    --base_dir ./data/xca_dataset \
    --image_list "CVAI-1207/CVAI-1207LAO44_CRA29/00031.png,CVAI-1208/CVAI-1208RAO2_CAU30/00045.png" \
    --models "UTANetMamba,UTANetMamba_Ablation1" \
    --output_dir ./visualizations \
    --exp_name default_exp \
    --gpu 0
```

## 输出结果

脚本会在输出目录中生成以下内容：

```
visualizations/
├── XCAD/
│   └── default_exp/
│       ├── 00018_33_comparison.png          # 对比图
│       ├── 00026_38_comparison.png
│       └── individual/                       # 单独的结果图像
│           ├── 00018_33/
│           │   ├── original.png
│           │   ├── ground_truth.png
│           │   ├── UTANetMamba.png
│           │   ├── UTANetMamba_Ablation1.png
│           │   ├── UTANetMamba_Ablation2.png
│           │   ├── UTANetMamba_Ablation3.png
│           │   └── UTANetMamba_Ablation4.png
│           └── 00026_38/
│               └── ...
```

### 对比图说明

每张对比图包含：
1. **Original**: 原始输入图像
2. **Ground Truth**: 标注的真实分割结果
3. **模型1预测**: 第一个模型的预测结果
4. **模型2预测**: 第二个模型的预测结果
5. **...**: 其他模型的预测结果

所有图像横向排列，便于直观对比。

## 注意事项

1. **图像名称格式**：
   - XCAD: 直接使用文件名（不含.png），如 "00018_33"
   - busi/arcade: 直接使用文件名（不含.png），如 "benign_001"
   - xca_dataset: 使用完整路径格式，如 "CVAI-1207/CVAI-1207LAO44_CRA29/00031.png"

2. **模型checkpoint要求**：
   - 确保所有模型都已训练完成
   - checkpoint文件应位于 `{checkpoint_dir}/{model_name}/{dataset_name}/{exp_name}/checkpoint_best.pth`

3. **数据集路径要求**：
   - XCAD: 需要 `test/images` 和 `test/masks` 目录
   - busi/arcade: 需要 `images` 和 `masks/0` 目录
   - xca_dataset: 需要 `CVAI-*/images` 和 `CVAI-*/ground_truth` 目录

4. **GPU内存**：
   - 如果GPU内存不足，可以使用 `--gpu cpu` 在CPU上运行
   - 或者减少同时对比的模型数量

## 常见问题

**Q: 提示找不到checkpoint文件？**
A: 检查模型是否已训练，以及checkpoint_dir、dataset_name、exp_name是否正确。

**Q: 提示找不到图像文件？**
A: 检查base_dir路径是否正确，以及image_list中的图像名称格式是否符合数据集要求。

**Q: 如何批量处理多张图像？**
A: 在--image_list参数中用逗号分隔多个图像名称即可。

**Q: 可以对比不同实验的模型吗？**
A: 目前脚本假设所有模型使用相同的exp_name。如需对比不同实验，需要修改脚本或分别运行。
