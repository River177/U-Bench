"""
测试 XCA 数据集加载器是否能正确加载CATH版本的标注
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataset_xca import XCADataset
from albumentations import Compose, Resize, Normalize
import numpy as np

# 测试一个样本
base_dir = "data/xca_dataset"
sample_id = "CVAI-1207/CVAI-1207LAO44_CRA29/00031.png"

transform = Compose([
    Resize(512, 512),
    Normalize(),
])

dataset = XCADataset(
    base_dir=base_dir,
    mode="train",
    transform=transform
)

# 找到这个样本的索引
try:
    idx = dataset.sample_list.index(sample_id)
    print(f"Found sample at index {idx}: {sample_id}")
    
    # 加载样本
    sample = dataset[idx]
    
    print(f"\nSample loaded successfully!")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label shape: {sample['label'].shape}")
    
    # 检查标注值
    label_values = np.unique(sample['label'])
    print(f"  Label unique values: {label_values}")
    print(f"  Label value counts:")
    for val in label_values:
        count = np.sum(sample['label'] == val)
        print(f"    {val}: {count} pixels")
    
    # 检查是否使用了CATH版本
    parts = sample_id.split('/')
    case_id, sequence_id, frame_id = parts
    gt_path_cath = os.path.join(base_dir, case_id, 'ground_truth', sequence_id + 'CATH', frame_id)
    gt_path_normal = os.path.join(base_dir, case_id, 'ground_truth', sequence_id, frame_id)
    
    print(f"\nAnnotation paths:")
    print(f"  CATH version: {gt_path_cath}")
    print(f"    Exists: {os.path.exists(gt_path_cath)}")
    print(f"  Normal version: {gt_path_normal}")
    print(f"    Exists: {os.path.exists(gt_path_normal)}")
    
    if os.path.exists(gt_path_cath):
        import cv2
        gt_cath = cv2.imread(gt_path_cath, cv2.IMREAD_GRAYSCALE)
        print(f"\nCATH annotation values: {np.unique(gt_cath)}")
        print("✓ Successfully using CATH version!")
    else:
        print("✗ CATH version not found!")
        
except ValueError:
    print(f"Sample {sample_id} not found in dataset")
    print(f"Total samples: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First sample: {dataset.sample_list[0]}")

