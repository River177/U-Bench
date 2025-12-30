#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bingren 测试集 3D 体积生成脚本
仅重新生成 test/ 目录的 3D 体积数据
"""

import os
import sys
import numpy as np
import pydicom
import random
import gc
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入原脚本的函数
sys.path.insert(0, os.path.dirname(__file__))
from convert_bingren_to_ubench import (
    CORE_ORGANS, ORGAN_LABELS, ORGAN_NAME_MAPPING,
    normalize_organ_name, get_contour_data, contour_to_mask,
    load_ct_series, process_patient, normalize_ct, 
    TRAIN_RATIO, VALID_RATIO
)

def save_volume_as_npz(patient_data: Dict, output_dir: str) -> str:
    """将病人数据保存为 3D 体积 npz"""
    volume = patient_data['volume']
    mask = patient_data['mask']
    name = patient_data['name'].replace(' ', '_')
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    os.makedirs(output_dir, exist_ok=True)
    
    vol_name = f"{name}.npz"
    vol_path = os.path.join(output_dir, vol_name)
    
    np.savez_compressed(
        vol_path,
        img=volume,    # 形状: [slices, H, W]
        label=mask     # 形状: [slices, H, W]
    )
    
    return vol_name


def main():
    input_root = r"d:\曲线分割\U-Bench\data\bingren"
    output_root = r"d:\曲线分割\U-Bench\data\bingren_processed"
    test_dir = os.path.join(output_root, 'test')
    lists_dir = os.path.join(output_root, 'lists_bingren')
    
    print("="*70)
    print("bingren 测试集 3D 体积生成")
    print("="*70)
    
    # 收集所有病人路径
    valid_patient_paths = []
    
    for batch_folder in os.listdir(input_root):
        batch_path = os.path.join(input_root, batch_folder)
        if not os.path.isdir(batch_path):
            continue
        if batch_folder == 'dfyr':
            continue
        
        patient_parent = batch_path
        subdirs = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        if len(subdirs) == 1 and subdirs[0] == batch_folder:
            patient_parent = os.path.join(batch_path, subdirs[0])
            subdirs = [d for d in os.listdir(patient_parent) if os.path.isdir(os.path.join(patient_parent, d))]
        
        for patient_name in subdirs:
            patient_path = os.path.join(patient_parent, patient_name)
            valid_patient_paths.append((patient_path, patient_name, batch_folder))
    
    # 使用相同的随机种子和划分
    random.seed(42)
    random.shuffle(valid_patient_paths)
    
    n_total = len(valid_patient_paths)
    n_train = int(n_total * TRAIN_RATIO)
    n_valid = int(n_total * VALID_RATIO)
    
    test_paths = valid_patient_paths[n_train + n_valid:]
    
    print(f"测试集病人数: {len(test_paths)}")
    print(f"输出目录: {test_dir}")
    print("="*70)
    
    # 处理测试集
    test_files = []
    success_count = 0
    
    print("\n[处理测试集...]")
    for patient_path, patient_name, batch in test_paths:
        try:
            result = process_patient(patient_path, patient_name)
            if result:
                vol_name = save_volume_as_npz(result, test_dir)
                test_files.append(vol_name)
                success_count += 1
                del result
                gc.collect()
        except Exception as e:
            print(f"[错误] 跳过 {patient_name}: {e}", flush=True)
    
    # 更新文件列表
    with open(os.path.join(lists_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)
    print(f"成功生成: {success_count} 个 3D 体积")
    print(f"文件列表: {os.path.join(lists_dir, 'test.txt')}")
    print("="*70)


if __name__ == "__main__":
    main()
