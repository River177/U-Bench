#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bingren 数据集清洗与转换脚本
功能：
1. 统一器官命名（修正拼写错误）
2. 筛选包含完整6个核心器官的病人
3. 将 DICOM CT + RTStruct 转换为 npz 格式
4. 划分 train/valid/test 集
"""

import os
import sys
import numpy as np
import pydicom
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区
# ============================================================

# 6个核心器官的标准命名（用于统一命名）
ORGAN_NAME_MAPPING = {
    # Bladder 膀胱
    'bladder': 'Bladder',
    'blader': 'Bladder',
    
    # Rectum 直肠
    'rectum': 'Rectum',
    'rectum1': 'Rectum',
    
    # Femur-R 右股骨头
    'femur-r': 'Femur_R',
    'femur-right': 'Femur_R',
    
    # Femur-L 左股骨头
    'femur-l': 'Femur_L',
    'femur-left': 'Femur_L',
    'remur-l': 'Femur_L',
    
    # Intestine 小肠
    'intestine': 'Intestine',
    'inestine': 'Intestine',
    
    # CE 宫颈
    'ce': 'CE',
}

# 6个核心器官（标准化后的名称）
CORE_ORGANS = ['Bladder', 'Rectum', 'Femur_R', 'Femur_L', 'Intestine', 'CE']

# 器官对应的标签值
ORGAN_LABELS = {
    'Bladder': 1,
    'Rectum': 2,
    'Femur_R': 3,
    'Femur_L': 4,
    'Intestine': 5,
    'CE': 6,
}

# 数据集划分比例
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# 工具函数
# ============================================================

def normalize_organ_name(name: str) -> Optional[str]:
    """标准化器官名称"""
    name_lower = name.lower().strip()
    return ORGAN_NAME_MAPPING.get(name_lower, None)


def get_contour_data(rtstruct, roi_number: int) -> List:
    """获取指定 ROI 的轮廓数据"""
    for roi_contour in rtstruct.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            if hasattr(roi_contour, 'ContourSequence'):
                return roi_contour.ContourSequence
    return []


def contour_to_mask(contour_data, ct_slices: List, image_shape: Tuple[int, int]) -> np.ndarray:
    """将轮廓数据转换为3D mask"""
    from matplotlib.path import Path
    
    # 创建空的3D mask
    num_slices = len(ct_slices)
    mask = np.zeros((num_slices, image_shape[0], image_shape[1]), dtype=np.uint8)
    
    # 获取每个切片的 z 位置和相关信息
    slice_info = {}
    for i, ct in enumerate(ct_slices):
        z_pos = float(ct.ImagePositionPatient[2])
        slice_info[i] = {
            'z': z_pos,
            'origin': [float(x) for x in ct.ImagePositionPatient],
            'spacing': [float(x) for x in ct.PixelSpacing],
        }
    
    # 处理每个轮廓
    for contour in contour_data:
        if not hasattr(contour, 'ContourData'):
            continue
        
        contour_points = np.array(contour.ContourData).reshape(-1, 3)
        contour_z = contour_points[0, 2]
        
        # 找到对应的切片
        slice_idx = None
        min_dist = float('inf')
        for idx, info in slice_info.items():
            dist = abs(info['z'] - contour_z)
            if dist < min_dist:
                min_dist = dist
                slice_idx = idx
        
        if slice_idx is None or min_dist > 5:  # 5mm 容差
            continue
        
        # 将物理坐标转换为像素坐标
        info = slice_info[slice_idx]
        pixel_coords = []
        for point in contour_points:
            col = (point[0] - info['origin'][0]) / info['spacing'][0]
            row = (point[1] - info['origin'][1]) / info['spacing'][1]
            pixel_coords.append([col, row])
        
        if len(pixel_coords) < 3:
            continue
        
        # 填充多边形
        pixel_coords = np.array(pixel_coords)
        try:
            path = Path(pixel_coords)
            x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
            points = np.vstack((x.flatten(), y.flatten())).T
            mask_slice = path.contains_points(points).reshape(image_shape)
            mask[slice_idx] |= mask_slice.astype(np.uint8)
        except Exception as e:
            continue
    
    return mask


def load_ct_series(ct_folder: str) -> Tuple[np.ndarray, List]:
    """加载 CT 序列"""
    dcm_files = [f for f in os.listdir(ct_folder) if f.endswith('.dcm')]
    
    ct_slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(os.path.join(ct_folder, f), force=True)
            if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                ct_slices.append(ds)
        except:
            continue
    
    if not ct_slices:
        return None, None
    
    # 按 z 位置排序
    ct_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 提取像素数据
    images = []
    for ct in ct_slices:
        pixel_array = ct.pixel_array.astype(np.float32)
        # 应用 rescale
        if hasattr(ct, 'RescaleSlope') and hasattr(ct, 'RescaleIntercept'):
            pixel_array = pixel_array * ct.RescaleSlope + ct.RescaleIntercept
        images.append(pixel_array)
    
    volume = np.stack(images, axis=0)
    return volume, ct_slices


def process_patient(patient_path: str, patient_name: str) -> Optional[Dict]:
    """处理单个病人数据"""
    ct_folder = None
    rtstruct_path = None
    
    print(f"    处理: {patient_name}...", end=' ', flush=True)
    
    # 查找 CT 和 RTStruct
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if not dcm_files:
            continue
        
        try:
            first_dcm = pydicom.dcmread(os.path.join(root, dcm_files[0]), force=True)
            modality = getattr(first_dcm, 'Modality', 'Unknown')
            
            if modality == 'CT' and ct_folder is None:
                ct_folder = root
            elif modality == 'RTSTRUCT' and rtstruct_path is None:
                rtstruct_path = os.path.join(root, dcm_files[0])
        except:
            continue
    
    if ct_folder is None or rtstruct_path is None:
        print(f"[跳过] 缺少CT或RTStruct", flush=True)
        return None
    
    # 加载 CT
    volume, ct_slices = load_ct_series(ct_folder)
    if volume is None:
        print(f"[跳过] 无法加载CT", flush=True)
        return None
    
    # 加载 RTStruct
    try:
        rtstruct = pydicom.dcmread(rtstruct_path, force=True)
    except Exception as e:
        print(f"[跳过] 无法加载RTStruct", flush=True)
        return None
    
    # 获取 ROI 信息
    roi_info = {}
    if hasattr(rtstruct, 'StructureSetROISequence'):
        for roi in rtstruct.StructureSetROISequence:
            roi_name = getattr(roi, 'ROIName', 'Unknown')
            roi_number = getattr(roi, 'ROINumber', -1)
            normalized = normalize_organ_name(roi_name)
            if normalized and normalized in CORE_ORGANS:
                roi_info[normalized] = roi_number
    
    # 检查是否包含所有6个核心器官
    missing = set(CORE_ORGANS) - set(roi_info.keys())
    if missing:
        print(f"[跳过] 缺少 {missing}", flush=True)
        return None
    
    # 生成标签 mask
    image_shape = (ct_slices[0].Rows, ct_slices[0].Columns)
    combined_mask = np.zeros((len(ct_slices), image_shape[0], image_shape[1]), dtype=np.uint8)
    
    for organ_name, roi_number in roi_info.items():
        contour_data = get_contour_data(rtstruct, roi_number)
        if contour_data:
            organ_mask = contour_to_mask(contour_data, ct_slices, image_shape)
            label_value = ORGAN_LABELS[organ_name]
            combined_mask[organ_mask > 0] = label_value
    
    print(f"[成功] CT {volume.shape}", flush=True)
    
    return {
        'name': patient_name,
        'volume': volume,
        'mask': combined_mask,
        'organs': list(roi_info.keys())
    }


def normalize_ct(volume: np.ndarray, window_center: float = 40, window_width: float = 400) -> np.ndarray:
    """CT 窗宽窗位归一化"""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    volume = np.clip(volume, min_val, max_val)
    volume = (volume - min_val) / (max_val - min_val)
    return volume.astype(np.float32)


def save_slices_as_npz(patient_data: Dict, output_dir: str, split: str) -> List[str]:
    """将病人数据保存为 npz 切片"""
    volume = patient_data['volume']
    mask = patient_data['mask']
    name = patient_data['name'].replace(' ', '_')
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    saved_files = []
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    for i in range(volume.shape[0]):
        # 跳过没有任何标注的切片（可选）
        if mask[i].max() == 0:
            continue
        
        slice_name = f"{name}_slice_{i:03d}.npz"
        slice_path = os.path.join(split_dir, slice_name)
        
        np.savez_compressed(
            slice_path,
            img=volume[i],
            label=mask[i]
        )
        saved_files.append(slice_name)
    
    return saved_files


def save_volume_as_npz(patient_data: Dict, output_dir: str, split: str) -> str:
    """将病人数据保存为 3D 体积 npz (用于测试)"""
    volume = patient_data['volume']
    mask = patient_data['mask']
    name = patient_data['name'].replace(' ', '_')
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    vol_name = f"{name}.npz"
    vol_path = os.path.join(split_dir, vol_name)
    
    np.savez_compressed(
        vol_path,
        img=volume,    # 形状: [slices, H, W]
        label=mask     # 形状: [slices, H, W]
    )
    
    return vol_name


def main():
    # 输入输出路径
    input_root = r"d:\曲线分割\U-Bench\data\bingren"
    output_root = r"d:\曲线分割\U-Bench\data\bingren_processed"
    
    print("="*70)
    print("bingren 数据集清洗与转换（边处理边保存模式）")
    print("="*70)
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"核心器官: {CORE_ORGANS}")
    print(f"划分比例: train={TRAIN_RATIO}, valid={VALID_RATIO}, test={TEST_RATIO}")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    lists_dir = os.path.join(output_root, 'lists_bingren')
    os.makedirs(lists_dir, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test'), exist_ok=True)
    
    # 先收集所有符合条件的病人路径
    valid_patient_paths = []
    
    for batch_folder in os.listdir(input_root):
        batch_path = os.path.join(input_root, batch_folder)
        if not os.path.isdir(batch_path):
            continue
        
        # 跳过 dfyr（无标注）
        if batch_folder == 'dfyr':
            continue
        
        # 找到病人目录
        patient_parent = batch_path
        subdirs = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        if len(subdirs) == 1 and subdirs[0] == batch_folder:
            patient_parent = os.path.join(batch_path, subdirs[0])
            subdirs = [d for d in os.listdir(patient_parent) if os.path.isdir(os.path.join(patient_parent, d))]
        
        for patient_name in subdirs:
            patient_path = os.path.join(patient_parent, patient_name)
            valid_patient_paths.append((patient_path, patient_name, batch_folder))
    
    # 随机打乱并划分
    import random
    random.seed(42)
    random.shuffle(valid_patient_paths)
    
    n_total = len(valid_patient_paths)
    n_train = int(n_total * TRAIN_RATIO)
    n_valid = int(n_total * VALID_RATIO)
    
    train_paths = valid_patient_paths[:n_train]
    valid_paths = valid_patient_paths[n_train:n_train+n_valid]
    test_paths = valid_patient_paths[n_train+n_valid:]
    
    print(f"\n预计划分: train={len(train_paths)}, valid={len(valid_paths)}, test={len(test_paths)}")
    
    train_files, valid_files, test_files = [], [], []
    success_count = 0
    
    def process_and_save(paths, split, file_list, save_3d=False):
        nonlocal success_count
        print(f"\n[处理 {split} 集...]")
        for patient_path, patient_name, batch in paths:
            try:
                result = process_patient(patient_path, patient_name)
                if result:
                    if save_3d:
                        # 测试集保存为 3D 体积
                        vol_name = save_volume_as_npz(result, output_root, split)
                        file_list.append(vol_name)
                    else:
                        # 训练/验证集保存为 2D 切片
                        files = save_slices_as_npz(result, output_root, split)
                        file_list.extend(files)
                    success_count += 1
                    # 立即释放内存
                    del result
                    import gc
                    gc.collect()
            except Exception as e:
                print(f"[错误] 跳过 {patient_name}: {e}", flush=True)
    
    process_and_save(train_paths, 'train', train_files, save_3d=False)
    process_and_save(valid_paths, 'valid', valid_files, save_3d=False)
    process_and_save(test_paths, 'test', test_files, save_3d=True)  # 测试集保存 3D
    
    # 保存文件列表
    with open(os.path.join(lists_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(os.path.join(lists_dir, 'valid.txt'), 'w') as f:
        f.write('\n'.join(valid_files))
    
    with open(os.path.join(lists_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))
    
    # 汇总
    print("\n" + "="*70)
    print("转换完成!")
    print("="*70)
    print(f"成功处理病人数: {success_count}")
    print(f"输出目录: {output_root}")
    print(f"  - train/: {len(train_files)} 个切片")
    print(f"  - valid/: {len(valid_files)} 个切片")
    print(f"  - test/: {len(test_files)} 个 3D 体积")
    print(f"  - lists_bingren/: train.txt, valid.txt, test.txt")
    print("\n器官标签映射:")
    for organ, label in ORGAN_LABELS.items():
        print(f"  {label}: {organ}")
    print("="*70)


if __name__ == "__main__":
    main()
