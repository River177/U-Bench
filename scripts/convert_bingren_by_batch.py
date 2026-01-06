#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bingren 数据集清洗与转换脚本（按批次划分版本）
功能：
1. 统一器官命名（修正拼写错误）
2. 筛选包含完整6个核心器官的病人
3. 将 DICOM CT + RTStruct 转换为 npz 格式
4. 每个批次内部划分为 train/valid 集，方便分别训练和评估
"""

import os
import sys
import numpy as np
import pydicom
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import random
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

# 每个批次内部的训练/验证划分比例
TRAIN_RATIO = 0.8  # 80% 训练集
VALID_RATIO = 0.2  # 20% 验证集

# 要处理的批次列表（可以指定特定批次，或设置为 None 处理所有批次）
TARGET_BATCHES = ['bingren1-50', 'bingren2-15', 'bingren3-17', 'bingren4-13', 'bingren5-11']
# TARGET_BATCHES = None  # 设置为 None 则处理所有批次（除了 dfyr）

# ============================================================
# 工具函数（与原脚本相同）
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
    
    # 查找 CT 和 RTStruct（遍历所有DICOM文件，不只是第一个）
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if not dcm_files:
            continue
        
        # 遍历所有DICOM文件以找到CT和RTStruct
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(os.path.join(root, dcm_file), force=True)
                modality = getattr(ds, 'Modality', 'Unknown')
                
                if modality == 'CT' and ct_folder is None:
                    ct_folder = root
                elif modality == 'RTSTRUCT' and rtstruct_path is None:
                    rtstruct_path = os.path.join(root, dcm_file)
                
                # 如果两者都找到了，可以提前退出
                if ct_folder and rtstruct_path:
                    break
            except:
                continue
        
        # 如果两者都找到了，可以提前退出外层循环
        if ct_folder and rtstruct_path:
            break
    
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


def save_slices_as_npz(patient_data: Dict, output_dir: str, split: str, batch_name: str) -> List[str]:
    """将病人数据保存为 npz 切片（添加批次前缀）"""
    volume = patient_data['volume']
    mask = patient_data['mask']
    name = patient_data['name'].replace(' ', '_')
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    saved_files = []
    split_dir = os.path.join(output_dir, batch_name, split)
    os.makedirs(split_dir, exist_ok=True)
    
    for i in range(volume.shape[0]):
        # 跳过没有任何标注的切片（可选）
        if mask[i].max() == 0:
            continue
        
        # 添加批次前缀，避免不同批次间的文件名冲突
        slice_name = f"{batch_name}_{name}_slice_{i:03d}.npz"
        slice_path = os.path.join(split_dir, slice_name)
        
        np.savez_compressed(
            slice_path,
            img=volume[i],
            label=mask[i]
        )
        saved_files.append(slice_name)
    
    return saved_files


def save_volume_as_npz(patient_data: Dict, output_dir: str, split: str, batch_name: str) -> str:
    """将病人数据保存为 3D 体积 npz (用于验证/测试)"""
    volume = patient_data['volume']
    mask = patient_data['mask']
    name = patient_data['name'].replace(' ', '_')
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    split_dir = os.path.join(output_dir, batch_name, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # 添加批次前缀，避免不同批次间的文件名冲突
    vol_name = f"{batch_name}_{name}.npz"
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
    output_root = r"d:\曲线分割\U-Bench\data\bingren_processed_by_batch"
    
    print("="*70)
    print("bingren 数据集清洗与转换（按批次划分版本）")
    print("="*70)
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"核心器官: {CORE_ORGANS}")
    print(f"每个批次划分比例: train={TRAIN_RATIO}, valid={VALID_RATIO}")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 获取要处理的批次列表
    if TARGET_BATCHES is None:
        batch_folders = [d for d in os.listdir(input_root) 
                        if os.path.isdir(os.path.join(input_root, d)) and d != 'dfyr']
    else:
        batch_folders = [d for d in TARGET_BATCHES 
                        if os.path.isdir(os.path.join(input_root, d))]
    
    print(f"\n将处理以下批次: {batch_folders}")
    print("="*70)
    
    # 设置随机种子，确保可重复性
    random.seed(42)
    
    # 统计信息
    total_stats = {
        'batches_processed': 0,
        'total_patients': 0,
        'total_train_slices': 0,
        'total_valid_slices': 0
    }
    
    # 为每个批次单独处理
    for batch_folder in batch_folders:
        batch_path = os.path.join(input_root, batch_folder)
        if not os.path.isdir(batch_path):
            print(f"\n[跳过] 批次 {batch_folder} 不存在")
            continue
        
        print(f"\n{'='*70}")
        print(f"处理批次: {batch_folder}")
        print(f"{'='*70}")
        
        # 收集该批次的所有病人路径
        patient_paths = []
        
        # 找到病人目录
        patient_parent = batch_path
        subdirs = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        
        # 处理嵌套目录结构（如 bingren2-15/bingren2-15/）
        if len(subdirs) == 1 and subdirs[0] == batch_folder:
            patient_parent = os.path.join(batch_path, subdirs[0])
            subdirs = [d for d in os.listdir(patient_parent) if os.path.isdir(os.path.join(patient_parent, d))]
        
        for patient_name in subdirs:
            patient_path = os.path.join(patient_parent, patient_name)
            patient_paths.append((patient_path, patient_name))
        
        if not patient_paths:
            print(f"[跳过] 批次 {batch_folder} 没有找到病人数据")
            continue
        
        # 随机打乱该批次的病人列表
        random.shuffle(patient_paths)
        
        # 划分训练集和验证集
        n_total = len(patient_paths)
        n_train = int(n_total * TRAIN_RATIO)
        
        train_paths = patient_paths[:n_train]
        valid_paths = patient_paths[n_train:]
        
        print(f"批次 {batch_folder}: 共 {n_total} 个病人")
        print(f"  - 训练集: {len(train_paths)} 个病人")
        print(f"  - 验证集: {len(valid_paths)} 个病人")
        
        # 创建该批次的输出目录
        batch_output_dir = os.path.join(output_root, batch_folder)
        batch_lists_dir = os.path.join(batch_output_dir, 'lists_bingren')
        os.makedirs(batch_lists_dir, exist_ok=True)
        os.makedirs(os.path.join(batch_output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(batch_output_dir, 'valid'), exist_ok=True)
        
        train_files, valid_files = [], []
        batch_success_count = 0
        
        def process_and_save(paths, split, file_list, save_3d=False):
            nonlocal batch_success_count
            print(f"\n[处理 {batch_folder} 的 {split} 集...]")
            for patient_path, patient_name in paths:
                try:
                    result = process_patient(patient_path, patient_name)
                    if result:
                        if save_3d:
                            # 验证集保存为 3D 体积（用于测试）
                            vol_name = save_volume_as_npz(result, output_root, split, batch_folder)
                            file_list.append(vol_name)
                        else:
                            # 训练集保存为 2D 切片
                            files = save_slices_as_npz(result, output_root, split, batch_folder)
                            file_list.extend(files)
                        batch_success_count += 1
                        # 立即释放内存
                        del result
                        import gc
                        gc.collect()
                except Exception as e:
                    print(f"[错误] 跳过 {patient_name}: {e}", flush=True)
        
        process_and_save(train_paths, 'train', train_files, save_3d=False)
        process_and_save(valid_paths, 'valid', valid_files, save_3d=True)  # 验证集保存为 3D
        
        # 保存该批次的文件列表
        train_list_path = os.path.join(batch_lists_dir, 'train.txt')
        valid_list_path = os.path.join(batch_lists_dir, 'valid.txt')
        
        with open(train_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_files))
        
        with open(valid_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_files))
        
        # 打印该批次的统计信息
        print(f"\n批次 {batch_folder} 处理完成:")
        print(f"  - 成功处理病人数: {batch_success_count}")
        print(f"  - 训练集切片数: {len(train_files)}")
        print(f"  - 验证集3D体积数: {len(valid_files)}")
        print(f"  - 文件列表保存在: {batch_lists_dir}")
        
        # 更新总统计
        total_stats['batches_processed'] += 1
        total_stats['total_patients'] += batch_success_count
        total_stats['total_train_slices'] += len(train_files)
        total_stats['total_valid_slices'] += len(valid_files)
    
    # 打印总体统计信息
    print("\n" + "="*70)
    print("所有批次处理完成!")
    print("="*70)
    print(f"处理批次数: {total_stats['batches_processed']}")
    print(f"总成功处理病人数: {total_stats['total_patients']}")
    print(f"总训练集切片数: {total_stats['total_train_slices']}")
    print(f"总验证集3D体积数: {total_stats['total_valid_slices']}")
    print(f"\n输出目录结构:")
    print(f"  {output_root}/")
    for batch_folder in batch_folders:
        print(f"    {batch_folder}/")
        print(f"      train/  (训练集2D切片)")
        print(f"      valid/  (验证集3D体积，可用于测试)")
        print(f"      lists_bingren/")
        print(f"        train.txt  (训练集文件列表)")
        print(f"        valid.txt  (验证集文件列表)")
    print("\n器官标签映射:")
    for organ, label in ORGAN_LABELS.items():
        print(f"  {label}: {organ}")
    print("\n使用说明:")
    print("  每个批次都有独立的 train/valid 划分，可以分别训练和评估")
    print("  训练时指定对应的批次目录即可，例如:")
    print(f"    --base_dir {output_root}/bingren1-50")
    print("\n测试说明:")
    print("  验证集已保存为3D体积格式，可以直接用于测试")
    print("  测试时使用 valid 集，例如:")
    print(f"    python main.py --base_dir {output_root}/bingren1-50 --dataset_name bingren --just_for_test")
    print("  或者使用 main_multi3d.py:")
    print(f"    python main_multi3d.py --base_dir {output_root}/bingren1-50 --dataset_name bingren --just_for_test")
    print("="*70)


if __name__ == "__main__":
    main()

