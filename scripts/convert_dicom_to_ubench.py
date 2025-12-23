#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 DICOM + RTStruct 数据集转换为 U-Bench 格式
分割目标：
  0: 背景
  1: Bladder（膀胱）
  2: Rectum（直肠）
  3: Femur-L（左股骨头）
  4: Femur-R（右股骨头）
"""

import os
import sys
import numpy as np
import pydicom
from collections import defaultdict
from scipy.ndimage import zoom
import random

# 分割目标映射（6个目标 + 背景 = 7类）
ROI_MAPPING = {
    'Bladder': 1,
    'Rectum': 2,
    'Femur-L': 3,
    'Femur-R': 4,
    'CE': 5,
    'Intestine': 6,
}

# 输出图像尺寸
OUTPUT_SIZE = 256

# 数据划分比例
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1


def load_ct_series(ct_folder):
    """加载 CT 序列"""
    dcm_files = [f for f in os.listdir(ct_folder) if f.endswith('.dcm')]
    slices = []
    
    for dcm_file in dcm_files:
        dcm_path = os.path.join(ct_folder, dcm_file)
        ds = pydicom.dcmread(dcm_path, force=True)
        slices.append(ds)
    
    # 按 InstanceNumber 或 SliceLocation 排序
    try:
        slices.sort(key=lambda x: float(x.InstanceNumber))
    except:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except:
            pass
    
    # 构建 3D 体积
    volume = []
    for s in slices:
        pixel_array = s.pixel_array.astype(np.float32)
        # 应用 RescaleSlope 和 RescaleIntercept 转换为 HU 值
        if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
            pixel_array = pixel_array * s.RescaleSlope + s.RescaleIntercept
        volume.append(pixel_array)
    
    volume = np.stack(volume, axis=0)  # [D, H, W]
    
    # 获取空间信息
    spacing = [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1])]
    
    # 获取 ImagePositionPatient 用于后续坐标转换
    positions = []
    for s in slices:
        if hasattr(s, 'ImagePositionPatient'):
            positions.append([float(p) for p in s.ImagePositionPatient])
    
    return volume, slices, spacing, positions


def load_rtstruct(rtstruct_path):
    """加载 RTStruct 文件"""
    ds = pydicom.dcmread(rtstruct_path, force=True)
    
    # 获取 ROI 名称和编号的映射
    roi_info = {}
    if hasattr(ds, 'StructureSetROISequence'):
        for roi in ds.StructureSetROISequence:
            roi_number = roi.ROINumber
            roi_name = roi.ROIName
            roi_info[roi_number] = roi_name
    
    # 获取轮廓数据
    contours = defaultdict(list)
    if hasattr(ds, 'ROIContourSequence'):
        for roi_contour in ds.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            roi_name = roi_info.get(roi_number, f'Unknown_{roi_number}')
            
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    if hasattr(contour, 'ContourData'):
                        contour_data = contour.ContourData
                        # 轮廓数据是 [x1,y1,z1, x2,y2,z2, ...] 格式
                        points = np.array(contour_data).reshape(-1, 3)
                        contours[roi_name].append(points)
    
    return contours


def contour_to_mask(contours, ct_slices, volume_shape, spacing, positions):
    """将轮廓转换为分割掩码"""
    mask = np.zeros(volume_shape, dtype=np.uint8)
    
    # 获取第一个切片的参考信息
    ref_slice = ct_slices[0]
    
    # ImageOrientationPatient
    if hasattr(ref_slice, 'ImageOrientationPatient'):
        orientation = [float(o) for o in ref_slice.ImageOrientationPatient]
        row_cosine = np.array(orientation[:3])
        col_cosine = np.array(orientation[3:])
    else:
        row_cosine = np.array([1, 0, 0])
        col_cosine = np.array([0, 1, 0])
    
    # 为每个切片创建 z 坐标到索引的映射
    z_to_idx = {}
    for idx, pos in enumerate(positions):
        z = pos[2]
        z_to_idx[round(z, 2)] = idx
    
    # 处理每个 ROI
    for roi_name, contour_list in contours.items():
        if roi_name not in ROI_MAPPING:
            continue
        
        label_value = ROI_MAPPING[roi_name]
        
        for points in contour_list:
            if len(points) < 3:
                continue
            
            # 获取该轮廓的 z 坐标
            z_coord = points[0, 2]
            
            # 找到对应的切片索引
            slice_idx = None
            min_diff = float('inf')
            for idx, pos in enumerate(positions):
                diff = abs(pos[2] - z_coord)
                if diff < min_diff:
                    min_diff = diff
                    slice_idx = idx
            
            if slice_idx is None or min_diff > 5:  # 容差 5mm
                continue
            
            # 获取该切片的 ImagePositionPatient
            slice_pos = np.array(positions[slice_idx])
            
            # 将物理坐标转换为像素坐标
            pixel_coords = []
            for point in points:
                # 计算相对于切片原点的偏移
                offset = point[:2] - slice_pos[:2]
                # 转换为像素坐标
                col = offset[0] / spacing[0]
                row = offset[1] / spacing[1]
                pixel_coords.append([row, col])
            
            pixel_coords = np.array(pixel_coords)
            
            # 使用多边形填充
            from skimage.draw import polygon
            rr, cc = polygon(pixel_coords[:, 0], pixel_coords[:, 1], shape=mask.shape[1:])
            
            # 确保索引在有效范围内
            valid_mask = (rr >= 0) & (rr < mask.shape[1]) & (cc >= 0) & (cc < mask.shape[2])
            rr = rr[valid_mask]
            cc = cc[valid_mask]
            
            mask[slice_idx, rr, cc] = label_value
    
    return mask


def normalize_ct(volume, window_center=-400, window_width=1500):
    """CT 窗口化和归一化"""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    volume = np.clip(volume, min_val, max_val)
    volume = (volume - min_val) / (max_val - min_val)
    
    return volume


def resize_slice(image, label, output_size):
    """调整切片大小"""
    h, w = image.shape
    scale_h = output_size / h
    scale_w = output_size / w
    
    # 图像使用双线性插值
    image_resized = zoom(image, (scale_h, scale_w), order=1)
    # 标签使用最近邻插值
    label_resized = zoom(label, (scale_h, scale_w), order=0)
    
    return image_resized, label_resized


def process_patient_slices(patient_path, output_dir, patient_name, split):
    """处理单个病人的数据，保存为 2D 切片格式（用于 train/valid）"""
    print(f"  处理病人: {patient_name}")
    
    ct_folder = None
    rtstruct_path = None
    
    # 查找 CT 和 RTStruct 文件夹
    for subfolder in os.listdir(patient_path):
        subfolder_path = os.path.join(patient_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        dcm_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        if not dcm_files:
            continue
        
        first_dcm = pydicom.dcmread(os.path.join(subfolder_path, dcm_files[0]), force=True)
        modality = getattr(first_dcm, 'Modality', 'Unknown')
        
        if modality == 'CT':
            ct_folder = subfolder_path
        elif modality == 'RTSTRUCT':
            rtstruct_path = os.path.join(subfolder_path, dcm_files[0])
    
    if ct_folder is None or rtstruct_path is None:
        print(f"    警告: 缺少 CT 或 RTStruct 数据，跳过")
        return []
    
    # 加载数据
    print(f"    加载 CT 数据...")
    volume, ct_slices, spacing, positions = load_ct_series(ct_folder)
    
    print(f"    加载 RTStruct...")
    contours = load_rtstruct(rtstruct_path)
    
    print(f"    生成分割掩码...")
    mask = contour_to_mask(contours, ct_slices, volume.shape, spacing, positions)
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    # 创建输出目录
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # 保存每个切片
    slice_names = []
    for i in range(volume.shape[0]):
        img_slice = volume[i]
        label_slice = mask[i]
        
        # 调整大小
        img_resized, label_resized = resize_slice(img_slice, label_slice, OUTPUT_SIZE)
        
        # 保存为 npz
        slice_name = f"{patient_name}_slice{i:03d}.npz"
        save_path = os.path.join(split_dir, slice_name)
        np.savez(save_path, img=img_resized.astype(np.float32), label=label_resized.astype(np.uint8))
        
        slice_names.append(slice_name)
    
    print(f"    保存了 {len(slice_names)} 个切片")
    return slice_names


def process_patient_volume(patient_path, output_dir, patient_name):
    """处理单个病人的数据，保存为 3D 体积格式（用于 test）"""
    print(f"  处理病人: {patient_name}")
    
    ct_folder = None
    rtstruct_path = None
    
    # 查找 CT 和 RTStruct 文件夹
    for subfolder in os.listdir(patient_path):
        subfolder_path = os.path.join(patient_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        dcm_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        if not dcm_files:
            continue
        
        first_dcm = pydicom.dcmread(os.path.join(subfolder_path, dcm_files[0]), force=True)
        modality = getattr(first_dcm, 'Modality', 'Unknown')
        
        if modality == 'CT':
            ct_folder = subfolder_path
        elif modality == 'RTSTRUCT':
            rtstruct_path = os.path.join(subfolder_path, dcm_files[0])
    
    if ct_folder is None or rtstruct_path is None:
        print(f"    警告: 缺少 CT 或 RTStruct 数据，跳过")
        return None
    
    # 加载数据
    print(f"    加载 CT 数据...")
    volume, ct_slices, spacing, positions = load_ct_series(ct_folder)
    
    print(f"    加载 RTStruct...")
    contours = load_rtstruct(rtstruct_path)
    
    print(f"    生成分割掩码...")
    mask = contour_to_mask(contours, ct_slices, volume.shape, spacing, positions)
    
    # 归一化 CT
    volume = normalize_ct(volume)
    
    # 创建输出目录
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # 调整每个切片大小并堆叠为 3D 体积
    volume_resized = []
    mask_resized = []
    for i in range(volume.shape[0]):
        img_slice = volume[i]
        label_slice = mask[i]
        img_r, label_r = resize_slice(img_slice, label_slice, OUTPUT_SIZE)
        volume_resized.append(img_r)
        mask_resized.append(label_r)
    
    volume_resized = np.stack(volume_resized, axis=0)  # [D, H, W]
    mask_resized = np.stack(mask_resized, axis=0)  # [D, H, W]
    
    # 保存为 3D 体积 npz
    vol_name = f"{patient_name}.npz"
    save_path = os.path.join(test_dir, vol_name)
    np.savez(save_path, img=volume_resized.astype(np.float32), label=mask_resized.astype(np.uint8))
    
    print(f"    保存了 3D 体积: {vol_name}, shape: {volume_resized.shape}")
    return vol_name


def main():
    # 数据集路径（可修改为其他数据集）
    dataset_root = r"d:\曲线分割\U-Bench\data\bingren2-15"
    output_root = r"d:\曲线分割\U-Bench\data\pelvis2_ACDC"
    
    if not os.path.exists(dataset_root):
        print(f"错误: 找不到数据集目录 {dataset_root}")
        sys.exit(1)
    
    print("="*60)
    print("DICOM 数据集转换工具")
    print("="*60)
    print(f"输入目录: {dataset_root}")
    print(f"输出目录: {output_root}")
    print(f"分割目标: {ROI_MAPPING}")
    print(f"输出尺寸: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    
    # 获取所有病人
    patient_folders = [f for f in os.listdir(dataset_root) 
                       if os.path.isdir(os.path.join(dataset_root, f))]
    
    print(f"\n病人数量: {len(patient_folders)}")
    
    # 随机划分数据集
    random.seed(42)
    random.shuffle(patient_folders)
    
    n_train = int(len(patient_folders) * TRAIN_RATIO)
    n_valid = int(len(patient_folders) * VALID_RATIO)
    
    train_patients = patient_folders[:n_train]
    valid_patients = patient_folders[n_train:n_train+n_valid]
    test_patients = patient_folders[n_train+n_valid:]
    
    print(f"训练集: {len(train_patients)} 个病人")
    print(f"验证集: {len(valid_patients)} 个病人")
    print(f"测试集: {len(test_patients)} 个病人")
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    lists_dir = os.path.join(output_root, "lists_ACDC")
    os.makedirs(lists_dir, exist_ok=True)
    
    # 处理每个数据集
    train_slices = []
    valid_slices = []
    test_slices = []
    
    print("\n" + "="*60)
    print("处理训练集（2D 切片格式）")
    print("="*60)
    for patient in train_patients:
        patient_path = os.path.join(dataset_root, patient)
        slices = process_patient_slices(patient_path, output_root, patient.replace(' ', '_'), 'train')
        train_slices.extend(slices)
    
    print("\n" + "="*60)
    print("处理验证集（2D 切片格式）")
    print("="*60)
    for patient in valid_patients:
        patient_path = os.path.join(dataset_root, patient)
        slices = process_patient_slices(patient_path, output_root, patient.replace(' ', '_'), 'valid')
        valid_slices.extend(slices)
    
    print("\n" + "="*60)
    print("处理测试集（3D 体积格式）")
    print("="*60)
    for patient in test_patients:
        patient_path = os.path.join(dataset_root, patient)
        vol_name = process_patient_volume(patient_path, output_root, patient.replace(' ', '_'))
        if vol_name:
            test_slices.append(vol_name)
    
    # 保存列表文件
    with open(os.path.join(lists_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_slices))
    
    with open(os.path.join(lists_dir, 'valid.txt'), 'w') as f:
        f.write('\n'.join(valid_slices))
    
    with open(os.path.join(lists_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_slices))
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"训练集切片数: {len(train_slices)}")
    print(f"验证集切片数: {len(valid_slices)}")
    print(f"测试集切片数: {len(test_slices)}")
    print(f"\n输出目录结构:")
    print(f"  {output_root}/")
    print(f"    ├── lists_ACDC/")
    print(f"    │   ├── train.txt")
    print(f"    │   ├── valid.txt")
    print(f"    │   └── test.txt")
    print(f"    ├── train/     (2D 切片: patient_sliceXXX.npz)")
    print(f"    ├── valid/     (2D 切片: patient_sliceXXX.npz)")
    print(f"    └── test/      (3D 体积: patient.npz)")
    print(f"\n使用命令:")
    print(f"  python main_multi3d.py --model VMUNet --base_dir ./data/pelvis_ACDC --dataset_name pelvis --num_classes 5 --input_channel 3 --val_interval 10 --gpu 1")


if __name__ == "__main__":
    main()
