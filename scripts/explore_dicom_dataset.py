#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探索 DICOM 数据集的脚本
用于查看数据集结构、CT 图像信息和 RTStruct 标注内容
"""

import os
import sys
import pydicom
from collections import defaultdict

def explore_patient_folder(patient_path):
    """探索单个病人文件夹"""
    print(f"\n{'='*60}")
    print(f"病人文件夹: {os.path.basename(patient_path)}")
    print('='*60)
    
    ct_info = None
    rtstruct_info = None
    
    for subfolder in os.listdir(patient_path):
        subfolder_path = os.path.join(patient_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
            
        # 查找 DICOM 文件
        dcm_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        
        if not dcm_files:
            continue
            
        # 读取第一个 DICOM 文件来判断类型
        first_dcm = pydicom.dcmread(os.path.join(subfolder_path, dcm_files[0]), force=True)
        modality = getattr(first_dcm, 'Modality', 'Unknown')
        
        if modality == 'CT':
            print(f"\n[CT 图像] 文件夹: {subfolder}")
            print(f"  - DICOM 文件数量: {len(dcm_files)}")
            
            # 读取更多 CT 信息
            try:
                print(f"  - 图像尺寸: {first_dcm.Rows} x {first_dcm.Columns}")
                if hasattr(first_dcm, 'PixelSpacing'):
                    print(f"  - 像素间距: {first_dcm.PixelSpacing}")
                if hasattr(first_dcm, 'SliceThickness'):
                    print(f"  - 层厚: {first_dcm.SliceThickness} mm")
                if hasattr(first_dcm, 'PatientID'):
                    print(f"  - 患者ID: {first_dcm.PatientID}")
                if hasattr(first_dcm, 'StudyDescription'):
                    print(f"  - 研究描述: {first_dcm.StudyDescription}")
            except Exception as e:
                print(f"  - 读取详细信息出错: {e}")
                
            ct_info = {
                'path': subfolder_path,
                'num_slices': len(dcm_files),
                'rows': getattr(first_dcm, 'Rows', None),
                'cols': getattr(first_dcm, 'Columns', None)
            }
            
        elif modality == 'RTSTRUCT':
            print(f"\n[RTStruct 标注] 文件夹: {subfolder}")
            print(f"  - DICOM 文件数量: {len(dcm_files)}")
            
            # 读取 RTStruct 信息
            try:
                rtstruct = pydicom.dcmread(os.path.join(subfolder_path, dcm_files[0]), force=True)
                
                if hasattr(rtstruct, 'StructureSetROISequence'):
                    print(f"\n  [标注的结构/器官列表]:")
                    roi_names = []
                    for i, roi in enumerate(rtstruct.StructureSetROISequence):
                        roi_name = getattr(roi, 'ROIName', f'Unknown_{i}')
                        roi_number = getattr(roi, 'ROINumber', i)
                        roi_names.append(roi_name)
                        print(f"    {roi_number}: {roi_name}")
                    
                    rtstruct_info = {
                        'path': os.path.join(subfolder_path, dcm_files[0]),
                        'roi_names': roi_names
                    }
                else:
                    print("  - 未找到 StructureSetROISequence")
                    
            except Exception as e:
                print(f"  - 读取 RTStruct 出错: {e}")
        else:
            print(f"\n[其他模态: {modality}] 文件夹: {subfolder}")
            print(f"  - DICOM 文件数量: {len(dcm_files)}")
    
    return ct_info, rtstruct_info


def main():
    # 数据集根目录
    dataset_root = r"d:\曲线分割\U-Bench\data\bingren2-15"
    
    if not os.path.exists(dataset_root):
        # 尝试 Linux 路径
        dataset_root = "/dev/shm/ubench/U-Bench/data/bingren5-11"
    
    if not os.path.exists(dataset_root):
        print(f"错误: 找不到数据集目录 {dataset_root}")
        sys.exit(1)
    
    print("="*60)
    print("DICOM 数据集探索工具")
    print("="*60)
    print(f"数据集路径: {dataset_root}")
    
    # 获取所有病人文件夹
    patient_folders = [f for f in os.listdir(dataset_root) 
                       if os.path.isdir(os.path.join(dataset_root, f))]
    
    print(f"病人数量: {len(patient_folders)}")
    print(f"病人列表: {patient_folders}")
    
    # 统计所有标注结构
    all_roi_names = defaultdict(int)
    all_patients_info = []
    
    for patient_folder in patient_folders:
        patient_path = os.path.join(dataset_root, patient_folder)
        ct_info, rtstruct_info = explore_patient_folder(patient_path)
        
        if rtstruct_info:
            for roi_name in rtstruct_info['roi_names']:
                all_roi_names[roi_name] += 1
        
        all_patients_info.append({
            'name': patient_folder,
            'ct_info': ct_info,
            'rtstruct_info': rtstruct_info
        })
    
    # 汇总信息
    print("\n" + "="*60)
    print("数据集汇总")
    print("="*60)
    
    print(f"\n总病人数: {len(patient_folders)}")
    
    print(f"\n[所有标注结构及出现次数]:")
    for roi_name, count in sorted(all_roi_names.items(), key=lambda x: -x[1]):
        print(f"  - {roi_name}: {count}/{len(patient_folders)} 个病人")
    
    # CT 图像统计
    ct_slices = [p['ct_info']['num_slices'] for p in all_patients_info if p['ct_info']]
    if ct_slices:
        print(f"\n[CT 图像统计]:")
        print(f"  - 切片数范围: {min(ct_slices)} - {max(ct_slices)}")
        print(f"  - 平均切片数: {sum(ct_slices)/len(ct_slices):.1f}")
    
    print("\n" + "="*60)
    print("探索完成!")
    print("="*60)


if __name__ == "__main__":
    main()
