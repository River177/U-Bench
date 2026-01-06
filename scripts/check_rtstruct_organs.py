#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查RTStruct中的器官标注情况
"""

import os
import pydicom
from collections import defaultdict

# 6个核心器官的标准命名
CORE_ORGANS = ['Bladder', 'Rectum', 'Femur_R', 'Femur_L', 'Intestine', 'CE']

# 器官名称映射
ORGAN_NAME_MAPPING = {
    'bladder': 'Bladder',
    'blader': 'Bladder',
    'rectum': 'Rectum',
    'rectum1': 'Rectum',
    'femur-r': 'Femur_R',
    'femur-right': 'Femur_R',
    'femur-l': 'Femur_L',
    'femur-left': 'Femur_L',
    'remur-l': 'Femur_L',
    'intestine': 'Intestine',
    'inestine': 'Intestine',
    'ce': 'CE',
}

def normalize_organ_name(name: str):
    """标准化器官名称"""
    name_lower = name.lower().strip()
    return ORGAN_NAME_MAPPING.get(name_lower, None)

def check_patient_organs(patient_path, patient_name):
    """检查病人的器官标注"""
    print(f"\n{'='*70}")
    print(f"检查病人: {patient_name}")
    print(f"{'='*70}")
    
    # 查找RTStruct文件（遍历所有DICOM文件）
    rtstruct_path = None
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if not dcm_files:
            continue
        
        # 遍历所有DICOM文件，不只是第一个
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(os.path.join(root, dcm_file), force=True)
                modality = getattr(ds, 'Modality', 'Unknown')
                if modality == 'RTSTRUCT':
                    rtstruct_path = os.path.join(root, dcm_file)
                    break
            except:
                continue
        
        if rtstruct_path:
            break
    
    if not rtstruct_path:
        print("  ❌ 未找到RTStruct文件")
        return
    
    print(f"  RTStruct路径: {rtstruct_path}")
    
    # 读取RTStruct
    try:
        rtstruct = pydicom.dcmread(rtstruct_path, force=True)
    except Exception as e:
        print(f"  ❌ 无法读取RTStruct: {e}")
        return
    
    # 获取所有ROI
    all_rois = []
    matched_organs = {}
    unmatched_rois = []
    
    if hasattr(rtstruct, 'StructureSetROISequence'):
        print(f"\n  RTStruct中的所有ROI ({len(rtstruct.StructureSetROISequence)}个):")
        for roi in rtstruct.StructureSetROISequence:
            roi_name = getattr(roi, 'ROIName', 'Unknown')
            roi_number = getattr(roi, 'ROINumber', -1)
            all_rois.append((roi_number, roi_name))
            
            normalized = normalize_organ_name(roi_name)
            if normalized and normalized in CORE_ORGANS:
                matched_organs[normalized] = roi_name
                print(f"    ✓ [{roi_number}] {roi_name} -> {normalized}")
            else:
                unmatched_rois.append(roi_name)
                print(f"    ✗ [{roi_number}] {roi_name} (未匹配)")
    
    # 检查缺少的器官
    missing_organs = set(CORE_ORGANS) - set(matched_organs.keys())
    found_organs = set(matched_organs.keys())
    
    print(f"\n  匹配结果:")
    print(f"    ✓ 已匹配器官 ({len(found_organs)}/6): {sorted(found_organs)}")
    if missing_organs:
        print(f"    ❌ 缺少器官 ({len(missing_organs)}): {sorted(missing_organs)}")
    
    if unmatched_rois:
        print(f"\n  未匹配的ROI名称 ({len(unmatched_rois)}个):")
        for roi_name in unmatched_rois[:10]:  # 只显示前10个
            print(f"    - '{roi_name}'")
        if len(unmatched_rois) > 10:
            print(f"    ... 还有 {len(unmatched_rois) - 10} 个")
    
    # 建议
    print(f"\n  建议:")
    if missing_organs:
        print(f"    - 这些器官在RTStruct中不存在或名称不匹配")
        print(f"    - 可能需要检查原始标注或添加名称映射")
    else:
        print(f"    ✓ 所有必需器官都已找到")


def main():
    input_root = r"d:\曲线分割\U-Bench\data\bingren"
    
    # 检查几个被跳过的病人
    test_patients = [
        "DAI LI HUA",
        "DAI SU E", 
        "DUAN SHU XIA",
        "GENG HUA",
        "GUAN HONG XIA",
        "ZHAN XIU LI",  # 从之前的输出看这个也被跳过了
    ]
    
    for batch_folder in os.listdir(input_root):
        batch_path = os.path.join(input_root, batch_folder)
        if not os.path.isdir(batch_path) or batch_folder == 'dfyr':
            continue
        
        patient_parent = batch_path
        subdirs = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        if len(subdirs) == 1 and subdirs[0] == batch_folder:
            patient_parent = os.path.join(batch_path, subdirs[0])
            subdirs = [d for d in os.listdir(patient_parent) if os.path.isdir(os.path.join(patient_parent, d))]
        
        for patient_name in subdirs:
            if patient_name in test_patients:
                patient_path = os.path.join(patient_parent, patient_name)
                check_patient_organs(patient_path, patient_name)


if __name__ == "__main__":
    main()

