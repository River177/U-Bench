#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速分析 bingren 数据集结构
"""
import os
import pydicom
from collections import defaultdict

def analyze_dataset(dataset_root):
    """分析整个数据集"""
    print("="*70)
    print("bingren 数据集结构分析")
    print("="*70)
    
    # 统计
    total_patients = 0
    all_roi_names = defaultdict(int)
    patients_with_complete_organs = []
    patients_missing_organs = []
    
    # 期望的6个器官（根据常见盆腔器官）
    expected_organs = set()
    
    for batch_folder in os.listdir(dataset_root):
        batch_path = os.path.join(dataset_root, batch_folder)
        if not os.path.isdir(batch_path):
            continue
            
        print(f"\n{'='*60}")
        print(f"批次文件夹: {batch_folder}")
        print("="*60)
        
        # 找到实际的病人父目录（有时候有嵌套）
        patient_parent = batch_path
        subdirs = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        
        # 如果只有一个子目录且名字相同，进入它
        if len(subdirs) == 1 and subdirs[0] == batch_folder:
            patient_parent = os.path.join(batch_path, subdirs[0])
            subdirs = [d for d in os.listdir(patient_parent) if os.path.isdir(os.path.join(patient_parent, d))]
        
        patient_folders = subdirs
        print(f"病人数量: {len(patient_folders)}")
        
        for patient_name in patient_folders[:3]:  # 每批只详细分析前3个
            patient_path = os.path.join(patient_parent, patient_name)
            total_patients += 1
            
            print(f"\n  病人: {patient_name}")
            
            # 递归找 DICOM 文件
            ct_count = 0
            rtstruct_path = None
            roi_names = []
            
            for root, dirs, files in os.walk(patient_path):
                dcm_files = [f for f in files if f.endswith('.dcm')]
                if not dcm_files:
                    continue
                
                try:
                    first_dcm = pydicom.dcmread(os.path.join(root, dcm_files[0]), force=True)
                    modality = getattr(first_dcm, 'Modality', 'Unknown')
                    
                    if modality == 'CT':
                        ct_count = len(dcm_files)
                        print(f"    [CT] {ct_count} 张切片")
                        if hasattr(first_dcm, 'Rows'):
                            print(f"         尺寸: {first_dcm.Rows}x{first_dcm.Columns}")
                    
                    elif modality == 'RTSTRUCT':
                        rtstruct = pydicom.dcmread(os.path.join(root, dcm_files[0]), force=True)
                        if hasattr(rtstruct, 'StructureSetROISequence'):
                            roi_names = [getattr(roi, 'ROIName', 'Unknown') 
                                        for roi in rtstruct.StructureSetROISequence]
                            print(f"    [RTStruct] 标注器官 ({len(roi_names)}个):")
                            for roi in roi_names:
                                print(f"         - {roi}")
                                all_roi_names[roi] += 1
                except Exception as e:
                    print(f"    读取出错: {e}")
        
        # 统计该批次剩余病人（只收集ROI信息）
        for patient_name in patient_folders[3:]:
            patient_path = os.path.join(patient_parent, patient_name)
            total_patients += 1
            
            for root, dirs, files in os.walk(patient_path):
                dcm_files = [f for f in files if f.endswith('.dcm')]
                if not dcm_files:
                    continue
                try:
                    first_dcm = pydicom.dcmread(os.path.join(root, dcm_files[0]), force=True)
                    modality = getattr(first_dcm, 'Modality', 'Unknown')
                    if modality == 'RTSTRUCT':
                        rtstruct = pydicom.dcmread(os.path.join(root, dcm_files[0]), force=True)
                        if hasattr(rtstruct, 'StructureSetROISequence'):
                            for roi in rtstruct.StructureSetROISequence:
                                roi_name = getattr(roi, 'ROIName', 'Unknown')
                                all_roi_names[roi_name] += 1
                except:
                    pass
    
    # 汇总
    print("\n" + "="*70)
    print("数据集汇总统计")
    print("="*70)
    print(f"\n总病人数: {total_patients}")
    
    print(f"\n[所有标注器官及出现次数] (共{len(all_roi_names)}种):")
    for roi_name, count in sorted(all_roi_names.items(), key=lambda x: -x[1]):
        completeness = f"{count}/{total_patients}" if total_patients > 0 else "N/A"
        marker = "★" if count >= total_patients * 0.8 else ""
        print(f"  {marker} {roi_name}: {completeness} ({count*100/total_patients:.1f}%)")
    
    # 找出最常见的6个器官
    print(f"\n[最常见的6个标注器官]:")
    top6 = sorted(all_roi_names.items(), key=lambda x: -x[1])[:6]
    for roi_name, count in top6:
        print(f"  - {roi_name}: {count}/{total_patients}")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)

if __name__ == "__main__":
    dataset_root = r"d:\曲线分割\U-Bench\data\bingren"
    analyze_dataset(dataset_root)
