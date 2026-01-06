#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：检查为什么某些病人缺少CT或RTStruct
"""

import os
import pydicom
from collections import defaultdict

def diagnose_patient(patient_path, patient_name):
    """诊断单个病人的数据情况"""
    print(f"\n{'='*70}")
    print(f"诊断病人: {patient_name}")
    print(f"路径: {patient_path}")
    print(f"{'='*70}")
    
    ct_folders = []
    rtstruct_files = []
    other_modalities = defaultdict(list)
    no_modality_files = []
    error_files = []
    
    # 遍历所有DICOM文件
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if not dcm_files:
            continue
        
        for dcm_file in dcm_files:
            file_path = os.path.join(root, dcm_file)
            try:
                ds = pydicom.dcmread(file_path, force=True)
                modality = getattr(ds, 'Modality', None)
                
                if modality == 'CT':
                    ct_folders.append(root)
                    print(f"  ✓ 找到CT: {root} ({len(dcm_files)} 个文件)")
                elif modality == 'RTSTRUCT':
                    rtstruct_files.append(file_path)
                    print(f"  ✓ 找到RTStruct: {file_path}")
                elif modality:
                    other_modalities[modality].append(file_path)
                    print(f"  ⚠ 其他模态 [{modality}]: {file_path}")
                else:
                    no_modality_files.append(file_path)
                    print(f"  ⚠ 无Modality字段: {file_path}")
            except Exception as e:
                error_files.append((file_path, str(e)))
                print(f"  ✗ 读取错误: {file_path} - {e}")
    
    # 汇总
    print(f"\n诊断结果:")
    print(f"  - CT文件夹数量: {len(set(ct_folders))}")
    if ct_folders:
        print(f"    CT文件夹: {set(ct_folders)}")
    else:
        print(f"    ❌ 未找到CT文件夹")
    
    print(f"  - RTStruct文件数量: {len(rtstruct_files)}")
    if rtstruct_files:
        for rt in rtstruct_files:
            print(f"    RTStruct: {rt}")
    else:
        print(f"    ❌ 未找到RTStruct文件")
    
    if other_modalities:
        print(f"  - 其他模态文件:")
        for mod, files in other_modalities.items():
            print(f"    [{mod}]: {len(files)} 个文件")
            if len(files) <= 3:
                for f in files:
                    print(f"      - {f}")
    
    if no_modality_files:
        print(f"  - 无Modality字段的文件: {len(no_modality_files)} 个")
    
    if error_files:
        print(f"  - 读取错误的文件: {len(error_files)} 个")
        for f, e in error_files[:3]:  # 只显示前3个
            print(f"      - {f}: {e}")
    
    # 结论
    print(f"\n结论:")
    if not ct_folders and not rtstruct_files:
        print(f"  ❌ 完全缺少CT和RTStruct数据")
    elif not ct_folders:
        print(f"  ❌ 缺少CT数据（但找到了RTStruct）")
        if other_modalities:
            print(f"  💡 提示: 发现了其他模态 [{', '.join(other_modalities.keys())}]，可能CT数据使用了不同的Modality值")
    elif not rtstruct_files:
        print(f"  ❌ 缺少RTStruct数据（但找到了CT）")
    else:
        print(f"  ✓ CT和RTStruct都存在，但可能缺少某些器官标注")


def main():
    input_root = r"d:\曲线分割\U-Bench\data\bingren"
    
    # 列出所有病人
    all_patients = []
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
            patient_path = os.path.join(patient_parent, patient_name)
            all_patients.append((patient_path, patient_name))
    
    print(f"找到 {len(all_patients)} 个病人")
    print("开始诊断前5个被跳过的病人...")
    
    # 诊断前几个病人
    count = 0
    for patient_path, patient_name in all_patients:
        # 快速检查是否缺少CT或RTStruct
        has_ct = False
        has_rtstruct = False
        
        for root, dirs, files in os.walk(patient_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if not dcm_files:
                continue
            
            try:
                first_dcm = pydicom.dcmread(os.path.join(root, dcm_files[0]), force=True)
                modality = getattr(first_dcm, 'Modality', 'Unknown')
                if modality == 'CT':
                    has_ct = True
                elif modality == 'RTSTRUCT':
                    has_rtstruct = True
            except:
                pass
        
        if not has_ct or not has_rtstruct:
            diagnose_patient(patient_path, patient_name)
            count += 1
            if count >= 5:  # 只诊断前5个
                break


if __name__ == "__main__":
    main()

