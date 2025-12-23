#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试多个模型在 busi 数据集上的表现
"""

import subprocess
import sys

# 要测试的模型列表
MODELS = [
    "MDSA_UNet",
    "UTANet",
    "SimpleUNet",
    "ResU_KAN",
    "MBSNet",
    "MSLAU_Net",
    "DDS_UNet",
    "ERDUnet",
    "LV_UNet",
    "DDANet",
    "MEGANet",
    "CMUNeXt",
    "Tinyunet",
    "RollingUnet",
    "MMUNet",
    "CSCAUNet",
    "EMCAD",
    "CSWin_UNet",
    "CFM_UNet",
    "Perspective_Unet",
]

# 配置
GPU = 0
BATCH_SIZE = 8
BASE_DIR = "./data/Database_134_Angiograms"
DATASET_NAME = "busi"

def run_test(model_name):
    """运行单个模型的测试"""
    cmd = [
        sys.executable, "main.py",
        "--gpu", str(GPU),
        "--batch_size", str(BATCH_SIZE),
        "--model", model_name,
        "--base_dir", BASE_DIR,
        "--dataset_name", DATASET_NAME,
        "--just_for_test", "True"
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {model_name} 测试完成")
            return True
        else:
            print(f"✗ {model_name} 测试失败 (return code: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ {model_name} 测试出错: {e}")
        return False

def main():
    print("="*60)
    print("批量模型测试 - BUSI 数据集")
    print("="*60)
    print(f"模型数量: {len(MODELS)}")
    print(f"GPU: {GPU}")
    print(f"数据集: {DATASET_NAME}")
    print(f"数据目录: {BASE_DIR}")
    
    success_count = 0
    failed_models = []
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] 开始测试 {model}")
        if run_test(model):
            success_count += 1
        else:
            failed_models.append(model)
    
    print("\n" + "="*60)
    print("测试完成汇总")
    print("="*60)
    print(f"成功: {success_count}/{len(MODELS)}")
    if failed_models:
        print(f"失败模型: {', '.join(failed_models)}")
    print(f"\n结果保存在: ./result/result_{DATASET_NAME}_test.csv")

if __name__ == "__main__":
    main()
