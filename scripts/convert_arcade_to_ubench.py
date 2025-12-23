#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 ARCADE 数据集（COCO JSON 格式）转换为 U-Bench 格式
ARCADE: Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw

# 配置
ARCADE_ROOT = r"d:\曲线分割\U-Bench\data\arcade"
OUTPUT_ROOT = r"d:\曲线分割\U-Bench\data\arcade"  # 使用 busi 兼容格式
TASK = "syntax"  # 血管分割任务


def polygon_to_mask(segmentation, height, width):
    """将多边形标注转换为二值 mask（纯 Python 实现）"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for seg in segmentation:
        if isinstance(seg, list):
            # 多边形格式: [x1, y1, x2, y2, ...]
            # 转换为 [(x1, y1), (x2, y2), ...] 格式
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
    
    return np.array(mask, dtype=np.uint8)


def process_split(task_dir, split, output_root):
    """处理单个数据划分，输出为 busi 兼容格式"""
    split_dir = os.path.join(task_dir, split)
    images_dir = os.path.join(split_dir, "images")
    ann_file = os.path.join(split_dir, "annotations", f"{split}.json")
    
    if not os.path.exists(ann_file):
        print(f"  警告: 找不到标注文件 {ann_file}")
        return []
    
    # busi 格式：所有图像放在 images/ 目录，所有 mask 放在 masks/0/ 目录
    out_images_dir = os.path.join(output_root, "images")
    out_masks_dir = os.path.join(output_root, "masks", "0")  # MedicalDataSets 期望 masks/0/ 目录
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    
    # 加载 COCO JSON 标注（纯 Python 实现）
    print(f"  加载标注文件: {ann_file}")
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 构建图像 ID 到图像信息的映射
    images_dict = {img['id']: img for img in coco_data.get('images', [])}
    
    # 构建图像 ID 到标注列表的映射
    annotations_dict = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_dict:
            annotations_dict[img_id] = []
        annotations_dict[img_id].append(ann)
    
    print(f"  图像数量: {len(images_dict)}")
    
    file_list = []
    
    for img_id, img_info in images_dict.items():
        img_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        
        # 读取图像
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"    警告: 找不到图像 {img_path}")
            continue
        
        # 获取该图像的所有标注
        anns = annotations_dict.get(img_id, [])
        
        # 创建合并的 mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in anns:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                # 确保 segmentation 是列表格式
                if isinstance(seg, list):
                    seg_mask = polygon_to_mask(seg, height, width)
                    combined_mask = np.maximum(combined_mask, seg_mask)
        
        # 复制图像
        img = Image.open(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # 添加 split 前缀避免文件名冲突
        out_name = f"{split}_{base_name}"
        
        # 保存图像和 mask（busi 格式）
        out_img_path = os.path.join(out_images_dir, f"{out_name}.png")
        out_mask_path = os.path.join(out_masks_dir, f"{out_name}.png")
        
        img.save(out_img_path)
        Image.fromarray(combined_mask * 255).save(out_mask_path)
        
        file_list.append(out_name)
    
    print(f"  处理完成: {len(file_list)} 个样本")
    return file_list


def main():
    print("="*60)
    print("ARCADE 数据集转换工具")
    print("="*60)
    print(f"输入目录: {ARCADE_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"任务类型: {TASK}")
    
    task_dir = os.path.join(ARCADE_ROOT, TASK)
    
    if not os.path.exists(task_dir):
        print(f"错误: 找不到任务目录 {task_dir}")
        return
    
    # 创建输出目录（busi 格式）
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 处理各个数据划分
    all_files = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        # busi 格式：所有数据放在同一目录，通过 lists 文件区分
        files = process_split(task_dir, split, OUTPUT_ROOT)
        all_files[split] = files
    
    # 保存列表文件（busi 格式：直接放在 base_dir 下）
    with open(os.path.join(OUTPUT_ROOT, 'train.txt'), 'w') as f:
        f.write('\n'.join(all_files['train']))
    
    with open(os.path.join(OUTPUT_ROOT, 'val.txt'), 'w') as f:
        f.write('\n'.join(all_files['val']))
    
    with open(os.path.join(OUTPUT_ROOT, 'test.txt'), 'w') as f:
        f.write('\n'.join(all_files['test']))
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"训练集: {len(all_files['train'])} 个样本")
    print(f"验证集: {len(all_files['val'])} 个样本")
    print(f"测试集: {len(all_files['test'])} 个样本")
    print(f"\n输出目录结构 (busi 兼容格式):")
    print(f"  {OUTPUT_ROOT}/")
    print(f"    ├── images/       # 所有图像")
    print(f"    ├── masks/0/      # 所有 mask")
    print(f"    ├── train.txt")
    print(f"    ├── val.txt")
    print(f"    └── test.txt")
    print(f"\n使用命令:")
    print(f"  python main.py --model VMUNet --base_dir ./data/arcade --dataset_name arcade --num_classes 2")


if __name__ == "__main__":
    main()
