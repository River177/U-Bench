"""
XCAD 数据集划分脚本
将 test 目录中的数据划分为 train 和 val 集合
"""
import os
import random
import shutil
from pathlib import Path


def split_xcad_dataset(base_dir, train_ratio=0.8, seed=42):
    """
    将 XCAD 数据集的 test 目录中的数据划分为 train 和 val
    
    Args:
        base_dir: XCAD 数据集根目录，例如 'data/XCAD'
        train_ratio: 训练集比例，默认 0.8 (80% train, 20% val)
        seed: 随机种子
    """
    base_dir = Path(base_dir)
    test_dir = base_dir / "test"
    
    # 检查 test 目录是否存在
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    images_dir = test_dir / "images"
    masks_dir = test_dir / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(f"Images or masks directory not found in {test_dir}")
    
    # 获取所有图像文件名（不包含扩展名）
    image_files = sorted([f.stem for f in images_dir.glob("*.png")])
    
    print(f"Found {len(image_files)} images in test directory")
    
    # 验证每个图像都有对应的 mask
    valid_files = []
    for img_name in image_files:
        img_path = images_dir / f"{img_name}.png"
        mask_path = masks_dir / f"{img_name}.png"
        
        if img_path.exists() and mask_path.exists():
            valid_files.append(img_name)
        else:
            print(f"Warning: Missing pair for {img_name}")
    
    print(f"Valid image-mask pairs: {len(valid_files)}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 随机打乱
    random.shuffle(valid_files)
    
    # 划分数据集
    split_idx = int(len(valid_files) * train_ratio)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    
    # 创建 train 和 val 目录结构
    train_images_dir = base_dir / "train_new" / "images"
    train_masks_dir = base_dir / "train_new" / "masks"
    val_images_dir = base_dir / "val_new" / "images"
    val_masks_dir = base_dir / "val_new" / "masks"
    
    # 创建目录
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_masks_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制训练集文件
    print("Copying train files...")
    for img_name in train_files:
        src_img = images_dir / f"{img_name}.png"
        src_mask = masks_dir / f"{img_name}.png"
        dst_img = train_images_dir / f"{img_name}.png"
        dst_mask = train_masks_dir / f"{img_name}.png"
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
    
    # 复制验证集文件
    print("Copying val files...")
    for img_name in val_files:
        src_img = images_dir / f"{img_name}.png"
        src_mask = masks_dir / f"{img_name}.png"
        dst_img = val_images_dir / f"{img_name}.png"
        dst_mask = val_masks_dir / f"{img_name}.png"
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
    
    # 生成 train.txt 和 val.txt 文件（只包含文件名，不包含扩展名）
    train_txt = base_dir / "train.txt"
    val_txt = base_dir / "val.txt"
    
    with open(train_txt, "w", encoding="utf-8") as f:
        for img_name in train_files:
            f.write(f"{img_name}\n")
    
    with open(val_txt, "w", encoding="utf-8") as f:
        for img_name in val_files:
            f.write(f"{img_name}\n")
    
    print(f"\nDataset split completed!")
    print(f"Train list saved to: {train_txt}")
    print(f"Val list saved to: {val_txt}")
    print(f"Train images: {train_images_dir}")
    print(f"Train masks: {train_masks_dir}")
    print(f"Val images: {val_images_dir}")
    print(f"Val masks: {val_masks_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split XCAD dataset")
    parser.add_argument("--base_dir", type=str, default="data/XCAD", 
                       help="XCAD dataset base directory")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training set ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_xcad_dataset(args.base_dir, args.train_ratio, args.seed)

