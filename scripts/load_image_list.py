"""
辅助函数：从数据集加载图像列表
"""
import os


def load_image_list_from_file(base_dir, val_file_dir='val.txt', dataset_name=''):
    """
    从val.txt或test.txt文件加载图像列表
    
    Args:
        base_dir: 数据集根目录
        val_file_dir: 验证集文件名
        dataset_name: 数据集名称
    
    Returns:
        image_list: 图像名称列表（不含扩展名）
    """
    # 尝试多个可能的文件
    possible_files = [
        os.path.join(base_dir, val_file_dir),
        os.path.join(base_dir, 'test.txt'),
        os.path.join(base_dir, 'val.txt'),
    ]
    
    file_path = None
    for path in possible_files:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(
            f"Cannot find image list file in {base_dir}. "
            f"Tried: {[os.path.basename(p) for p in possible_files]}"
        )
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    image_list = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 移除扩展名
        if line.endswith('.png'):
            line = line[:-4]
        elif line.endswith('.jpg'):
            line = line[:-4]
        
        image_list.append(line)
    
    print(f"Loaded {len(image_list)} images from {file_path}")
    return image_list


def auto_scan_images(base_dir, dataset_name):
    """
    自动扫描数据集目录获取所有图像
    
    Args:
        base_dir: 数据集根目录
        dataset_name: 数据集名称
    
    Returns:
        image_list: 图像名称列表（不含扩展名）
    """
    if dataset_name == 'XCAD':
        # XCAD: 扫描test/images目录
        img_dir = os.path.join(base_dir, 'test', 'images')
    elif dataset_name == 'xca_dataset':
        # xca_dataset: 需要从文件加载，不支持自动扫描
        raise ValueError("xca_dataset does not support auto-scan, please use val.txt")
    elif dataset_name in ['busi', 'arcade']:
        # busi/arcade: 扫描images目录
        img_dir = os.path.join(base_dir, 'images')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # 获取所有png文件
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    image_list = [os.path.splitext(f)[0] for f in image_files]
    
    print(f"Auto-scanned {len(image_list)} images from {img_dir}")
    return image_list
