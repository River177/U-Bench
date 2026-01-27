"""
模型对比可视化脚本
用于生成原图、Ground Truth和多个模型预测结果的对比图

支持的数据集: XCAD, busi, arcade

使用方法:
python scripts/visualize_model_comparison.py \
    --dataset_name XCAD \
    --base_dir ./data/XCAD \
    --image_list "00018_33,00026_38" \
    --models "UTANetMamba,UTANetMamba_Ablation1,UTANetMamba_Ablation2" \
    --output_dir ./visualizations \
    --exp_name default_exp
"""

import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from albumentations import Compose, Resize, Normalize
from scripts.load_image_list import load_image_list_from_file, auto_scan_images


def parse_arguments():
    parser = argparse.ArgumentParser(description='模型对比可视化脚本')
    parser.add_argument('--dataset_name', type=str, required=True, 
                        choices=['XCAD', 'busi', 'arcade', 'xca_dataset'],
                        help='数据集名称')
    parser.add_argument('--base_dir', type=str, required=True, 
                        help='数据集路径')
    parser.add_argument('--image_list', type=str, default='',
                        help='要可视化的图片名称列表，用逗号分隔（不含扩展名）。如果不指定且--all_images为True，则自动加载所有测试图片')
    parser.add_argument('--all_images', action='store_true', default=False,
                        help='处理所有测试集图片（从val.txt或test.txt加载）')
    parser.add_argument('--val_file_dir', type=str, default='val.txt',
                        help='验证集文件列表')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最多处理的图片数量（用于测试，None表示处理全部）')
    parser.add_argument('--models', type=str, required=True,
                        help='要测试的模型名称列表，用逗号分隔')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='输出文件夹路径')
    parser.add_argument('--exp_name', type=str, default='default_exp',
                        help='实验名称')
    parser.add_argument('--img_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU设备')
    parser.add_argument('--checkpoint_dir', type=str, default='./output',
                        help='模型checkpoint基础目录')
    parser.add_argument('--use_best', action='store_true', default=True,
                        help='使用best模型而非final模型')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    
    return parser.parse_args()


def setup_device(gpu_arg):
    """配置计算设备"""
    if gpu_arg.lower() == "cpu":
        return torch.device("cpu")
    
    if not torch.cuda.is_available():
        print("[警告] CUDA不可用，使用CPU")
        return torch.device("cpu")
    
    gpu_index = int(gpu_arg.split(",")[0])
    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def load_image_and_mask(dataset_name, base_dir, image_name):
    """
    根据数据集类型加载图像和mask
    
    返回:
        image: RGB图像 (H, W, 3)
        mask: 二值mask (H, W)
        image_path: 图像路径
    """
    if dataset_name == 'XCAD':
        # XCAD数据集: test/images 和 test/masks
        img_path = os.path.join(base_dir, 'test', 'images', f'{image_name}.png')
        mask_path = os.path.join(base_dir, 'test', 'masks', f'{image_name}.png')
        
    elif dataset_name == 'xca_dataset':
        # 旧的xca_dataset格式: CVAI-*/images 和 ground_truth
        # image_name格式: CVAI-1207/CVAI-1207LAO44_CRA29/00031.png
        parts = image_name.split('/')
        if len(parts) == 3:
            case_id, sequence_id, frame_id = parts
            case_path = os.path.join(base_dir, case_id)
            
            # 尝试多个可能的图像路径
            img_paths = [
                os.path.join(case_path, 'images', sequence_id, frame_id),
                os.path.join(case_path, 'images', sequence_id + 'CATH', frame_id),
            ]
            img_path = None
            for path in img_paths:
                if os.path.exists(path):
                    img_path = path
                    break
            
            # 尝试多个可能的mask路径
            base_sequence = sequence_id.replace('CATH', '')
            mask_paths = [
                os.path.join(case_path, 'ground_truth', base_sequence, frame_id),
                os.path.join(case_path, 'ground_truth', base_sequence + 'CATH', frame_id),
            ]
            mask_path = None
            for path in mask_paths:
                if os.path.exists(path):
                    mask_path = path
                    break
        else:
            raise ValueError(f"Invalid image_name format for xca_dataset: {image_name}")
            
    elif dataset_name in ['busi', 'arcade']:
        # MedicalDataSets格式: images 和 masks/0 或 masks
        img_path = os.path.join(base_dir, 'images', f'{image_name}.png')
        
        # 尝试多个可能的mask路径
        possible_mask_paths = [
            os.path.join(base_dir, 'masks', '0', f'{image_name}.png'),
            os.path.join(base_dir, 'masks', f'{image_name}.png'),
            os.path.join(base_dir, 'GT', f'{image_name}.png'),
        ]
        
        mask_path = None
        for path in possible_mask_paths:
            if os.path.exists(path):
                mask_path = path
                break
        
        # 如果都不存在，使用第一个作为默认（后续会报错并提示）
        if mask_path is None:
            mask_path = possible_mask_paths[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 加载图像
    if not os.path.exists(img_path):
        # 提供更友好的错误信息
        img_dir = os.path.dirname(img_path)
        if os.path.exists(img_dir):
            available_files = [f for f in os.listdir(img_dir) if f.endswith('.png')][:5]
            suggestion = f"\n  Available files in {img_dir}: {available_files}"
        else:
            suggestion = f"\n  Directory does not exist: {img_dir}"
        raise FileNotFoundError(f"Image not found: {img_path}{suggestion}")
    
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 加载mask
    if not os.path.exists(mask_path):
        # 提供更友好的错误信息
        mask_dir = os.path.dirname(mask_path)
        if os.path.exists(mask_dir):
            available_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')][:5]
            suggestion = f"\n  Available files in {mask_dir}: {available_files}"
        else:
            # 显示base_dir下的实际目录结构
            if os.path.exists(base_dir):
                subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                suggestion = f"\n  Directory does not exist: {mask_dir}"
                suggestion += f"\n  Available directories in {base_dir}: {subdirs}"
                
                # 如果有masks目录，显示其子目录
                masks_dir = os.path.join(base_dir, 'masks')
                if os.path.exists(masks_dir):
                    mask_subdirs = os.listdir(masks_dir)
                    suggestion += f"\n  Contents of masks directory: {mask_subdirs[:10]}"
            else:
                suggestion = f"\n  Base directory does not exist: {base_dir}"
        raise FileNotFoundError(f"Mask not found: {mask_path}{suggestion}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    mask = (mask > 0).astype(np.uint8) * 255
    
    return image, mask, img_path


def preprocess_image(image, img_size):
    """预处理图像用于模型推理"""
    transform = Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])
    
    augmented = transform(image=image)
    processed = augmented['image']
    
    # 转换为tensor
    processed = processed.astype('float32')
    processed = processed.transpose(2, 0, 1) / 255.0
    processed = torch.from_numpy(processed).unsqueeze(0)
    
    return processed


def load_model_checkpoint(model_name, dataset_name, exp_name, checkpoint_dir, use_best, device):
    """加载模型checkpoint"""
    # 构建checkpoint路径
    model_dir = os.path.join(checkpoint_dir, model_name, dataset_name, exp_name)
    checkpoint_file = 'checkpoint_best.pth' if use_best else 'checkpoint_final.pth'
    checkpoint_path = os.path.join(model_dir, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 加载checkpoint - 强制映射到当前设备，避免CUDA设备不匹配错误
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"  Loaded checkpoint from: {checkpoint_path}")
    
    # 获取配置
    class Config:
        pass
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config = Config()
        
        # 如果saved_config是字典，转换为对象属性
        if isinstance(saved_config, dict):
            for key, value in saved_config.items():
                setattr(config, key, value)
        else:
            # 如果已经是对象，复制其属性
            for key in dir(saved_config):
                if not key.startswith('_'):
                    setattr(config, key, getattr(saved_config, key))
    else:
        # 创建默认配置
        config = Config()
        config.model = model_name
        config.num_classes = 1
        config.input_channel = 3
        config.img_size = 256
        config.do_deeps = False
    
    # 构建模型
    model = build_model(
        config=config,
        input_channel=getattr(config, 'input_channel', 3),
        num_classes=getattr(config, 'num_classes', 1)
    ).to(device)
    
    # 加载权重
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, config


def predict_single_image(model, image_tensor, config, device, threshold=0.5):
    """对单张图像进行预测"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # 处理深度监督
        if hasattr(config, 'do_deeps') and config.do_deeps:
            output = output[-1]
        
        # 应用sigmoid
        prob = torch.sigmoid(output)
        prob = prob.cpu().numpy()[0, 0]  # [H, W]
        
        # 二值化
        binary = (prob > threshold).astype(np.uint8) * 255
        
    return binary


def create_comparison_figure(image_name, original_img, gt_mask, predictions, model_names, output_path):
    """创建对比图"""
    n_models = len(model_names)
    n_cols = n_models + 2  # 原图 + GT + 各模型预测
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # 原图
    axes[0].imshow(original_img)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 各模型预测
    for i, (pred, model_name) in enumerate(zip(predictions, model_names)):
        axes[i + 2].imshow(pred, cmap='gray')
        axes[i + 2].set_title(model_name, fontsize=12, fontweight='bold')
        axes[i + 2].axis('off')
    
    plt.suptitle(f'Comparison: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison figure: {output_path}")


def save_individual_results(output_dir, image_name, original_img, gt_mask, predictions, model_names):
    """保存单独的结果图像"""
    # 创建子目录
    img_dir = os.path.join(output_dir, 'individual', image_name)
    os.makedirs(img_dir, exist_ok=True)
    
    # 保存原图
    Image.fromarray(original_img).save(os.path.join(img_dir, 'original.png'))
    
    # 保存GT
    Image.fromarray(gt_mask).save(os.path.join(img_dir, 'ground_truth.png'))
    
    # 保存各模型预测
    for pred, model_name in zip(predictions, model_names):
        Image.fromarray(pred).save(os.path.join(img_dir, f'{model_name}.png'))
    
    print(f"Saved individual results to: {img_dir}")


def main():
    args = parse_arguments()
    
    # 设置设备
    device = setup_device(args.gpu)
    print(f"Using device: {device}")
    
    # 解析模型列表
    model_names = [name.strip() for name in args.models.split(',')]
    
    # 解析图像列表
    if args.all_images:
        # 从文件或自动扫描加载所有图像
        try:
            image_list = load_image_list_from_file(
                args.base_dir, args.val_file_dir, args.dataset_name
            )
        except FileNotFoundError:
            print("Warning: No val.txt/test.txt found, trying auto-scan...")
            image_list = auto_scan_images(args.base_dir, args.dataset_name)
        
        # 限制图像数量（如果指定）
        if args.max_images is not None and args.max_images > 0:
            image_list = image_list[:args.max_images]
            print(f"Limited to first {args.max_images} images")
    elif args.image_list:
        # 使用指定的图像列表
        image_list = [name.strip() for name in args.image_list.split(',') if name.strip()]
    else:
        raise ValueError("Either --image_list or --all_images must be specified")
    
    print(f"Dataset: {args.dataset_name}")
    print(f"Total images to process: {len(image_list)}")
    if len(image_list) <= 10:
        print(f"Images to visualize: {image_list}")
    else:
        print(f"First 10 images: {image_list[:10]}")
    print(f"Models to compare: {model_names}")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有模型
    print("\nLoading models...")
    models = {}
    configs = {}
    for model_name in tqdm(model_names, desc="Loading models"):
        try:
            model, config = load_model_checkpoint(
                model_name, args.dataset_name, args.exp_name,
                args.checkpoint_dir, args.use_best, device
            )
            models[model_name] = model
            configs[model_name] = config
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            raise
    
    # 处理每张图像
    print("\nProcessing images...")
    for image_name in tqdm(image_list, desc="Processing"):
        try:
            # 加载原图和GT
            original_img, gt_mask, img_path = load_image_and_mask(
                args.dataset_name, args.base_dir, image_name
            )
            print(f"\nProcessing: {image_name}")
            print(f"  Image path: {img_path}")
            print(f"  Image shape: {original_img.shape}")
            
            # 预处理图像
            image_tensor = preprocess_image(original_img, args.img_size)
            
            # 获取所有模型的预测
            predictions = []
            for model_name in model_names:
                pred = predict_single_image(
                    models[model_name], image_tensor,
                    configs[model_name], device, args.threshold
                )
                # 调整预测结果到原图大小
                pred_resized = cv2.resize(pred, (original_img.shape[1], original_img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                predictions.append(pred_resized)
            
            # 创建对比图
            comparison_path = os.path.join(output_dir, f'{image_name}_comparison.png')
            create_comparison_figure(
                image_name, original_img, gt_mask,
                predictions, model_names, comparison_path
            )
            
            # 保存单独的结果
            save_individual_results(
                output_dir, image_name, original_img, gt_mask,
                predictions, model_names
            )
            
        except Exception as e:
            print(f"✗ Failed to process {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
