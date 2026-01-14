import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import logging

device = None  # will be configured after argument parsing
from models import build_model
from dataloader.dataloader import getDataloader, getZeroShotDataloader


def seed_torch(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预测模型输出二值分割结果')
    parser.add_argument('--model', type=str, default="U_Net", help='模型名称')
    parser.add_argument('--base_dir', type=str, default="./data/busi", help='数据基础目录')
    parser.add_argument('--dataset_name', type=str, default="busi", help='数据集名称')
    parser.add_argument('--train_file_dir', type=str, default="train.txt", help='训练文件目录')
    parser.add_argument('--val_file_dir', type=str, default="val.txt", help='验证文件目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小，预测时建议设置为1')
    parser.add_argument('--gpu', type=str, default="0", help='GPU设备')
    parser.add_argument('--seed', type=int, default=41, help='随机种子')
    parser.add_argument('--img_size', type=int, default=256, help='图像大小')
    parser.add_argument('--num_classes', type=int, default=1, help='类别数量')
    parser.add_argument('--input_channel', type=int, default=3, help='输入通道数')
    parser.add_argument('--exp_name', type=str, default="default_exp", help='实验名称')
    parser.add_argument('--zero_shot_base_dir', type=str, default="", help='零样本数据基础目录')
    parser.add_argument('--zero_shot_dataset_name', type=str, default="", help='零样本数据集名称')
    parser.add_argument('--do_deeps', type=bool, default=False, help='是否使用深度监督')
    parser.add_argument('--model_id', type=int, default=0, help='模型ID')
    parser.add_argument('--pretrained', action='store_true', default=True, help='是否使用预训练权重')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='禁用预训练权重')
    parser.add_argument('--model_best_or_final', type=str, default="best", choices=["best", "final"], help='使用最佳或最终模型')
    parser.add_argument('--output_dir', type=str, default="./predictions", help='预测结果输出目录')
    parser.add_argument('--save_prob', action='store_true', default=False, help='是否保存概率图')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--use_zero_shot', action='store_true', default=False, help='是否使用零样本数据集进行预测')
    parser.add_argument('--checkpoint_path', type=str, default="", help='直接指定模型checkpoint路径（可选）')
    
    args = parser.parse_args()
    seed_torch(args.seed)
    return args


def configure_device(gpu_arg: str):
    """配置计算设备"""
    global device
    if gpu_arg.lower() == "cpu":
        device = torch.device("cpu")
        print(f"使用设备: {device} (gpu参数: {gpu_arg})")
        return

    if not torch.cuda.is_available():
        print("[警告] CUDA不可用，使用CPU。")
        device = torch.device("cpu")
        return

    primary_gpu = gpu_arg.split(",")[0].strip()
    if primary_gpu == "":
        primary_gpu = "0"

    try:
        gpu_index = int(primary_gpu)
    except ValueError:
        print(f"[警告] 无效的GPU参数 '{gpu_arg}'，使用GPU 0。")
        gpu_index = 0

    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    print(f"使用设备: {device} (gpu参数: {gpu_arg})")


def load_model(args, model_best_or_final="best"):
    """加载训练好的模型"""
    # 如果直接指定了 checkpoint 路径，使用它
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        model_path = args.checkpoint_path
        exp_save_dir = os.path.dirname(model_path)
        args.exp_save_dir = exp_save_dir
        print(f"使用指定的 checkpoint 路径: {model_path}")
    else:
        # 否则根据参数构建路径
        exp_save_dir = f'./output/{args.model}/{args.dataset_name}/{args.exp_name}/'
        args.exp_save_dir = exp_save_dir
        
        if model_best_or_final == "best":
            model_path = os.path.join(exp_save_dir, 'checkpoint_best.pth')
        else:
            model_path = os.path.join(exp_save_dir, 'checkpoint_final.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 首先加载 checkpoint 以获取保存的配置
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 如果 checkpoint 中有 config，使用它来更新 args
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print(f"从 checkpoint 中加载配置，模型: {saved_config.get('model', args.model)}")
        
        # 更新关键配置参数（保持原有的预测相关参数）
        for key in ['model', 'model_id', 'do_deeps', 'img_size', 'num_classes', 'input_channel']:
            if key in saved_config:
                setattr(args, key, saved_config[key])
    
    # 使用更新后的配置构建模型
    model = build_model(config=args, input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    
    # 加载模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"成功加载模型: {model_path}")
    return model, model_path


def setup_logger(output_dir):
    """设置日志记录器"""
    log_file = os.path.join(output_dir, 'prediction.log')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger


def predict_and_save(args, model, logger):
    """预测并保存二值分割结果"""
    
    # 根据参数选择使用验证集还是零样本数据集
    if args.use_zero_shot and args.zero_shot_dataset_name != "":
        dataloader = getZeroShotDataloader(args)
        dataset_name = args.zero_shot_dataset_name
        logger.info(f"使用零样本数据集: {dataset_name}")
    else:
        _, dataloader = getDataloader(args)
        dataset_name = args.dataset_name
        logger.info(f"使用验证数据集: {dataset_name}")
    
    # 创建输出目录
    output_base_dir = os.path.join(args.output_dir, args.model, dataset_name, args.exp_name)
    binary_dir = os.path.join(output_base_dir, 'binary')
    os.makedirs(binary_dir, exist_ok=True)
    
    if args.save_prob:
        prob_dir = os.path.join(output_base_dir, 'probability')
        os.makedirs(prob_dir, exist_ok=True)
    
    logger.info(f"输出目录: {output_base_dir}")
    logger.info(f"二值化阈值: {args.threshold}")
    logger.info(f"开始预测...")
    
    model.eval()
    
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="预测中"):
            input_images = sampled_batch['image'].to(device)
            
            # 获取图像名称
            if 'name' in sampled_batch:
                image_names = sampled_batch['name']
            elif 'case' in sampled_batch:
                image_names = sampled_batch['case']
            else:
                # 如果没有名称，使用索引
                image_names = [f"image_{i_batch * args.batch_size + i}" for i in range(input_images.size(0))]
            
            # 模型预测
            outputs = model(input_images)
            
            # 如果使用深度监督，取最后一个输出
            if args.do_deeps:
                outputs = outputs[-1]
            
            # 应用sigmoid获取概率
            probs = torch.sigmoid(outputs)
            
            # 处理每张图像
            for idx in range(probs.size(0)):
                prob = probs[idx].cpu().numpy()  # [C, H, W] or [1, H, W]
                
                # 如果是多通道，取第一个通道
                if prob.shape[0] == 1:
                    prob = prob[0]  # [H, W]
                else:
                    prob = prob[0]  # 取第一个通道
                
                # 获取图像名称
                if isinstance(image_names, list):
                    img_name = image_names[idx]
                else:
                    img_name = image_names[idx].item() if hasattr(image_names[idx], 'item') else str(image_names[idx])
                
                # 确保名称是字符串且移除扩展名
                img_name = str(img_name)
                if '.' in img_name:
                    img_name = os.path.splitext(img_name)[0]
                
                # 保存概率图（可选）
                if args.save_prob:
                    prob_normalized = (prob * 255).astype(np.uint8)
                    prob_path = os.path.join(prob_dir, f"{img_name}_prob.png")
                    Image.fromarray(prob_normalized).save(prob_path)
                
                # 二值化
                binary = (prob > args.threshold).astype(np.uint8) * 255
                
                # 保存二值图像
                binary_path = os.path.join(binary_dir, f"{img_name}_pred.png")
                Image.fromarray(binary).save(binary_path)
    
    logger.info(f"预测完成！")
    logger.info(f"二值结果保存在: {binary_dir}")
    if args.save_prob:
        logger.info(f"概率图保存在: {prob_dir}")
    
    return output_base_dir


def main():
    """主函数"""
    args = parse_arguments()
    
    # 配置设备
    configure_device(args.gpu)
    
    try:
        # 先加载模型（会从 checkpoint 中更新配置）
        model, model_path = load_model(args, model_best_or_final=args.model_best_or_final)
        
        # 使用更新后的配置创建输出目录
        output_dir = os.path.join(args.output_dir, args.model, 
                                  args.zero_shot_dataset_name if args.use_zero_shot else args.dataset_name, 
                                  args.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        logger = setup_logger(output_dir)
        
        logger.info("=" * 60)
        logger.info(f"模型: {args.model}")
        logger.info(f"数据集: {args.dataset_name}")
        logger.info(f"实验名称: {args.exp_name}")
        logger.info(f"使用{'最佳' if args.model_best_or_final == 'best' else '最终'}模型")
        logger.info(f"模型路径: {model_path}")
        logger.info("=" * 60)
        
        # 预测并保存结果
        result_dir = predict_and_save(args, model, logger)
        
        logger.info(f"所有预测结果已保存到: {result_dir}")
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
