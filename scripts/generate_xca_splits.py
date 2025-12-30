"""
生成 XCA 数据集的 train.txt, val.txt, test.txt 文件
"""
import os
import glob
import random
from collections import defaultdict

def scan_xca_dataset(base_dir):
    """
    扫描 XCA 数据集，找到所有有标注的图像对
    
    返回: 
        sample_list: [(case_id, sequence_id, frame_id, img_path, gt_path), ...]
    """
    sample_list = []
    
    # 遍历所有病例目录
    case_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) 
                and d.startswith('CVAI-')]
    
    print(f"找到 {len(case_dirs)} 个病例目录")
    
    for case_id in sorted(case_dirs):
        case_path = os.path.join(base_dir, case_id)
        gt_path = os.path.join(case_path, 'ground_truth')
        
        if not os.path.exists(gt_path):
            continue
        
        # 遍历 ground_truth 下的所有序列目录，只处理非 CATH 版本
        sequence_dirs = [d for d in os.listdir(gt_path) 
                       if os.path.isdir(os.path.join(gt_path, d))
                       and 'CATH' not in d]  # 只扫描不含 CATH 的序列
        
        for sequence_id in sequence_dirs:
                gt_seq_path = os.path.join(gt_path, sequence_id)
                
                # 获取该序列下所有标注文件
                gt_files = glob.glob(os.path.join(gt_seq_path, '*.png'))
                
                for gt_file in gt_files:
                    frame_id = os.path.basename(gt_file)
                    
                    # 构建图像路径
                    # 非 CATH 序列直接使用原始名称
                    base_sequence = sequence_id
                    img_seq_path = os.path.join(case_path, 'images', base_sequence)
                    
                    # 如果原始序列不存在，尝试附加 CATH 后缀的目录
                    if not os.path.exists(img_seq_path):
                        img_seq_path = os.path.join(case_path, 'images', f"{base_sequence}CATH")
                    
                    img_file = os.path.join(img_seq_path, frame_id)
                    
                    # 检查图像文件是否存在
                    if os.path.exists(img_file):
                        # 存储样本信息（使用 sample_id 作为唯一标识）
                        sample_id = f"{case_id}/{base_sequence}/{frame_id}"
                        sample_list.append({
                            'case_id': case_id,
                            'sequence_id': base_sequence,
                            'frame_id': frame_id,
                            'sample_id': sample_id,
                            'img_path': img_file,
                            'gt_path': gt_file
                        })
    
    # 去重：同一个 sample_id 可能因为 CATH 版本而重复
    seen = set()
    unique_samples = []
    for sample in sample_list:
        if sample['sample_id'] not in seen:
            seen.add(sample['sample_id'])
            unique_samples.append(sample)
    
    print(f"去重前: {len(sample_list)} 个样本, 去重后: {len(unique_samples)} 个样本")
    return unique_samples


def split_by_case(sample_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    按病例划分数据集（推荐方式）
    确保同一个病例的所有样本都在同一个集合中
    """
    random.seed(seed)
    
    # 按病例分组
    case_samples = defaultdict(list)
    for sample in sample_list:
        case_samples[sample['case_id']].append(sample)
    
    cases = list(case_samples.keys())
    random.shuffle(cases)
    
    # 计算划分点
    total_cases = len(cases)
    train_end = int(total_cases * train_ratio)
    val_end = train_end + int(total_cases * val_ratio)
    
    # 划分病例
    train_cases = cases[:train_end]
    val_cases = cases[train_end:val_end]
    test_cases = cases[val_end:]
    
    # 收集样本
    train_samples = []
    val_samples = []
    test_samples = []
    
    for case in train_cases:
        train_samples.extend(case_samples[case])
    
    for case in val_cases:
        val_samples.extend(case_samples[case])
    
    for case in test_cases:
        test_samples.extend(case_samples[case])
    
    return train_samples, val_samples, test_samples


def split_random(sample_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    随机划分数据集
    """
    random.seed(seed)
    random.shuffle(sample_list)
    
    total = len(sample_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_samples = sample_list[:train_end]
    val_samples = sample_list[train_end:val_end]
    test_samples = sample_list[val_end:]
    
    return train_samples, val_samples, test_samples


def save_splits(base_dir, train_samples, val_samples, test_samples):
    """
    保存划分结果到文件
    """
    train_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'val.txt')
    test_file = os.path.join(base_dir, 'test.txt')
    
    # 保存训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(sample['sample_id'] + '\n')
    print(f"[OK] 训练集已保存到: {train_file} ({len(train_samples)} 个样本)")
    
    # 保存验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(sample['sample_id'] + '\n')
    print(f"[OK] 验证集已保存到: {val_file} ({len(val_samples)} 个样本)")
    
    # 保存测试集
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(sample['sample_id'] + '\n')
    print(f"[OK] 测试集已保存到: {test_file} ({len(test_samples)} 个样本)")


def print_statistics(sample_list, train_samples, val_samples, test_samples):
    """
    打印数据集统计信息
    """
    print("\n" + "=" * 60)
    print("数据集划分统计")
    print("=" * 60)
    
    # 按病例统计
    def count_cases(samples):
        cases = set(s['case_id'] for s in samples)
        return len(cases), len(samples)
    
    train_cases, train_count = count_cases(train_samples)
    val_cases, val_count = count_cases(val_samples)
    test_cases, test_count = count_cases(test_samples)
    
    print(f"\n总样本数: {len(sample_list)}")
    print(f"总病例数: {len(set(s['case_id'] for s in sample_list))}")
    
    print(f"\n训练集:")
    print(f"  - 病例数: {train_cases}")
    print(f"  - 样本数: {train_count} ({train_count/len(sample_list)*100:.1f}%)")
    
    print(f"\n验证集:")
    print(f"  - 病例数: {val_cases}")
    print(f"  - 样本数: {val_count} ({val_count/len(sample_list)*100:.1f}%)")
    
    print(f"\n测试集:")
    print(f"  - 病例数: {test_cases}")
    print(f"  - 样本数: {test_count} ({test_count/len(sample_list)*100:.1f}%)")
    
    # 按病例列出划分情况
    print(f"\n按病例划分详情:")
    all_cases = set(s['case_id'] for s in sample_list)
    train_case_set = set(s['case_id'] for s in train_samples)
    val_case_set = set(s['case_id'] for s in val_samples)
    test_case_set = set(s['case_id'] for s in test_samples)
    
    print(f"  训练集病例: {sorted(train_case_set)}")
    print(f"  验证集病例: {sorted(val_case_set)}")
    print(f"  测试集病例: {sorted(test_case_set)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 XCA 数据集的划分文件')
    parser.add_argument('--base_dir', type=str, default='data/xca_dataset',
                       help='数据集根目录')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例 (默认: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='测试集比例 (默认: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--split_method', type=str, default='case',
                       choices=['case', 'random'],
                       help='划分方式: case(按病例) 或 random(随机) (默认: case)')
    
    args = parser.parse_args()
    
    # 检查比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和为 {total_ratio}，将自动归一化")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    print("=" * 60)
    print("XCA 数据集划分工具")
    print("=" * 60)
    print(f"\n数据集目录: {args.base_dir}")
    print(f"划分方式: {args.split_method}")
    print(f"比例: 训练集={args.train_ratio:.1%}, 验证集={args.val_ratio:.1%}, 测试集={args.test_ratio:.1%}")
    print(f"随机种子: {args.seed}")
    
    # 扫描数据集
    print("\n正在扫描数据集...")
    sample_list = scan_xca_dataset(args.base_dir)
    
    if len(sample_list) == 0:
        print("错误: 未找到任何有标注的图像对！")
        print("请检查数据集路径和结构是否正确。")
        return
    
    print(f"[OK] 找到 {len(sample_list)} 个有标注的图像对")
    
    # 划分数据集
    print(f"\n正在划分数据集 ({args.split_method} 方式)...")
    if args.split_method == 'case':
        train_samples, val_samples, test_samples = split_by_case(
            sample_list, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    else:
        train_samples, val_samples, test_samples = split_random(
            sample_list, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    
    # 打印统计信息
    print_statistics(sample_list, train_samples, val_samples, test_samples)
    
    # 保存文件
    print(f"\n正在保存划分文件...")
    save_splits(args.base_dir, train_samples, val_samples, test_samples)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

