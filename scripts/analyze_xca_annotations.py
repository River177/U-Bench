"""
分析 XCA 数据集的标注统计信息
遍历 ground_truth 文件夹，统计标注数量
"""
import os
import glob
from collections import defaultdict

def analyze_xca_annotations(base_dir):
    """
    分析 XCA 数据集的标注情况
    
    返回统计信息字典
    """
    stats = {
        'total_cases': 0,
        'total_sequences': 0,
        'total_annotations': 0,
        'cases_with_annotations': 0,
        'sequences_with_annotations': 0,
        'case_stats': {},
        'sequence_stats': defaultdict(int),
        'annotation_files': []
    }
    
    # 遍历所有病例目录
    case_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) 
                and d.startswith('CVAI-')]
    
    stats['total_cases'] = len(case_dirs)
    print(f"找到 {len(case_dirs)} 个病例目录\n")
    
    for case_id in sorted(case_dirs):
        case_path = os.path.join(base_dir, case_id)
        gt_path = os.path.join(case_path, 'ground_truth')
        
        if not os.path.exists(gt_path):
            continue
        
        case_annotations = 0
        case_sequences = []
        
        # 遍历 ground_truth 下的所有序列目录
        sequence_dirs = [d for d in os.listdir(gt_path) 
                       if os.path.isdir(os.path.join(gt_path, d))]
        
        for sequence_id in sequence_dirs:
            gt_seq_path = os.path.join(gt_path, sequence_id)
            
            # 获取该序列下所有标注文件
            gt_files = glob.glob(os.path.join(gt_seq_path, '*.png'))
            annotation_count = len(gt_files)
            
            if annotation_count > 0:
                stats['sequences_with_annotations'] += 1
                stats['total_annotations'] += annotation_count
                case_annotations += annotation_count
                case_sequences.append(sequence_id)
                
                # 记录序列统计
                stats['sequence_stats'][sequence_id] = annotation_count
                
                # 记录标注文件
                for gt_file in gt_files:
                    frame_id = os.path.basename(gt_file)
                    stats['annotation_files'].append({
                        'case_id': case_id,
                        'sequence_id': sequence_id,
                        'frame_id': frame_id,
                        'path': gt_file
                    })
        
        if case_annotations > 0:
            stats['cases_with_annotations'] += 1
            stats['case_stats'][case_id] = {
                'annotations': case_annotations,
                'sequences': len(case_sequences),
                'sequence_list': case_sequences
            }
    
    return stats


def print_statistics(stats):
    """
    打印统计信息
    """
    print("=" * 80)
    print("XCA 数据集标注统计")
    print("=" * 80)
    
    print(f"\n总体统计:")
    print(f"  总病例数: {stats['total_cases']}")
    print(f"  有标注的病例数: {stats['cases_with_annotations']}")
    print(f"  有标注的序列数: {stats['sequences_with_annotations']}")
    print(f"  总标注文件数: {stats['total_annotations']}")
    
    if stats['cases_with_annotations'] > 0:
        avg_per_case = stats['total_annotations'] / stats['cases_with_annotations']
        print(f"  平均每个病例的标注数: {avg_per_case:.1f}")
    
    if stats['sequences_with_annotations'] > 0:
        avg_per_sequence = stats['total_annotations'] / stats['sequences_with_annotations']
        print(f"  平均每个序列的标注数: {avg_per_sequence:.1f}")
    
    # 按病例统计
    print(f"\n按病例统计 (前20个):")
    sorted_cases = sorted(stats['case_stats'].items(), 
                         key=lambda x: x[1]['annotations'], 
                         reverse=True)
    
    for i, (case_id, case_info) in enumerate(sorted_cases[:20], 1):
        print(f"  {i:2d}. {case_id}: {case_info['annotations']:3d} 个标注, "
              f"{case_info['sequences']} 个序列")
    
    if len(sorted_cases) > 20:
        print(f"  ... (还有 {len(sorted_cases) - 20} 个病例)")
    
    # 标注数量分布
    print(f"\n标注数量分布:")
    annotation_counts = defaultdict(int)
    for case_info in stats['case_stats'].values():
        annotation_counts[case_info['annotations']] += 1
    
    for count in sorted(annotation_counts.keys()):
        print(f"  {count:3d} 个标注的病例: {annotation_counts[count]:3d} 个")
    
    # 序列统计
    print(f"\n序列标注统计:")
    sequence_counts = defaultdict(int)
    for seq_count in stats['sequence_stats'].values():
        sequence_counts[seq_count] += 1
    
    for count in sorted(sequence_counts.keys()):
        print(f"  {count:2d} 个标注的序列: {sequence_counts[count]:3d} 个")
    
    # 检查是否有CATH版本
    cath_sequences = [seq for seq in stats['sequence_stats'].keys() if 'CATH' in seq]
    non_cath_sequences = [seq for seq in stats['sequence_stats'].keys() if 'CATH' not in seq]
    
    print(f"\n序列类型统计:")
    print(f"  带 CATH 后缀的序列: {len(cath_sequences)}")
    print(f"  不带 CATH 后缀的序列: {len(non_cath_sequences)}")
    
    # 检查重复标注（同一序列可能有多个版本）
    print(f"\n重复标注检查:")
    base_sequences = {}
    for seq_id in stats['sequence_stats'].keys():
        base_seq = seq_id.replace('CATH', '')
        if base_seq not in base_sequences:
            base_sequences[base_seq] = []
        base_sequences[base_seq].append(seq_id)
    
    duplicates = {k: v for k, v in base_sequences.items() if len(v) > 1}
    if duplicates:
        print(f"  发现 {len(duplicates)} 个序列有多个标注版本:")
        for base_seq, versions in list(duplicates.items())[:10]:
            total_annos = sum(stats['sequence_stats'][v] for v in versions)
            print(f"    {base_seq}: {versions} (共 {total_annos} 个标注)")
        if len(duplicates) > 10:
            print(f"    ... (还有 {len(duplicates) - 10} 个)")
    else:
        print(f"  未发现重复标注")
    
    print("\n" + "=" * 80)


def save_detailed_report(stats, output_file):
    """
    保存详细报告到文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("XCA 数据集标注详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("总体统计:\n")
        f.write(f"  总病例数: {stats['total_cases']}\n")
        f.write(f"  有标注的病例数: {stats['cases_with_annotations']}\n")
        f.write(f"  有标注的序列数: {stats['sequences_with_annotations']}\n")
        f.write(f"  总标注文件数: {stats['total_annotations']}\n\n")
        
        f.write("按病例详细统计:\n")
        sorted_cases = sorted(stats['case_stats'].items(), 
                             key=lambda x: x[1]['annotations'], 
                             reverse=True)
        
        for case_id, case_info in sorted_cases:
            f.write(f"\n{case_id}:\n")
            f.write(f"  标注数: {case_info['annotations']}\n")
            f.write(f"  序列数: {case_info['sequences']}\n")
            f.write(f"  序列列表:\n")
            for seq_id in case_info['sequence_list']:
                seq_count = stats['sequence_stats'][seq_id]
                f.write(f"    - {seq_id}: {seq_count} 个标注\n")
        
        f.write("\n所有标注文件列表:\n")
        for anno in stats['annotation_files']:
            f.write(f"{anno['case_id']}/{anno['sequence_id']}/{anno['frame_id']}\n")
    
    print(f"\n详细报告已保存到: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 XCA 数据集的标注统计')
    parser.add_argument('--base_dir', type=str, default='data/xca_dataset',
                       help='数据集根目录')
    parser.add_argument('--output', type=str, default=None,
                       help='保存详细报告的文件路径（可选）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"错误: 数据集路径 {args.base_dir} 不存在！")
        return
    
    print(f"正在分析数据集: {args.base_dir}\n")
    
    # 分析标注
    stats = analyze_xca_annotations(args.base_dir)
    
    # 打印统计信息
    print_statistics(stats)
    
    # 保存详细报告（如果指定）
    if args.output:
        save_detailed_report(stats, args.output)


if __name__ == "__main__":
    main()

