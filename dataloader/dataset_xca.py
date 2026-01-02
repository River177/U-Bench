import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import glob


class XCADataset(Dataset):
    """
    XCA (X-ray Coronary Angiography) Dataset Loader - 用于 xca_dataset (CVAI-* 格式)
    
    数据集结构:
    xca_dataset/
        CVAI-1207/
            images/
                CVAI-1207LAO44_CRA29/
                    00000.png
                    00001.png
                    ...
                CVAI-1207RAO2_CAU30/
                    ...
            ground_truth/
                CVAI-1207LAO44_CRA29/
                    00031.png  (普通版本，不使用)
                    ...
                CVAI-1207LAO44_CRA29CATH/
                    00031.png  (CATH版本，优先使用)
                    ...
        train.txt  (格式: CVAI-1207/CVAI-1207LAO44_CRA29/00031.png)
        val.txt
        test.txt
    
    注意：
    - 默认使用普通版本标注（非 CATH），若缺失再回退到 CATH
    - 标注中白色/灰白色都视为血管（前景），统一为 1
    """
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        
        # 如果存在 train.txt/val.txt，则从文件读取
        if self.mode == "train":
            file_path = os.path.join(self._base_dir, train_file_dir)
        elif self.mode == "val":
            file_path = os.path.join(self._base_dir, val_file_dir)
        elif self.mode == "test":
            test_path = os.path.join(self._base_dir, "test.txt")
            if os.path.exists(test_path):
                file_path = test_path
            else:
                file_path = os.path.join(self._base_dir, val_file_dir)
        else:
            file_path = None
        
        if file_path and os.path.exists(file_path):
            # 从文件读取样本列表
            with open(file_path, "r", encoding='utf-8') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.strip() for item in self.sample_list if item.strip()]
        else:
            # 自动扫描数据集，找到所有有标注的图像
            self.sample_list = self._scan_dataset()
        
        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def _scan_dataset(self):
        """
        扫描数据集，找到所有有CATH版本标注的图像对
        返回格式: [case_id/sequence_id/frame_id, ...]
        """
        sample_list = []
        
        # 遍历所有病例目录
        case_dirs = [d for d in os.listdir(self._base_dir) 
                    if os.path.isdir(os.path.join(self._base_dir, d)) 
                    and d.startswith('CVAI-')]
        
        for case_id in sorted(case_dirs):
            case_path = os.path.join(self._base_dir, case_id)
            gt_path = os.path.join(case_path, 'ground_truth')
            
            if not os.path.exists(gt_path):
                continue
            
            # 遍历 ground_truth 下的所有序列目录，只处理非 CATH 版本
            sequence_dirs = [d for d in os.listdir(gt_path) 
                           if os.path.isdir(os.path.join(gt_path, d))
                           and 'CATH' not in d]  # 只扫描普通版本的序列
            
            for sequence_id in sequence_dirs:
                gt_seq_path = os.path.join(gt_path, sequence_id)
                
                # 获取该序列下所有标注文件
                gt_files = glob.glob(os.path.join(gt_seq_path, '*.png'))
                
                for gt_file in gt_files:
                    frame_id = os.path.basename(gt_file)
                    
                    # 普通序列直接使用原名称
                    base_sequence = sequence_id
                    img_seq_path = os.path.join(case_path, 'images', base_sequence)
                    
                    # 如果原始序列不存在，尝试 CATH 版本目录
                    if not os.path.exists(img_seq_path):
                        img_seq_path = os.path.join(case_path, 'images', sequence_id + 'CATH')
                    
                    img_file = os.path.join(img_seq_path, frame_id)
                    
                    # 检查图像文件是否存在
                    if os.path.exists(img_file):
                        # 存储相对路径: case_id/sequence_id/frame_id
                        sample_id = f"{case_id}/{base_sequence}/{frame_id}"
                        sample_list.append(sample_id)
        
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_id = self.sample_list[idx]
        
        # 解析样本ID: case_id/sequence_id/frame_id
        parts = sample_id.split('/')
        if len(parts) == 3:
            case_id, sequence_id, frame_id = parts
        else:
            # 兼容其他格式
            case_id = parts[0]
            sequence_id = '/'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
            frame_id = parts[-1]
        
        # 构建完整路径
        case_path = os.path.join(self._base_dir, case_id)
        
        # 图像路径 - 尝试多个可能的路径
        img_paths = [
            os.path.join(case_path, 'images', sequence_id, frame_id),
            os.path.join(case_path, 'images', sequence_id.replace('CATH', ''), frame_id),
            os.path.join(case_path, 'images', sequence_id + 'CATH', frame_id),
        ]
        
        img_path = None
        for path in img_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        # 如果还是找不到，使用第一个路径（会在后面处理错误）
        if img_path is None:
            img_path = img_paths[0]
        
        # 标注路径 - 优先使用普通版本，缺失时回退至 CATH
        base_sequence = sequence_id.replace('CATH', '')
        gt_paths = [
            os.path.join(case_path, 'ground_truth', base_sequence, frame_id),  # 优先普通版本
            os.path.join(case_path, 'ground_truth', sequence_id, frame_id),
            os.path.join(case_path, 'ground_truth', base_sequence + 'CATH', frame_id),  # 回退：CATH版本
            os.path.join(case_path, 'ground_truth', sequence_id + 'CATH', frame_id),
        ]
        
        gt_path = None
        for path in gt_paths:
            if os.path.exists(path):
                gt_path = path
                break
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Failed to read image: {img_path}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标注
        if gt_path and os.path.exists(gt_path):
            label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                print(f"Warning: Failed to read label: {gt_path}")
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                # CATH版本标注处理：将白色(255)和灰白色(127)都统一为前景(血管)
                # 0: 背景, 127: 灰白色血管, 255: 白色狭窄/斑块区域
                # 统一处理：非0值都视为血管（前景）
                label = (label > 0).astype(np.uint8) * 255  # 将127和255都转换为255
        else:
            print(f"Warning: Label not found for {sample_id}")
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 确保图像和标注尺寸一致
        if image.shape[:2] != label.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        # 添加通道维度
        if len(label.shape) == 2:
            label = label[..., None]
        
        # 手动归一化和转置（如果transform没有处理）
        if isinstance(image, np.ndarray):
            image = image.astype('float32')
            image = image.transpose(2, 0, 1) / 255
            
            label = label.astype('float32')
            label = label.transpose(2, 0, 1) / 255
            # 将非0值都视为前景（血管）
            label[label > 0] = 1
        
        sample = {"image": image, "label": label, "case": sample_id}
        return sample


class XCADNewDataset(Dataset):
    """
    XCAD (X-ray Coronary Angiography) Dataset Loader - 用于新的 XCAD 数据集
    
    数据集结构:
    XCAD/
        test/
            images/
                00018_33.png
                00026_38.png
                ...
            masks/
                00018_33.png
                00026_38.png
                ...
        train.txt  (由 prepare_xcad_dataset.py 生成，只包含文件名，不含扩展名)
        val.txt    (由 prepare_xcad_dataset.py 生成)
    
    注意：
    - 只有 test 目录中有完整的 GT，需要先运行 prepare_xcad_dataset.py 划分数据集
    - 数据集划分后，train 和 val 的数据仍然从 test/images 和 test/masks 中读取
    - 标注中非0值都视为前景（血管），统一为 1
    """
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        
        # 确定要读取的文件列表
        if self.mode == "train":
            file_path = os.path.join(self._base_dir, train_file_dir)
        elif self.mode == "val":
            file_path = os.path.join(self._base_dir, val_file_dir)
        elif self.mode == "test":
            # test 模式：直接扫描 test 目录
            test_images_dir = os.path.join(self._base_dir, "test", "images")
            if os.path.exists(test_images_dir):
                # 自动扫描 test 目录
                image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png')])
                self.sample_list = [os.path.splitext(f)[0] for f in image_files]
            else:
                # 尝试从文件读取
                test_path = os.path.join(self._base_dir, "test.txt")
                if os.path.exists(test_path):
                    file_path = test_path
                else:
                    file_path = os.path.join(self._base_dir, val_file_dir)
                if file_path and os.path.exists(file_path):
                    with open(file_path, "r", encoding='utf-8') as f:
                        self.sample_list = f.readlines()
                    self.sample_list = [item.strip() for item in self.sample_list if item.strip()]
        else:
            file_path = None
        
        # 从文件读取样本列表（train 和 val 模式）
        if self.mode != "test" and file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.strip() for item in self.sample_list if item.strip()]
        elif self.mode == "test" and len(self.sample_list) == 0:
            # test 模式且没有找到文件，尝试自动扫描
            test_images_dir = os.path.join(self._base_dir, "test", "images")
            if os.path.exists(test_images_dir):
                image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png')])
                self.sample_list = [os.path.splitext(f)[0] for f in image_files]
        
        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # sample_list 中存储的是文件名（不含扩展名）
        img_name = self.sample_list[idx]
        
        # 构建图像和标注路径
        # 注意：划分后的 train 和 val 数据仍然从 test 目录中读取
        # 因为 prepare_xcad_dataset.py 只是创建了 train.txt 和 val.txt，并没有移动文件
        # 如果需要从 train_new/val_new 读取，可以修改这里的路径逻辑
        
        # 优先尝试从 test 目录读取（原始数据位置）
        test_images_dir = os.path.join(self._base_dir, "test", "images")
        test_masks_dir = os.path.join(self._base_dir, "test", "masks")
        
        # 如果 test 目录不存在，尝试从 train_new/val_new 读取
        if not os.path.exists(test_images_dir):
            if self.mode == "train":
                test_images_dir = os.path.join(self._base_dir, "train_new", "images")
                test_masks_dir = os.path.join(self._base_dir, "train_new", "masks")
            elif self.mode == "val":
                test_images_dir = os.path.join(self._base_dir, "val_new", "images")
                test_masks_dir = os.path.join(self._base_dir, "val_new", "masks")
        
        img_path = os.path.join(test_images_dir, f"{img_name}.png")
        mask_path = os.path.join(test_masks_dir, f"{img_name}.png")
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Failed to read image: {img_path}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标注
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"Warning: Failed to read label: {mask_path}")
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # 将非0值都视为前景（血管），统一处理
            label = (label > 0).astype(np.uint8) * 255
        
        # 确保图像和标注尺寸一致
        if image.shape[:2] != label.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        # 添加通道维度
        if len(label.shape) == 2:
            label = label[..., None]
        
        # 手动归一化和转置（如果transform没有处理）
        if isinstance(image, np.ndarray):
            image = image.astype('float32')
            image = image.transpose(2, 0, 1) / 255
            
            label = label.astype('float32')
            label = label.transpose(2, 0, 1) / 255
            # 将非0值都视为前景（血管）
            label[label > 0] = 1
        
        sample = {"image": image, "label": label, "case": img_name}
        return sample

