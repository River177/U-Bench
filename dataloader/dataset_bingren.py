"""
Bingren 数据集加载器
用于多器官分割任务（6类器官 + 背景）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import random
from scipy import ndimage


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator_bingren(object):
    """bingren 数据集的随机数据增强"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, mask, force_apply=False, **kwargs):
        if random.random() > 0.5:
            image, mask = random_rot_flip(image, mask)
        elif random.random() > 0.5:
            image, mask = random_rotate(image, mask)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32)).long()

        return {'image': image, 'mask': mask}


class BingrenDataset(Dataset):
    """
    Bingren 多器官分割数据集
    
    器官标签:
        0: 背景
        1: Bladder (膀胱)
        2: Rectum (直肠)
        3: Femur_R (右股骨)
        4: Femur_L (左股骨)
        5: Intestine (肠道)
        6: CE (临床靶区)
    
    训练/验证集: 2D 切片 (.npz with img[H,W], label[H,W])
    测试集: 3D 体积 (.npz with img[D,H,W], label[D,H,W])
    """
    
    def __init__(self, base_dir, split="train", nclass=7, transform=None):
        """
        Args:
            base_dir: 数据集根目录 (包含 train/, valid/, test/, lists_bingren/)
            split: 'train', 'valid', 或 'test'
            nclass: 类别数 (7 = 6个器官 + 背景)
            transform: 数据增强变换
        """
        self.base_dir = base_dir
        self.split = split
        self.nclass = nclass
        self.transform = transform
        
        # 加载文件列表
        list_dir = os.path.join(base_dir, 'lists_bingren')
        list_file = os.path.join(list_dir, f'{split}.txt')
        
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.sample_list = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 如果没有列表文件，直接扫描目录
            data_dir = os.path.join(base_dir, split)
            self.sample_list = [f.replace('.npz', '') for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        # 数据目录
        self.data_dir = os.path.join(base_dir, split)
        
        print(f"[BingrenDataset] Loaded {len(self.sample_list)} samples from {split} split")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        # 移除 .npz 后缀（如果存在）
        if slice_name.endswith('.npz'):
            slice_name = slice_name[:-4]
        
        # 加载 npz 文件
        data_path = os.path.join(self.data_dir, f'{slice_name}.npz')
        data = np.load(data_path)
        
        image = data['img'].astype(np.float32)  # CT 图像
        label = data['label'].astype(np.int64)  # 分割标签
        
        # 检测数据维度：3D 体积 (D, H, W) 或 2D 切片 (H, W)
        is_3d = len(image.shape) == 3
        
        # 3D 体积（测试集或验证集的3D格式），不应用变换
        if is_3d or self.split == 'test':
            # 3D 体积: 返回 tensor 格式用于 test_single_volume
            # 归一化 CT 值到 [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            # 转换为 tensor 并添加 batch 维度
            image = torch.from_numpy(image).unsqueeze(0).float()  # [1, D, H, W]
            label = torch.from_numpy(label).unsqueeze(0).long()   # [1, D, H, W]
            sample = {
                'image': image,
                'label': label,
                'case_name': slice_name
            }
            return sample
        
        # 训练集是 2D 切片
        # 归一化 CT 值到 [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            # 扩展为 3 通道 (适配预训练模型)
            if image.dim() == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        else:
            # 默认处理
            image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
            label = torch.from_numpy(label).long()
        
        sample = {
            'image': image,
            'label': label,
            'case_name': slice_name
        }
        
        return sample


class BingrenDatasetBinary(Dataset):
    """
    Bingren 二分类数据集（用于单器官分割）
    将多类别标签转换为二分类（前景/背景）
    """
    
    def __init__(self, base_dir, split="train", transform=None, target_organ=None):
        """
        Args:
            base_dir: 数据集根目录
            split: 'train', 'valid', 或 'test'
            transform: 数据增强变换
            target_organ: 目标器官标签 (1-6), None 表示所有器官为前景
        """
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.target_organ = target_organ
        
        # 加载文件列表
        list_dir = os.path.join(base_dir, 'lists_bingren')
        list_file = os.path.join(list_dir, f'{split}.txt')
        
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.sample_list = [line.strip() for line in f.readlines() if line.strip()]
        else:
            data_dir = os.path.join(base_dir, split)
            self.sample_list = [f.replace('.npz', '') for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        self.data_dir = os.path.join(base_dir, split)
        print(f"[BingrenDatasetBinary] Loaded {len(self.sample_list)} samples from {split} split")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, f'{slice_name}.npz')
        data = np.load(data_path)
        
        image = data['img'].astype(np.float32)
        label = data['label'].astype(np.int64)
        
        # 归一化
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # 转换为二分类
        if self.target_organ is not None:
            label = (label == self.target_organ).astype(np.int64)
        else:
            label = (label > 0).astype(np.int64)
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            if image.dim() == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        else:
            image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
            label = torch.from_numpy(label).float().unsqueeze(0)
        
        return {'image': image, 'label': label, 'case_name': slice_name}
