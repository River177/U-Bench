import argparse
import os
from datetime import datetime

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





import torch

import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import seaborn as sns

import SimpleITK as sitk
import pandas as pd

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM saving will be disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be disabled.")


from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP]([URL]
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0


def save_prediction_as_dicom(prediction, image, case_name, save_path, z_spacing=1.0, original_dicom_path=None):
    """
    将预测结果保存为 DICOM 格式
    
    Args:
        prediction: 预测结果数组 [D, H, W] 或 [H, W]
        image: 原始图像数组 [D, H, W] 或 [H, W]
        case_name: 病例名称
        save_path: 保存路径（目录）
        z_spacing: z轴间距
        original_dicom_path: 原始 DICOM 文件路径（可选，用于复制元数据）
    """
    if not PYDICOM_AVAILABLE:
        print("Warning: pydicom not available, skipping DICOM save")
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    # 确保预测结果是整数类型
    prediction = prediction.astype(np.uint16)
    
    # 记录原始维度
    is_3d = len(prediction.shape) == 3
    
    # 处理 3D 和 2D 情况
    if is_3d:
        num_slices = prediction.shape[0]
        height, width = prediction.shape[1], prediction.shape[2]
    else:
        num_slices = 1
        height, width = prediction.shape[0], prediction.shape[1]
        prediction = prediction[np.newaxis, ...]
        image = image[np.newaxis, ...] if len(image.shape) == 2 else image
    
    # 尝试从原始 DICOM 文件读取元数据
    reference_ds = None
    if original_dicom_path and os.path.exists(original_dicom_path):
        try:
            # 如果是目录，尝试读取第一个 DICOM 文件
            if os.path.isdir(original_dicom_path):
                dcm_files = [f for f in os.listdir(original_dicom_path) if f.endswith('.dcm')]
                if dcm_files:
                    reference_ds = pydicom.dcmread(os.path.join(original_dicom_path, dcm_files[0]), force=True)
            else:
                reference_ds = pydicom.dcmread(original_dicom_path, force=True)
        except Exception as e:
            print(f"Warning: Could not read reference DICOM file: {e}")
    
    # 为每个切片创建 DICOM 文件
    for slice_idx in range(num_slices):
        # 创建新的 DICOM 数据集
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
        
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        
        # 从参考 DICOM 复制元数据（如果可用）
        if reference_ds is not None:
            # 复制基本元数据
            for tag in ['PatientID', 'PatientName', 'PatientBirthDate', 'PatientSex',
                       'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription',
                       'SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription',
                       'Modality', 'Manufacturer', 'ManufacturerModelName',
                       'SliceThickness', 'KVP', 'ExposureTime', 'XRayTubeCurrent',
                       'PixelSpacing', 'ImagePositionPatient', 'ImageOrientationPatient']:
                if hasattr(reference_ds, tag):
                    try:
                        setattr(ds, tag, getattr(reference_ds, tag))
                    except:
                        pass
        
        # 设置基本 DICOM 标签
        if not hasattr(ds, 'Modality'):
            ds.Modality = 'SEG'  # Segmentation
        if not hasattr(ds, 'SOPClassUID'):
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        if not hasattr(ds, 'SOPInstanceUID'):
            ds.SOPInstanceUID = generate_uid()
        if not hasattr(ds, 'StudyInstanceUID'):
            ds.StudyInstanceUID = generate_uid()
        if not hasattr(ds, 'SeriesInstanceUID'):
            ds.SeriesInstanceUID = generate_uid()
        
        # 设置图像信息
        ds.Rows = height
        ds.Columns = width
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # Unsigned
        
        # 设置像素间距（如果参考 DICOM 中有）
        if not hasattr(ds, 'PixelSpacing'):
            ds.PixelSpacing = [1.0, 1.0]
        
        # 设置切片位置信息
        if is_3d:
            if not hasattr(ds, 'SliceLocation'):
                ds.SliceLocation = float(slice_idx * z_spacing)
            if not hasattr(ds, 'ImagePositionPatient'):
                # 假设从 (0, 0, 0) 开始
                ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx * z_spacing)]
        
        # 设置实例编号
        ds.InstanceNumber = str(slice_idx + 1)
        
        # 设置日期和时间
        now = datetime.now()
        if not hasattr(ds, 'StudyDate'):
            ds.StudyDate = now.strftime('%Y%m%d')
        if not hasattr(ds, 'StudyTime'):
            ds.StudyTime = now.strftime('%H%M%S')
        if not hasattr(ds, 'SeriesDate'):
            ds.SeriesDate = now.strftime('%Y%m%d')
        if not hasattr(ds, 'SeriesTime'):
            ds.SeriesTime = now.strftime('%H%M%S')
        
        # 设置像素数据
        slice_data = prediction[slice_idx].astype(np.uint16)
        ds.PixelData = slice_data.tobytes()
        
        # 保存文件
        filename = f"{case_name}_pred_slice_{slice_idx:04d}.dcm"
        filepath = os.path.join(save_path, filename)
        ds.save_as(filepath, write_like_original=False)
    
    print(f"Saved {num_slices} DICOM prediction files to {save_path}")


def visualize_prediction_overlay(image, prediction, label, case_name, save_path, num_classes=4, max_slices=10):
    """
    生成预测结果的可视化图像（叠加在原始图像上）
    
    Args:
        image: 原始图像数组 [D, H, W] 或 [H, W]
        prediction: 预测结果数组 [D, H, W] 或 [H, W]
        label: 真实标签数组 [D, H, W] 或 [H, W]
        case_name: 病例名称
        save_path: 保存路径（目录）
        num_classes: 类别数量
        max_slices: 最多保存的切片数量（3D 情况下）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    # 处理 2D 和 3D 情况
    if len(prediction.shape) == 3:
        num_slices = min(prediction.shape[0], max_slices)
        # 均匀选择切片
        slice_indices = np.linspace(0, prediction.shape[0] - 1, num_slices, dtype=int)
    else:
        num_slices = 1
        slice_indices = [0]
        prediction = prediction[np.newaxis, ...]
        label = label[np.newaxis, ...] if len(label.shape) == 2 else label
        image = image[np.newaxis, ...] if len(image.shape) == 2 else image
    
    # 定义颜色映射（为每个类别分配不同颜色）
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
    cmaps = []
    for i in range(num_classes):
        if i == 0:  # 背景
            cmaps.append(mcolors.ListedColormap(['black', 'black']))
        else:
            color = colors[i % len(colors)]
            cmaps.append(mcolors.ListedColormap(['black', color]))
    
    for idx, slice_idx in enumerate(slice_indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 归一化图像用于显示
        img_slice = image[slice_idx]
        img_min, img_max = img_slice.min(), img_slice.max()
        if img_max > img_min:
            img_slice = (img_slice - img_min) / (img_max - img_min)
        
        # 原始图像
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 预测结果叠加
        pred_slice = prediction[slice_idx]
        axes[1].imshow(img_slice, cmap='gray', alpha=0.7)
        # 为每个类别创建叠加
        for class_id in range(1, num_classes):
            mask = (pred_slice == class_id)
            if mask.sum() > 0:
                overlay = np.zeros((*mask.shape, 4))
                color = colors[class_id % len(colors)]
                overlay[mask] = [*mcolors.to_rgba(color)[:3], 0.5]  # 半透明
                axes[1].imshow(overlay)
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')
        
        # 真实标签叠加
        label_slice = label[slice_idx]
        axes[2].imshow(img_slice, cmap='gray', alpha=0.7)
        for class_id in range(1, num_classes):
            mask = (label_slice == class_id)
            if mask.sum() > 0:
                overlay = np.zeros((*mask.shape, 4))
                color = colors[class_id % len(colors)]
                overlay[mask] = [*mcolors.to_rgba(color)[:3], 0.5]
                axes[2].imshow(overlay)
        axes[2].set_title('Ground Truth Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'{case_name} - Slice {slice_idx}', fontsize=12)
        plt.tight_layout()
        
        # 保存图像
        if len(prediction.shape) == 3:
            filename = f"{case_name}_slice_{slice_idx:04d}_visualization.png"
        else:
            filename = f"{case_name}_visualization.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_slices} visualization images to {save_path}")


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,do_deeps=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    device = next(net.parameters()).device if next(net.parameters(), None) is not None else torch.device('cpu')
    if len(label.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            input = input.repeat(1, 3, 1, 1)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                if do_deeps:
                    outputs = outputs[-1]
                
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        slice_img = image
        if len(slice_img.shape) == 3:
            input = torch.from_numpy(slice_img).unsqueeze(0).float().to(device)
            if slice_img.shape[0] == 1:
                input = input.repeat(1, 3, 1, 1)
        else:
            x, y = slice_img.shape[0], slice_img.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice_img = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().to(device)
            input = input.repeat(1, 3, 1, 1)
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            if do_deeps and isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    
    # 保存预测结果为 DICOM 格式（如果指定了保存路径）
    if test_save_path is not None and case is not None:
        try:
            # 尝试查找原始 DICOM 文件路径（如果数据集提供了）
            original_dicom_path = None
            # 这里可以根据实际数据集结构添加查找原始 DICOM 文件的逻辑
            
            # 保存 DICOM 格式
            dicom_save_path = os.path.join(test_save_path, 'dicom')
            save_prediction_as_dicom(
                prediction=prediction,
                image=image,
                case_name=case,
                save_path=dicom_save_path,
                z_spacing=z_spacing,
                original_dicom_path=original_dicom_path
            )
            
            # 保存可视化图像
            vis_save_path = os.path.join(test_save_path, 'visualization')
            visualize_prediction_overlay(
                image=image,
                prediction=prediction,
                label=label,
                case_name=case,
                save_path=vis_save_path,
                num_classes=classes
            )
        except Exception as e:
            print(f"Warning: Failed to save prediction for {case}: {e}")
            import traceback
            traceback.print_exc()
    
    # 同时保留原有的 NIfTI 保存选项（如果需要）
    #if test_save_path is not None:
    #    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #    img_itk.SetSpacing((1, 1, z_spacing))
    #    prd_itk.SetSpacing((1, 1, z_spacing))
    #    lab_itk.SetSpacing((1, 1, z_spacing))
    #    sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #    sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #    sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    device = next(net.parameters()).device if next(net.parameters(), None) is not None else torch.device('cpu')
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list