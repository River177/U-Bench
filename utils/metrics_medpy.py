import os 

import torch
import numpy as np
from medpy.metric.binary import hd, assd, dc, precision, recall, specificity, hd95

def get_metrics(output, target):

    output = torch.sigmoid(output).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    

    output = (output > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)

    # Dice Similarity Coefficient (DSC)
    dice = dc(output, target)

    # HD95 (95th percentile Hausdorff Distance)
    try:
        if output.sum() > 0 and target.sum() > 0:
            hd95_value = hd95(output, target)
        else:
            hd95_value = 0.0
    except:
        # 如果计算失败（例如图像太小），返回0
        hd95_value = 0.0

    intersection = np.sum(output * target)
    union = np.sum(output) + np.sum(target) - intersection
    iou = intersection / union if union > 0 else 0

    SE = recall(output, target)

    PC = precision(output, target)

    SP = specificity(output, target)

    ACC = get_accuracy(output, target)

    F1 = 2 * (PC * SE) / (PC + SE) if (PC + SE) > 0 else 0

    return iou, dice, SE, PC, F1, SP, ACC, hd95_value

def dice_coef(output, target):

    output = torch.sigmoid(output).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    output = (output > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)
    dice = dc(output, target)
    return dice

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)  
    corr = np.sum(SR == GT)  
    acc = float(corr) / float(SR.size) 
    return acc
