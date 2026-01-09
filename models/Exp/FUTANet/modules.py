import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

# ==================== 拉普拉斯金字塔相关函数 ====================
# 用于提取图像的多尺度边缘特征（高频信息）

def gauss_kernel(channels=3, cuda=True):
    """
    创建5x5高斯卷积核
    基于二项式系数构建，用于图像平滑
    
    参数：
        channels: 输入图像通道数
        cuda: 是否放到GPU上
    
    返回：
        高斯卷积核 [channels, 1, 5, 5]
    """
    # 5x5高斯核，权重基于二项式分布
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.  # 归一化，使权重和为1
    kernel = kernel.repeat(channels, 1, 1, 1)  # 为每个通道复制一个核
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    """
    下采样：隔行隔列采样，将图像尺寸缩小一半
    [B, C, H, W] -> [B, C, H/2, W/2]
    """
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    """
    使用高斯核进行卷积平滑
    采用反射填充避免边界效应
    使用分组卷积，每个通道独立处理
    """
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')  # 边界反射填充
    out = F.conv2d(img, kernel, groups=img.shape[1])  # 分组卷积
    return out

def upsample(x, channels):
    """
    上采样：将图像尺寸扩大2倍
    方法：在像素间插入0，然后用高斯核平滑
    [B, C, H, W] -> [B, C, H*2, W*2]
    """
    # 在宽度方向插入0
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    # 在高度方向插入0
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    # 使用高斯核平滑（权重乘以4补偿插值）
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    """
    计算单层拉普拉斯差分
    原理：原图 - 降采样再上采样的图 = 高频细节（边缘信息）
    
    参数：
        img: 输入图像
        channels: 通道数
    
    返回：
        拉普拉斯差分图（高频信息）
    """
    filtered = conv_gauss(img, gauss_kernel(channels))  # 高斯平滑
    down = downsample(filtered)                         # 下采样
    up = upsample(down, channels)                       # 上采样回原尺寸
    # 确保尺寸一致
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up  # 计算差分，得到高频成分
    return diff

def make_laplace_pyramid(img, level, channels):
    """
    构建拉普拉斯金字塔
    金字塔每一层代表不同尺度的边缘信息
    
    参数：
        img: 输入图像
        level: 金字塔层数
        channels: 通道数
    
    返回：
        金字塔列表：[diff1, diff2, ..., diffN, residual]
        - diff: 各层的高频细节（边缘）
        - residual: 最底层的低频残差
    """
    current = img
    pyr = []
    for _ in range(level):
        # 计算当前层的拉普拉斯差分
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)  # 保存高频成分
        current = down    # 降采样后的图作为下一层输入
    pyr.append(current)   # 最后一层是低频残差
    return pyr


# ==================== 注意力机制模块 ====================
# CBAM (Convolutional Block Attention Module) 的实现

class ChannelGate(nn.Module):
    """
    通道注意力门
    功能：学习"看什么"，即哪些通道（特征）更重要
    
    原理：
    1. 通过全局平均池化和最大池化压缩空间维度
    2. 使用MLP学习通道间的依赖关系
    3. 生成通道注意力权重，重标定各通道的重要性
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # MLP：通道数先压缩再恢复，学习通道间关系
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # 降维
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)   # 升维
            )
            
    def forward(self, x):
        # 全局平均池化：捕获通道的平均激活强度
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        # 全局最大池化：捕获通道的最强激活
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        # 融合两种池化结果
        channel_att_sum = avg_out + max_out

        # 通过sigmoid生成0-1之间的注意力权重
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale  # 特征加权

class SpatialGate(nn.Module):
    """
    空间注意力门
    功能：学习"看哪里"，即空间位置上哪些区域更重要
    
    原理：
    1. 在通道维度进行最大池化和平均池化，压缩通道信息
    2. 拼接后通过卷积学习空间位置的重要性
    3. 生成空间注意力图，突出重要区域
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        # 7x7卷积学习空间注意力
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
        
    def forward(self, x):
        # 跨通道最大值：每个空间位置的最强响应
        # 跨通道平均值：每个空间位置的平均响应
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)  # 学习空间注意力图
        scale = torch.sigmoid(x_out)      # 生成0-1的权重图
        return x * scale  # 空间加权

class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力的双重注意力机制
    
    流程：
    1. 先应用通道注意力（学习看什么特征）
    2. 再应用空间注意力（学习看哪个位置）
    
    作用：
    - 增强有用特征，抑制无用特征
    - 突出重要区域，忽略背景干扰
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
        
    def forward(self, x):
        x_out = self.ChannelGate(x)   # 先通道注意力
        x_out = self.SpatialGate(x_out)  # 再空间注意力
        return x_out

# ==================== 边缘引导注意力模块 ====================
# EGA: Edge-Guided Attention Module (本文的核心创新)

class EGA(nn.Module):
    """
    边缘引导注意力模块（EGA）
    核心思想：利用边缘信息引导网络关注目标边界，提高分割精度
    
    设计理念：
    1. 反向注意力：关注背景区域（1-pred）
    2. 边界注意力：关注预测图的边缘（拉普拉斯算子提取边界）
    3. 高频特征：利用输入图像的边缘特征
    4. 三者融合后生成注意力图，增强特征表达
    
    工作流程：
    输入：edge_feature（输入图像边缘）, x（编码器特征）, pred（上一层预测）
    输出：增强后的特征图
    """
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        # 融合卷积：将3种特征（背景、边界、高频）融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        # 注意力生成：从融合特征中学习注意力图
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())  # 输出0-1的注意力权重

        # CBAM模块：进一步细化特征
        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        """
        前向传播
        
        参数：
            edge_feature: 输入图像的边缘特征（从拉普拉斯金字塔提取）
            x: 当前层的编码器特征
            pred: 上一层的预测图（用于生成边界注意力）
        
        返回：
            增强后的特征图
        """
        residual = x  # 保存残差连接
        xsize = x.size()[2:]  # 获取特征图尺寸
        pred = torch.sigmoid(pred)  # 将预测转为0-1概率
        
        # ========== 1. 反向注意力（Reverse Attention）==========
        # 目的：关注背景区域，避免模型过度关注前景
        background_att = 1 - pred  # 背景概率 = 1 - 前景概率
        background_x = x * background_att  # 背景特征
        
        # ========== 2. 边界注意力（Boundary Attention）==========
        # 目的：强调目标边界，提高分割边缘精度
        edge_pred = make_laplace(pred, 1)  # 用拉普拉斯算子提取预测图的边界
        pred_feature = x * edge_pred  # 边界特征

        # ========== 3. 高频特征（High-Frequency Feature）==========
        # 目的：利用输入图像的边缘信息指导特征学习
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input  # 输入边缘特征

        # ========== 特征融合 ==========
        # 将三种互补的特征拼接
        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)  # 融合为统一维度

        # ========== 注意力加权 ==========
        # 生成注意力图，突出重要区域
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map  # 注意力加权

        # ========== 残差连接 + CBAM ==========
        out = fusion_feature + residual  # 残差连接保留原始信息
        out = self.cbam(out)  # CBAM进一步细化特征
        return out