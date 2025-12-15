# Copyright (c) OpenMMLab. All rights reserved.
"""
深度引导注意力融合模块
用于在浅层融合RGB和深度信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGuidedAttention(nn.Module):
    """深度引导的空间注意力模块

    该模块使用深度信息为RGB特征生成空间注意力权重，
    实现深度引导的特征增强。

    Args:
        rgb_channels (int): RGB特征的通道数
        reduction (int): 注意力模块的通道缩减比例，默认16

    Example:
        >>> import torch
        >>> # 创建模块
        >>> fusion = DepthGuidedAttention(rgb_channels=64, reduction=16)
        >>> # 输入RGB特征和深度图
        >>> rgb_feat = torch.randn(2, 64, 128, 128)
        >>> depth = torch.randn(2, 1, 512, 512)
        >>> # 前向传播
        >>> output = fusion(rgb_feat, depth)
        >>> print(output.shape)  # torch.Size([2, 64, 128, 128])
    """

    def __init__(self, rgb_channels, reduction=16):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.reduction = reduction

        # 深度特征提取分支
        # 将1通道深度图提取为多通道特征
        depth_feat_channels = rgb_channels // 4
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, depth_feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_feat_channels, depth_feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_feat_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力权重生成网络
        # 将RGB特征和深度特征拼接后生成注意力图
        total_channels = rgb_channels + depth_feat_channels
        self.attention = nn.Sequential(
            nn.Conv2d(total_channels, rgb_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels // reduction, rgb_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )

    def forward(self, rgb_feat, depth):
        """前向传播

        Args:
            rgb_feat (Tensor): RGB特征图，shape为 [B, C, H, W]
            depth (Tensor): 深度图，shape为 [B, 1, H', W']
                注意：深度图的空间尺寸可以与RGB特征不同，会自动调整

        Returns:
            Tensor: 融合后的特征图，shape为 [B, C, H, W]
        """
        batch_size, channels, height, width = rgb_feat.shape

        # 调整深度图尺寸以匹配RGB特征
        # 使用双线性插值进行上采样/下采样
        if depth.shape[2:] != rgb_feat.shape[2:]:
            depth = F.interpolate(
                depth,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )

        # 提取深度特征
        depth_feat = self.depth_conv(depth)  # [B, C//4, H, W]

        # 拼接RGB特征和深度特征
        combined = torch.cat([rgb_feat, depth_feat], dim=1)  # [B, C + C//4, H, W]

        # 生成空间注意力权重
        att_weight = self.attention(combined)  # [B, C, H, W], 值在0-1之间

        # 使用注意力权重增强RGB特征
        # 残差连接：out = rgb * att + rgb = rgb * (att + 1)
        out = rgb_feat * att_weight + rgb_feat

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(rgb_channels={self.rgb_channels}, '
                f'reduction={self.reduction})')
