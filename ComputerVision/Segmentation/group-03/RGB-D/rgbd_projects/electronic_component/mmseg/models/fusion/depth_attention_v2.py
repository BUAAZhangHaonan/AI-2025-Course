# Copyright (c) OpenMMLab. All rights reserved.
"""
改进版深度引导注意力融合模块（稳定版）
核心改进：
1. 修正融合公式 - 深度特征内容参与融合
2. 通道数匹配 - depth_feat与rgb_feat通道数一致
3. 增强稳定性 - LayerNorm + 可学习残差权重
4. 控制学习速率 - 可选的scale参数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGuidedAttentionV2(nn.Module):
    """改进版深度引导注意力模块（稳定版）

    核心改进：
    - ✅ 深度特征内容参与融合（不仅仅是权重调制）
    - ✅ 通道数匹配（depth_feat与rgb_feat通道数一致）
    - ✅ LayerNorm增强稳定性
    - ✅ 可学习的残差权重alpha
    - ✅ 支持多种融合模式

    Args:
        rgb_channels (int): RGB特征的通道数
        fusion_mode (str): 融合模式
            - 'residual': out = rgb + alpha * (depth_feat * attention)  [推荐]
            - 'weighted': out = rgb * (1 - attention) + depth_feat * attention
            - 'adaptive': out = rgb + alpha * depth_feat * attention
        reduction (int): 注意力模块的通道缩减比例，默认16
        use_layer_norm (bool): 是否使用LayerNorm稳定训练，默认True
        init_alpha (float): 残差权重a lpha的初始值，默认0.1（小值有助于稳定训练）

    Example:
        >>> # 创建模块
        >>> fusion = DepthGuidedAttentionV2(
        ...     rgb_channels=64,
        ...     fusion_mode='residual',
        ...     use_layer_norm=True
        ... )
        >>> # 输入RGB特征和深度图
        >>> rgb_feat = torch.randn(2, 64, 128, 128)
        >>> depth = torch.randn(2, 1, 512, 512)
        >>> # 前向传播
        >>> output = fusion(rgb_feat, depth)
        >>> print(output.shape)  # torch.Size([2, 64, 128, 128])
    """

    def __init__(self,
                 rgb_channels,
                 fusion_mode='residual',
                 reduction=16,
                 use_layer_norm=True,
                 init_alpha=0.1):
        super().__init__()

        if fusion_mode not in ['residual', 'weighted', 'adaptive']:
            raise ValueError(f"fusion_mode must be 'residual', 'weighted', or 'adaptive', got {fusion_mode}")

        self.rgb_channels = rgb_channels
        self.fusion_mode = fusion_mode
        self.reduction = reduction
        self.use_layer_norm = use_layer_norm

        # 深度特征提取分支
        # 关键改进：输出通道数与RGB特征一致（rgb_channels）
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, rgb_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels // 2, rgb_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力权重生成网络
        # 输入：RGB特征 + 深度特征（通道数均为rgb_channels）
        total_channels = rgb_channels * 2
        self.attention = nn.Sequential(
            nn.Conv2d(total_channels, rgb_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels // reduction, rgb_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )

        # LayerNorm用于稳定训练（可选）
        if self.use_layer_norm:
            self.rgb_norm = nn.LayerNorm(rgb_channels)
            self.depth_norm = nn.LayerNorm(rgb_channels)
            self.out_norm = nn.LayerNorm(rgb_channels)

        # 可学习的残差权重alpha
        # 初始化为小值（如0.1），让模型逐渐学习深度特征的贡献
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb_feat, depth):
        """前向传播

        Args:
            rgb_feat (Tensor): RGB特征图，shape为 [B, C, H, W]
            depth (Tensor): 深度图，shape为 [B, 1, H', W']

        Returns:
            Tensor: 融合后的特征图，shape为 [B, C, H, W]
        """
        batch_size, channels, height, width = rgb_feat.shape

        # 1. 调整深度图尺寸以匹配RGB特征
        if depth.shape[2:] != rgb_feat.shape[2:]:
            depth = F.interpolate(
                depth,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )

        # 2. 提取深度特征（通道数与RGB一致）
        depth_feat = self.depth_conv(depth)  # [B, C, H, W]

        # 3. 可选的LayerNorm（增强稳定性）
        if self.use_layer_norm:
            # LayerNorm需要 [B, H, W, C] 格式
            rgb_feat_norm = rgb_feat.permute(0, 2, 3, 1)
            rgb_feat_norm = self.rgb_norm(rgb_feat_norm)
            rgb_feat_norm = rgb_feat_norm.permute(0, 3, 1, 2)

            depth_feat_norm = depth_feat.permute(0, 2, 3, 1)
            depth_feat_norm = self.depth_norm(depth_feat_norm)
            depth_feat_norm = depth_feat_norm.permute(0, 3, 1, 2)
        else:
            rgb_feat_norm = rgb_feat
            depth_feat_norm = depth_feat

        # 4. 拼接并生成注意力权重
        combined = torch.cat([rgb_feat_norm, depth_feat_norm], dim=1)  # [B, 2C, H, W]
        attention = self.attention(combined)  # [B, C, H, W], 值在0-1之间

        # 5. 根据融合模式进行融合
        if self.fusion_mode == 'residual':
            # 残差融合：out = rgb + alpha * (depth_feat * attention)
            # 深度特征的内容通过attention加权后，以残差方式加到RGB上
            out = rgb_feat + self.alpha * (depth_feat * attention)

        elif self.fusion_mode == 'weighted':
            # 加权融合：out = rgb * (1 - attention) + depth_feat * attention
            # RGB和深度特征按attention权重进行加权平均
            out = rgb_feat * (1 - attention) + depth_feat * attention

        elif self.fusion_mode == 'adaptive':
            # 自适应融合：out = rgb + alpha * depth_feat * attention
            # 与residual类似，但直接使用depth_feat_norm
            out = rgb_feat + self.alpha * (depth_feat_norm * attention)

        # 6. 可选的输出LayerNorm
        if self.use_layer_norm:
            out = out.permute(0, 2, 3, 1)
            out = self.out_norm(out)
            out = out.permute(0, 3, 1, 2)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(rgb_channels={self.rgb_channels}, '
                f'fusion_mode={self.fusion_mode}, '
                f'reduction={self.reduction}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'alpha={self.alpha.item():.4f})')


class DepthGuidedAttentionV2Light(nn.Module):
    """轻量级改进版深度注意力模块

    相比V2完整版，去除LayerNorm以减少计算量，
    适合对速度有要求的场景。

    Args:
        rgb_channels (int): RGB特征的通道数
        fusion_mode (str): 融合模式，默认'residual'
        reduction (int): 注意力模块的通道缩减比例，默认16
        init_alpha (float): 残差权重alpha的初始值，默认0.1
    """

    def __init__(self,
                 rgb_channels,
                 fusion_mode='residual',
                 reduction=16,
                 init_alpha=0.1):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.fusion_mode = fusion_mode

        # 深度特征提取（通道数与RGB匹配）
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, rgb_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels // 2, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(rgb_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels // reduction, rgb_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 可学习残差权重
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, rgb_feat, depth):
        B, C, H, W = rgb_feat.shape

        # 调整深度图尺寸
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

        # 提取深度特征
        depth_feat = self.depth_conv(depth)

        # 生成注意力
        combined = torch.cat([rgb_feat, depth_feat], dim=1)
        attention = self.attention(combined)

        # 融合
        if self.fusion_mode == 'residual':
            out = rgb_feat + self.alpha * (depth_feat * attention)
        elif self.fusion_mode == 'weighted':
            out = rgb_feat * (1 - attention) + depth_feat * attention
        else:
            out = rgb_feat + self.alpha * (depth_feat * attention)

        return out
