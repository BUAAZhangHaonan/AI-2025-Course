# Copyright (c) OpenMMLab. All rights reserved.
"""
RGBD ResNet Backbone

这个模块实现了支持4通道RGBD输入的ResNet骨干网络。
通过修改第一个卷积层的输入通道数从3改为4，支持RGB + Depth融合。

Key modifications:
1. 将conv1的in_channels从3改为4
2. 从预训练的RGB模型初始化时，对深度通道进行特殊初始化
"""

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmseg.registry import MODELS
from mmseg.models.backbones import ResNetV1c


@MODELS.register_module()
class ResNetV1c_RGBD(ResNetV1c):
    """
    ResNetV1c backbone adapted for 4-channel RGBD input.

    This class extends the standard ResNetV1c to accept 4-channel input (RGB + Depth)
    instead of the default 3-channel RGB input.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 4 (RGBD).
        depth_init_method (str): Method to initialize depth channel weights.
            Options: 'zero', 'mean', 'copy_red'. Default: 'mean'.
        **kwargs: Other arguments for ResNetV1c.

    Example:
        >>> model = ResNetV1c_RGBD(depth=50, in_channels=4)
    """

    def __init__(self,
                 depth,
                 in_channels=4,
                 depth_init_method='mean',
                 **kwargs):
        # 强制设置in_channels为4
        kwargs['in_channels'] = in_channels
        self.depth_init_method = depth_init_method

        # 调用父类初始化
        super(ResNetV1c_RGBD, self).__init__(depth=depth, **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        """
        Override stem layer creation to support 4-channel input.

        ResNetV1c uses three 3x3 convolutions in the stem (deep_stem=True):
        - conv1: in_channels -> stem_channels // 2
        - conv2: stem_channels // 2 -> stem_channels // 2
        - conv3: stem_channels // 2 -> stem_channels

        Creates a Sequential module named 'stem' to match parent class behavior.
        """
        # Create stem as Sequential module (required for ResNetV1c with deep_stem=True)
        self.stem = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,  # 4 for RGBD
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                stem_channels // 2,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, stem_channels)[1],
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        """
        Initialize weights for RGBD backbone.

        For the first conv layer in stem, we initialize:
        - RGB channels: from pretrained weights if available
        - Depth channel: using specified initialization method
        """
        # Call parent's init_weights to load pretrained RGB weights
        super(ResNetV1c_RGBD, self).init_weights()

        # Get the first conv layer from the stem Sequential module
        first_conv = self.stem[0]  # stem[0] is the first conv layer

        # Special initialization for the 4th channel (depth)
        if first_conv.in_channels == 4:
            with torch.no_grad():
                # Get the pretrained RGB weights
                if hasattr(self, 'init_cfg') and self.init_cfg is not None:
                    # If pretrained weights exist, the first 3 channels are already initialized
                    # We only need to initialize the 4th channel (depth)

                    if self.depth_init_method == 'zero':
                        # Initialize depth channel to zero
                        first_conv.weight[:, 3:4, :, :] = 0.0
                        print('[RGBD] Depth channel initialized to zero')

                    elif self.depth_init_method == 'mean':
                        # Initialize depth channel as mean of RGB channels
                        rgb_weights = first_conv.weight[:, :3, :, :].clone()
                        mean_weights = rgb_weights.mean(dim=1, keepdim=True)
                        first_conv.weight[:, 3:4, :, :] = mean_weights
                        print('[RGBD] Depth channel initialized as mean of RGB')

                    elif self.depth_init_method == 'copy_red':
                        # Copy red channel weights to depth channel
                        first_conv.weight[:, 3:4, :, :] = first_conv.weight[:, 0:1, :, :].clone()
                        print('[RGBD] Depth channel initialized by copying red channel')

                    else:
                        raise ValueError(f'Unknown depth_init_method: {self.depth_init_method}')
                else:
                    # No pretrained weights, use default initialization
                    print('[RGBD] No pretrained weights, using default initialization')
