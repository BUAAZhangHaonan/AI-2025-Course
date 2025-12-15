# Copyright (c) OpenMMLab. All rights reserved.
"""
带深度注意力的ResNet Backbone
在ResNet的浅层使用深度信息进行注意力增强
"""
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.backbones import ResNetV1c

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fusion.depth_attention import DepthGuidedAttention


@MODELS.register_module()
class ResNetV1c_DepthAttention(BaseModule):
    """带深度注意力的ResNet Backbone

    该Backbone在ResNet的浅层阶段使用深度信息对RGB特征进行注意力增强。
    相比直接拼接RGBD通道，这种方法可以更好地利用预训练权重，
    并且通过注意力机制实现更灵活的多模态融合。

    Args:
        depth (int): ResNet深度，支持18, 34, 50, 101, 152
        fusion_stage (str): 融合位置，可选 'stem' 或 'stage1'
            - 'stem': 在ResNet stem之后融合（特征尺寸为H/4, W/4, 64通道）
            - 'stage1': 在ResNet stage1之后融合（特征尺寸为H/4, W/4, 256通道）
        attention_reduction (int): 注意力模块的通道缩减比例，默认16
        **kwargs: 传递给ResNetV1c的其他参数

    Example:
        >>> # 创建模型
        >>> model = ResNetV1c_DepthAttention(
        ...     depth=50,
        ...     fusion_stage='stem',
        ...     num_stages=4,
        ...     out_indices=(0, 1, 2, 3),
        ...     init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
        ... )
        >>> # 输入RGBD数据
        >>> x = torch.randn(2, 4, 512, 512)
        >>> outputs = model(x)
        >>> for i, out in enumerate(outputs):
        ...     print(f'Stage {i+1}: {out.shape}')
    """

    def __init__(self,
                 depth=50,
                 fusion_stage='stem',
                 attention_reduction=16,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 **kwargs):
        super().__init__()

        if fusion_stage not in ['stem', 'stage1']:
            raise ValueError(f"fusion_stage must be 'stem' or 'stage1', got {fusion_stage}")

        self.fusion_stage = fusion_stage
        self.num_stages = num_stages
        self.out_indices = out_indices

        # 创建RGB主干网络（保持3通道输入，可加载ImageNet预训练权重）
        self.rgb_backbone = ResNetV1c(
            depth=depth,
            num_stages=num_stages,
            out_indices=out_indices,
            **kwargs
        )

        # 确定融合点的特征通道数
        # ResNet的通道数配置
        channel_configs = {
            18: [64, 128, 256, 512],
            34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048],
            101: [256, 512, 1024, 2048],
            152: [256, 512, 1024, 2048]
        }

        if depth not in channel_configs:
            raise ValueError(f"Unsupported depth: {depth}")

        if fusion_stage == 'stem':
            # stem输出固定为64通道
            fusion_channels = 64
        elif fusion_stage == 'stage1':
            # stage1输出通道数取决于ResNet深度
            fusion_channels = channel_configs[depth][0]

        # 创建深度注意力模块
        self.depth_attention = DepthGuidedAttention(
            rgb_channels=fusion_channels,
            reduction=attention_reduction
        )

        print(f"[DepthAttention] Fusion at {fusion_stage}, channels={fusion_channels}")

    def forward(self, x):
        """前向传播

        Args:
            x (Tensor): 输入tensor，shape为 [B, 4, H, W]
                前3通道为RGB，第4通道为深度

        Returns:
            list[Tensor]: 多尺度特征列表，根据out_indices返回
        """
        # 分离RGB和深度通道
        rgb = x[:, :3, :, :]    # [B, 3, H, W]
        depth = x[:, 3:4, :, :]  # [B, 1, H, W]

        # 1. RGB通过stem（conv1, bn1, relu, maxpool）
        rgb_feat = self.rgb_backbone.stem(rgb)  # [B, 64, H/4, W/4]

        # 2. 如果选择在stem后融合
        if self.fusion_stage == 'stem':
            rgb_feat = self.depth_attention(rgb_feat, depth)

        # 3. RGB通过stage1（layer1）
        rgb_feat = self.rgb_backbone.layer1(rgb_feat)  # [B, 256/64, H/4, W/4]

        # 4. 如果选择在stage1后融合
        if self.fusion_stage == 'stage1':
            rgb_feat = self.depth_attention(rgb_feat, depth)

        # 5. 收集输出（根据out_indices）
        outs = []
        if 0 in self.out_indices:
            outs.append(rgb_feat)  # stage1输出

        # 6. 后续stages
        if self.num_stages > 1:
            x = self.rgb_backbone.layer2(rgb_feat)
            if 1 in self.out_indices:
                outs.append(x)

        if self.num_stages > 2:
            x = self.rgb_backbone.layer3(x)
            if 2 in self.out_indices:
                outs.append(x)

        if self.num_stages > 3:
            x = self.rgb_backbone.layer4(x)
            if 3 in self.out_indices:
                outs.append(x)

        return outs

    def init_weights(self):
        """初始化权重

        rgb_backbone会自动加载预训练权重（如果在init_cfg中指定）
        depth_attention会使用默认初始化
        """
        super().init_weights()
        # rgb_backbone的权重通过BaseModule的init_weights自动处理

    def train(self, mode=True):
        """设置训练/评估模式"""
        super().train(mode)
        self.rgb_backbone.train(mode)
        self.depth_attention.train(mode)
        return self
