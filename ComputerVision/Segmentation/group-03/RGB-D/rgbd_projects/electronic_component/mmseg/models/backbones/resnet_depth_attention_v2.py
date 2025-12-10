# Copyright (c) OpenMMLab. All rights reserved.
"""
带改进版深度注意力的ResNet Backbone（稳定版）
核心改进：
1. 使用V2版本的注意力模块（深度特征内容参与融合）
2. 支持多种融合模式选择
3. 更稳定的训练策略
"""
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.backbones import ResNetV1c

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fusion.depth_attention_v2 import DepthGuidedAttentionV2, DepthGuidedAttentionV2Light


@MODELS.register_module()
class ResNetV1c_DepthAttentionV2(BaseModule):
    """带改进版深度注意力的ResNet Backbone（稳定版）

    相比V1版本的改进：
    ✅ 深度特征内容参与融合（不仅是权重调制）
    ✅ 通道数匹配（depth_feat与rgb_feat通道数一致）
    ✅ LayerNorm增强训练稳定性
    ✅ 可学习的残差权重alpha（初始值小，逐渐学习）
    ✅ 支持多种融合模式

    Args:
        depth (int): ResNet深度，支持18, 34, 50, 101, 152
        fusion_stage (str): 融合位置，可选 'stem' 或 'stage1'
            - 'stem': 在ResNet stem之后融合（特征尺寸为H/4, W/4, 64通道）
            - 'stage1': 在ResNet stage1之后融合（特征尺寸为H/4, W/4, 256通道）
        fusion_mode (str): 融合模式，可选 'residual', 'weighted', 'adaptive'
            - 'residual': out = rgb + alpha * (depth_feat * attention)  [推荐]
            - 'weighted': out = rgb * (1-att) + depth_feat * att
            - 'adaptive': out = rgb + alpha * depth_feat * attention
        attention_reduction (int): 注意力模块的通道缩减比例，默认16
        use_layer_norm (bool): 是否使用LayerNorm稳定训练，默认True
        init_alpha (float): 残差权重alpha的初始值，默认0.1
        use_light_version (bool): 是否使用轻量级版本（去除LayerNorm），默认False
        **kwargs: 传递给ResNetV1c的其他参数

    Example:
        >>> model = ResNetV1c_DepthAttentionV2(
        ...     depth=50,
        ...     fusion_stage='stem',
        ...     fusion_mode='residual',
        ...     use_layer_norm=True,
        ...     init_alpha=0.1
        ... )
    """

    def __init__(self,
                 depth=50,
                 fusion_stage='stem',
                 fusion_mode='residual',
                 attention_reduction=16,
                 use_layer_norm=True,
                 init_alpha=0.1,
                 use_light_version=False,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 **kwargs):
        super().__init__()

        if fusion_stage not in ['stem', 'stage1']:
            raise ValueError(f"fusion_stage must be 'stem' or 'stage1', got {fusion_stage}")

        self.fusion_stage = fusion_stage
        self.fusion_mode = fusion_mode
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.use_light_version = use_light_version

        # 创建RGB主干网络（保持3通道输入，可加载ImageNet预训练权重）
        self.rgb_backbone = ResNetV1c(
            depth=depth,
            num_stages=num_stages,
            out_indices=out_indices,
            **kwargs
        )

        # 确定融合点的特征通道数
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
            fusion_channels = 64
        elif fusion_stage == 'stage1':
            fusion_channels = channel_configs[depth][0]

        # 创建改进版深度注意力模块
        if use_light_version:
            self.depth_attention = DepthGuidedAttentionV2Light(
                rgb_channels=fusion_channels,
                fusion_mode=fusion_mode,
                reduction=attention_reduction,
                init_alpha=init_alpha
            )
            print(f"[DepthAttentionV2-Light] Fusion at {fusion_stage}, "
                  f"channels={fusion_channels}, mode={fusion_mode}, alpha={init_alpha}")
        else:
            self.depth_attention = DepthGuidedAttentionV2(
                rgb_channels=fusion_channels,
                fusion_mode=fusion_mode,
                reduction=attention_reduction,
                use_layer_norm=use_layer_norm,
                init_alpha=init_alpha
            )
            print(f"[DepthAttentionV2] Fusion at {fusion_stage}, "
                  f"channels={fusion_channels}, mode={fusion_mode}, "
                  f"LayerNorm={use_layer_norm}, alpha={init_alpha}")

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

        # 1. RGB通过stem
        rgb_feat = self.rgb_backbone.stem(rgb)  # [B, 64, H/4, W/4]

        # 2. 如果选择在stem后融合
        if self.fusion_stage == 'stem':
            rgb_feat = self.depth_attention(rgb_feat, depth)

        # 3. RGB通过stage1
        rgb_feat = self.rgb_backbone.layer1(rgb_feat)

        # 4. 如果选择在stage1后融合
        if self.fusion_stage == 'stage1':
            rgb_feat = self.depth_attention(rgb_feat, depth)

        # 5. 收集输出
        outs = []
        if 0 in self.out_indices:
            outs.append(rgb_feat)

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
        """初始化权重"""
        super().init_weights()

    def train(self, mode=True):
        """设置训练/评估模式"""
        super().train(mode)
        self.rgb_backbone.train(mode)
        self.depth_attention.train(mode)
        return self

    def get_alpha_value(self):
        """获取当前alpha值（用于监控）"""
        return self.depth_attention.alpha.item()
