#!/usr/bin/env python3
"""
验证 RGBD 配置和数据加载

这个脚本用于验证:
1. 配置文件是否正确
2. 数据加载是否正常
3. RGBD拼接是否成功
4. 模型是否能正确初始化
"""

import os
import sys
import os.path as osp
import numpy as np
import torch

# 添加项目路径
project_root = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from mmengine.config import Config
from mmseg.registry import DATASETS, TRANSFORMS, MODELS


def check_config():
    """检查配置文件"""
    print("=" * 80)
    print("1. 检查配置文件")
    print("=" * 80)

    config_file = 'rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py'

    if not osp.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return None

    try:
        cfg = Config.fromfile(config_file)
        print(f"✅ 配置文件加载成功: {config_file}")
        return cfg
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None


def check_dataset(cfg):
    """检查数据集"""
    print("\n" + "=" * 80)
    print("2. 检查数据集")
    print("=" * 80)

    try:
        # 构建训练数据集
        dataset_cfg = cfg.train_dataloader.dataset
        dataset = DATASETS.build(dataset_cfg)

        print(f"✅ 数据集构建成功: {dataset.__class__.__name__}")
        print(f"   - 数据根目录: {dataset.data_root}")
        print(f"   - 训练样本数: {len(dataset)}")

        # 检查第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n   样本数据结构:")
            for key in sample.keys():
                if hasattr(sample[key], 'shape'):
                    print(f"   - {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
                else:
                    print(f"   - {key}: {type(sample[key])}")

            # 检查是否是4通道
            if 'img' in sample:
                img_channels = sample['img'].shape[0] if len(sample['img'].shape) == 3 else sample['img'].shape[-1]
                if img_channels == 4:
                    print(f"\n   ✅ RGBD 拼接成功! 图像通道数: {img_channels}")
                else:
                    print(f"\n   ⚠️  警告: 图像通道数为 {img_channels}, 期望为 4")

        return True

    except Exception as e:
        print(f"❌ 数据集检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_transforms():
    """检查自定义 transforms"""
    print("\n" + "=" * 80)
    print("3. 检查自定义 Transforms")
    print("=" * 80)

    transforms_to_check = [
        'LoadDepthFromFile',
        'ConcatRGBD',
        'ConvertInstanceToSemantic'
    ]

    for transform_name in transforms_to_check:
        if transform_name in TRANSFORMS._module_dict:
            print(f"✅ {transform_name} 已注册")
        else:
            print(f"❌ {transform_name} 未注册")


def check_model(cfg):
    """检查模型"""
    print("\n" + "=" * 80)
    print("4. 检查模型")
    print("=" * 80)

    try:
        # 检查 backbone 是否注册
        backbone_type = cfg.model.backbone.type
        if backbone_type in MODELS._module_dict:
            print(f"✅ Backbone '{backbone_type}' 已注册")
        else:
            print(f"❌ Backbone '{backbone_type}' 未注册")
            return False

        # 构建模型
        model = MODELS.build(cfg.model)
        print(f"✅ 模型构建成功")

        # 测试前向传播
        dummy_input = torch.randn(1, 4, 512, 512)  # RGBD 输入
        model.eval()

        with torch.no_grad():
            # 只测试编码器
            if hasattr(model, 'backbone'):
                features = model.backbone(dummy_input)
                print(f"\n   Backbone 输出特征:")
                for i, feat in enumerate(features):
                    print(f"   - Stage {i}: {feat.shape}")
                print(f"\n   ✅ 模型前向传播成功!")
            else:
                print(f"   ⚠️  模型没有 backbone 属性")

        return True

    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_depth_data():
    """检查深度数据可用性"""
    print("\n" + "=" * 80)
    print("5. 检查深度数据")
    print("=" * 80)

    depth_dirs = [
        'data/electronic_component/depth/depth_npy/train',
        'data/electronic_component/depth/depth_npy/val',
        'data/electronic_component/depth/depth_npy/test'
    ]

    for depth_dir in depth_dirs:
        if osp.exists(depth_dir):
            npy_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
            print(f"✅ {depth_dir}: {len(npy_files)} 个深度文件")

            # 检查第一个文件
            if npy_files:
                sample_file = osp.join(depth_dir, npy_files[0])
                try:
                    depth = np.load(sample_file)
                    print(f"   样本: {npy_files[0]}")
                    print(f"   - Shape: {depth.shape}")
                    print(f"   - Dtype: {depth.dtype}")
                    print(f"   - Range: [{depth.min():.2f}, {depth.max():.2f}]")
                except Exception as e:
                    print(f"   ❌ 读取失败: {e}")
        else:
            print(f"❌ {depth_dir}: 目录不存在")


def main():
    print("\n" + "🔍 RGBD 配置验证脚本")
    print("=" * 80)

    # 1. 检查配置
    cfg = check_config()
    if cfg is None:
        return

    # 2. 检查深度数据
    check_depth_data()

    # 3. 检查 transforms
    check_transforms()

    # 4. 检查数据集
    dataset_ok = check_dataset(cfg)

    # 5. 检查模型
    if dataset_ok:
        check_model(cfg)

    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
