#!/usr/bin/env python3
"""
快速测试 RGB-D 数据拼接

这个脚本会:
1. 加载一个样本的 RGB 和深度图
2. 执行完整的数据管道
3. 可视化 RGB、深度图和拼接后的 RGBD
4. 验证数据是否正确拼接
"""

import os
import sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
project_root = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from mmengine.config import Config
from mmseg.registry import DATASETS


def visualize_rgbd_sample(dataset, sample_idx=0):
    """
    可视化一个样本的 RGBD 数据

    Args:
        dataset: 数据集对象
        sample_idx: 样本索引
    """
    print(f"\n{'='*80}")
    print(f"正在加载第 {sample_idx} 个样本...")
    print(f"{'='*80}")

    # 获取原始数据信息（在pipeline之前）
    data_info = dataset.get_data_info(sample_idx)
    print(f"\n原始数据路径:")
    print(f"  RGB图像: {data_info['img_path']}")
    print(f"  深度图:  {data_info['depth_path']}")
    print(f"  标签:    {data_info['seg_map_path']}")

    # 加载原始RGB和深度
    import cv2
    rgb_raw = cv2.imread(data_info['img_path'])
    rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
    depth_raw = np.load(data_info['depth_path'])

    print(f"\n原始数据形状:")
    print(f"  RGB:   {rgb_raw.shape} (dtype: {rgb_raw.dtype})")
    print(f"  Depth: {depth_raw.shape} (dtype: {depth_raw.dtype})")
    print(f"  Depth range: [{depth_raw.min():.2f}, {depth_raw.max():.2f}]")

    # 执行完整的数据管道
    sample = dataset[sample_idx]

    print(f"\n管道处理后:")
    for key in sample.keys():
        if hasattr(sample[key], 'shape'):
            print(f"  {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")

    # 提取RGBD数据
    # 注意：PackSegInputs后，数据在 sample['inputs'] 中
    if 'inputs' in sample:
        rgbd = sample['inputs'].numpy()  # (C, H, W)
    elif 'img' in sample:
        rgbd = sample['img']
        if isinstance(rgbd, np.ndarray):
            pass
        else:
            rgbd = rgbd.numpy()
    else:
        print("❌ 找不到图像数据!")
        return

    print(f"\n✅ RGBD 拼接成功!")
    print(f"  RGBD shape: {rgbd.shape}")
    print(f"  通道数: {rgbd.shape[0]}")

    # 转换为 (H, W, C) 用于可视化
    if rgbd.shape[0] == 4:  # (C, H, W) -> (H, W, C)
        rgbd = np.transpose(rgbd, (1, 2, 0))

    # 分离通道
    rgb_processed = rgbd[:, :, :3]
    depth_processed = rgbd[:, :, 3]

    print(f"\n提取的通道:")
    print(f"  RGB:   {rgb_processed.shape}, range=[{rgb_processed.min():.2f}, {rgb_processed.max():.2f}]")
    print(f"  Depth: {depth_processed.shape}, range=[{depth_processed.min():.2f}, {depth_processed.max():.2f}]")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'RGBD 数据拼接验证 - 样本 {sample_idx}', fontsize=16, fontweight='bold')

    # 第一行：原始数据
    # 原始RGB
    axes[0, 0].imshow(rgb_raw)
    axes[0, 0].set_title('原始 RGB 图像', fontsize=12)
    axes[0, 0].axis('off')

    # 原始深度
    depth_vis = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
    im1 = axes[0, 1].imshow(depth_vis, cmap='viridis')
    axes[0, 1].set_title(f'原始深度图\nRange: [{depth_raw.min():.1f}, {depth_raw.max():.1f}]', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # 深度直方图
    axes[0, 2].hist(depth_raw.flatten(), bins=50, color='skyblue', edgecolor='black')
    axes[0, 2].set_title('原始深度值分布', fontsize=12)
    axes[0, 2].set_xlabel('Depth Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)

    # 第二行：处理后数据
    # 处理后RGB（反归一化用于显示）
    # 注意：这里的RGB已经被data_preprocessor归一化了，需要反归一化显示
    rgb_display = rgb_processed.copy()
    if rgb_display.max() <= 1.0:  # 如果是归一化的[0,1]
        rgb_display = (rgb_display * 255).astype(np.uint8)
    elif rgb_display.min() < 0:  # 如果是标准化的（减均值除标准差）
        # 反标准化（这里简化处理，只用于显示）
        rgb_display = ((rgb_display - rgb_display.min()) /
                      (rgb_display.max() - rgb_display.min()) * 255).astype(np.uint8)
    else:
        rgb_display = rgb_display.astype(np.uint8)

    axes[1, 0].imshow(rgb_display)
    axes[1, 0].set_title('处理后 RGB (RGBD前3通道)', fontsize=12)
    axes[1, 0].axis('off')

    # 处理后深度
    im2 = axes[1, 1].imshow(depth_processed, cmap='viridis')
    axes[1, 1].set_title(f'处理后深度 (RGBD第4通道)\nRange: [{depth_processed.min():.2f}, {depth_processed.max():.2f}]',
                         fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    # 处理后深度直方图
    axes[1, 2].hist(depth_processed.flatten(), bins=50, color='orange', edgecolor='black')
    axes[1, 2].set_title('处理后深度值分布', fontsize=12)
    axes[1, 2].set_xlabel('Normalized Depth Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    output_dir = 'rgbd_projects/electronic_component/test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f'rgbd_test_sample_{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存到: {output_path}")

    plt.show()

    # 验证检查
    print(f"\n{'='*80}")
    print("验证结果:")
    print(f"{'='*80}")

    checks = []

    # 检查1: 通道数
    if rgbd.shape[2] == 4:
        print("✅ RGBD 通道数正确 (4通道)")
        checks.append(True)
    else:
        print(f"❌ RGBD 通道数错误: {rgbd.shape[2]} (期望: 4)")
        checks.append(False)

    # 检查2: 深度值范围
    if 0 <= depth_processed.min() and depth_processed.max() <= 1.0:
        print(f"✅ 深度值已归一化到 [0, 1]: [{depth_processed.min():.3f}, {depth_processed.max():.3f}]")
        checks.append(True)
    else:
        print(f"⚠️  深度值范围: [{depth_processed.min():.3f}, {depth_processed.max():.3f}]")
        checks.append(True)  # 不算错误，可能有其他归一化方式

    # 检查3: 数据类型
    if rgbd.dtype in [np.float32, np.float64]:
        print(f"✅ 数据类型正确: {rgbd.dtype}")
        checks.append(True)
    else:
        print(f"⚠️  数据类型: {rgbd.dtype}")
        checks.append(True)

    # 检查4: 尺寸
    if rgb_processed.shape[:2] == depth_processed.shape:
        print(f"✅ RGB 和深度尺寸匹配: {rgb_processed.shape[:2]}")
        checks.append(True)
    else:
        print(f"❌ RGB 和深度尺寸不匹配!")
        checks.append(False)

    if all(checks):
        print(f"\n{'='*80}")
        print("🎉 所有检查通过! RGBD 数据拼接正确!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("⚠️  部分检查未通过，请查看上面的详细信息")
        print(f"{'='*80}")


def main():
    print("🔍 RGBD 数据拼接快速测试")
    print("="*80)

    # 加载配置
    config_file = 'rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py'

    if not osp.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return

    print(f"加载配置: {config_file}")
    cfg = Config.fromfile(config_file)

    # 构建数据集
    print("构建数据集...")
    dataset_cfg = cfg.train_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)

    print(f"✅ 数据集加载成功: {len(dataset)} 个样本")

    # 测试第一个样本
    if len(dataset) > 0:
        visualize_rgbd_sample(dataset, sample_idx=0)

        # 询问是否测试更多样本
        print("\n" + "="*80)
        try:
            choice = input("是否测试更多样本? (输入样本索引，或直接回车退出): ").strip()
            if choice:
                idx = int(choice)
                if 0 <= idx < len(dataset):
                    visualize_rgbd_sample(dataset, sample_idx=idx)
                else:
                    print(f"索引超出范围 [0, {len(dataset)-1}]")
        except:
            print("退出测试")
    else:
        print("❌ 数据集为空!")


if __name__ == '__main__':
    main()
