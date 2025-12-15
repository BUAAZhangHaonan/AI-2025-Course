#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electronic Component模型测试和可视化脚本

此脚本用于测试训练好的Electronic Component分割模型，
并生成可视化结果。
"""

import os
import sys
from pathlib import Path

# 获取项目根目录并切换工作目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # rgbd_projects/electronic_component/scripts/ -> work_1/
os.chdir(PROJECT_ROOT)  # 切换到项目根目录，所有路径相对于此

import argparse
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.apis import init_model, inference_model
from mmseg.registry import MODELS

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Test Electronic Component Segmentation Model')
    
    parser.add_argument(
        '--config',
        default='rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_512x512_10k.py',
        help='测试配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/electronic_component_pspnet/best_mIoU_iter_6500.pth',
        help='模型检查点文件路径'
    )
    parser.add_argument(
        '--data-root',
        default='data/electronic_component',
        help='数据集根目录'
    )
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_ROOT / 'work_dirs' / 'electronic_component_pspnet' / 'visualizations'),
        help='可视化结果输出目录'
    )
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help='测试数据集划分'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='可视化样本数量'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='计算设备'
    )
    
    args = parser.parse_args()
    return args


def visualize_result(image, pred_mask, gt_mask=None, alpha=0.5):
    """
    可视化分割结果
    
    Args:
        image: 原始图像 (H, W, 3)
        pred_mask: 预测掩码 (H, W), 0=背景, 1=电子元件
        gt_mask: 真实掩码 (H, W), 可选
        alpha: 叠加透明度
        
    Returns:
        vis_image: 可视化图像
    """
    # 调整图像大小以匹配掩码
    if image.shape[:2] != pred_mask.shape:
        image = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))
    
    # 创建彩色掩码
    # 背景: 黑色 (0,0,0)
    # 电子元件: 红色 (0,0,255)
    pred_color = np.zeros_like(image)
    pred_color[pred_mask == 1] = [0, 0, 255]  # BGR格式：红色
    
    # 叠加原图和预测掩码
    overlay = cv2.addWeighted(image, 1-alpha, pred_color, alpha, 0)
    
    if gt_mask is not None:
        # 调整真实掩码大小
        if image.shape[:2] != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # 创建真实掩码彩色图
        gt_color = np.zeros_like(image)
        gt_color[gt_mask == 1] = [0, 255, 0]  # BGR格式：绿色
        
        # 创建对比图：真实(绿) vs 预测(红)
        # 正确预测: 黄色 (绿+红)
        # 仅真实: 绿色
        # 仅预测: 红色
        comparison = np.zeros_like(image)
        comparison[gt_mask == 1] = [0, 255, 0]  # 真实：绿色
        comparison[pred_mask == 1] = comparison[pred_mask == 1] + [0, 0, 255]  # 预测：加红色
        
        # 拼接：原图 | 预测叠加 | GT叠加 | 对比图
        gt_overlay = cv2.addWeighted(image, 1-alpha, gt_color, alpha, 0)
        vis_image = np.hstack([image, overlay, gt_overlay, comparison])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, 'Prediction', (image.shape[1]+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, 'Ground Truth', (2*image.shape[1]+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, 'Comparison', (3*image.shape[1]+10, 30), font, 1, (255, 255, 255), 2)
    else:
        # 仅拼接原图和预测
        vis_image = np.hstack([image, overlay])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, 'Prediction', (image.shape[1]+10, 30), font, 1, (255, 255, 255), 2)
    
    return vis_image


def main():
    """主函数"""
    # 切换到项目根目录，确保相对路径正确
    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("Electronic Component模型测试和可视化")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print(f"检查点:   {args.checkpoint}")
    print(f"数据集:   {args.data_root}/{args.split}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备:     {args.device}")
    print(f"工作目录: {PROJECT_ROOT}")
    print("=" * 70 + "\n")
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化默认作用域
    init_default_scope('mmseg')
    
    # 加载模型
    print("🔧 加载模型...")
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
        print("✅ 模型加载成功！\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 获取测试图像列表
    img_dir = os.path.join(args.data_root, 'images', args.split)
    mask_dir = os.path.join(args.data_root, 'mask', args.split)
    
    if not os.path.exists(img_dir):
        print(f"❌ 图像目录不存在: {img_dir}")
        sys.exit(1)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    if len(img_files) == 0:
        print(f"❌ 未找到图像文件: {img_dir}")
        sys.exit(1)
    
    print(f"📁 找到 {len(img_files)} 张测试图像")
    
    # 随机选择样本
    num_samples = min(args.num_samples, len(img_files))
    np.random.seed(42)
    selected_indices = np.random.choice(len(img_files), num_samples, replace=False)
    
    print(f"🎯 测试 {num_samples} 张图像\n")
    
    # 测试模型
    for i, idx in enumerate(selected_indices):
        img_file = img_files[idx]
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        
        print(f"[{i+1}/{num_samples}] 处理: {img_file}")
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ 无法读取图像: {img_path}")
            continue
        
        # 推理
        result = inference_model(model, img_path)
        pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
        
        # 加载真实掩码（如果存在）
        gt_mask = None
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # 转换实例标签为语义标签
            gt_mask_semantic = np.zeros_like(gt_mask)
            gt_mask_semantic[(gt_mask > 0) & (gt_mask < 255)] = 1
            gt_mask = gt_mask_semantic
        
        # 可视化
        vis_image = visualize_result(image, pred_mask, gt_mask, alpha=0.6)
        
        # 保存结果
        output_path = os.path.join(args.output_dir, f'vis_{img_file}')
        cv2.imwrite(output_path, vis_image)
        print(f"  ✅ 保存: {output_path}")
        
        # 计算IoU（如果有GT）
        if gt_mask is not None:
            intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
            union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()
            iou = intersection / union if union > 0 else 0
            print(f"  📊 IoU (component): {iou:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    print("=" * 70 + "\n")
    
    # 恢复原始目录
    os.chdir(original_dir)


if __name__ == '__main__':
    main()

