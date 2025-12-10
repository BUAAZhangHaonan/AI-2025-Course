#!/usr/bin/env python3
"""
测试已保存的DeepCrack模型
"""

import os
import sys
from pathlib import Path

# 获取项目根目录并切换工作目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # new_projects/deepcrack/scripts/ -> work_1/
os.chdir(PROJECT_ROOT)  # 切换到项目根目录，所有路径相对于此

import argparse
import numpy as np
import cv2
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model
from mmseg.registry import MODELS
import torch

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试DeepCrack模型')
    parser.add_argument('--config', 
                       default='new_projects/deepcrack/configs/pspnet_r50-deepcrack_512x512_40k_strong_reg.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/deepcrack_pspnet_optimized_v2/best_mIoU_iter_350.pth',
                       help='模型检查点路径')
    parser.add_argument('--test-img', 
                       default='data/DeepCrack/test_img',
                       help='测试图像目录')
    parser.add_argument('--output-dir', 
                       default='work_dirs/deepcrack_pspnet_optimized_v2/test_results',
                       help='输出目录')
    parser.add_argument('--num-samples', 
                       type=int, 
                       default=5,
                       help='测试样本数量')
    
    return parser.parse_args()

def visualize_result(img, result, output_path):
    """
    可视化分割结果
    
    Args:
        img: 原始图像
        result: 分割结果
        output_path: 输出路径
    """
    # 获取预测的分割图
    pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
    
    # 创建彩色可视化
    # 背景：黑色(0)，裂缝：白色(255)
    vis_mask = (pred_mask * 255).astype(np.uint8)
    
    # 将原图转换为BGR（如果是RGB）
    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # 创建叠加图像
    overlay = img_bgr.copy()
    # 将裂缝区域标记为红色
    overlay[pred_mask == 1] = [0, 0, 255]  # BGR格式的红色
    
    # 混合原图和叠加图
    alpha = 0.5
    blended = cv2.addWeighted(img_bgr, 1-alpha, overlay, alpha, 0)
    
    # 拼接显示：原图 | 预测掩码 | 叠加图
    h, w = img_bgr.shape[:2]
    vis_result = np.zeros((h, w*3, 3), dtype=np.uint8)
    vis_result[:, :w] = img_bgr
    vis_result[:, w:2*w] = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
    vis_result[:, 2*w:] = blended
    
    # 保存结果
    cv2.imwrite(output_path, vis_result)
    print(f"  保存可视化结果: {output_path}")

def test_model_simple(args):
    """
    简单测试模型（推理模式）
    """
    print("=" * 60)
    print("DeepCrack模型测试（推理模式）")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    print(f"\n配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"测试图像目录: {args.test_img}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}")
    
    # 初始化模型
    print("\n初始化模型...")
    try:
        model = init_model(args.config, args.checkpoint, device='cuda:0')
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return
    
    # 获取测试图像列表
    if not os.path.exists(args.test_img):
        raise FileNotFoundError(f"测试图像目录不存在: {args.test_img}")
    
    img_files = [f for f in os.listdir(args.test_img) if f.endswith('.jpg')]
    img_files = sorted(img_files)[:args.num_samples]
    
    print(f"\n找到 {len(img_files)} 张测试图像")
    print("=" * 60)
    
    # 测试每张图像
    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(args.test_img, img_file)
        print(f"\n[{idx}/{len(img_files)}] 测试: {img_file}")
        
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ✗ 无法读取图像: {img_path}")
                continue
            
            print(f"  图像尺寸: {img.shape}")
            
            # 推理
            result = inference_model(model, img)
            
            # 获取预测结果
            pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
            
            # 统计裂缝像素
            crack_pixels = np.sum(pred_mask == 1)
            total_pixels = pred_mask.size
            crack_ratio = crack_pixels / total_pixels * 100
            
            print(f"  预测结果:")
            print(f"    - 裂缝像素: {crack_pixels}/{total_pixels} ({crack_ratio:.2f}%)")
            print(f"    - 预测形状: {pred_mask.shape}")
            
            # 可视化并保存
            output_name = os.path.splitext(img_file)[0] + '_result.jpg'
            output_path = os.path.join(args.output_dir, output_name)
            visualize_result(img, result, output_path)
            
            print(f"  ✓ 测试成功")
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)
    print(f"\n结果保存在: {args.output_dir}")
    print("\n可视化说明:")
    print("  - 左侧: 原始图像")
    print("  - 中间: 预测掩码（白色=裂缝，黑色=背景）")
    print("  - 右侧: 叠加图像（红色=预测的裂缝）")

def main():
    """主函数"""
    args = parse_args()
    
    try:
        test_model_simple(args)
        return 0
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

