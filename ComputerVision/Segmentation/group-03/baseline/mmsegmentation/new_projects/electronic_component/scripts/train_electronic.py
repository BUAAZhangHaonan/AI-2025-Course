#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electronic Component数据集训练脚本

此脚本用于训练Electronic Component数据集的语义分割模型。
"""

import os
import sys
from pathlib import Path

# 获取项目根目录并切换工作目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # new_projects/electronic_component/scripts/ -> work_1/
os.chdir(PROJECT_ROOT)  # 切换到项目根目录，所有路径相对于此

import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='Train Electronic Component Segmentation Model')
    
    parser.add_argument(
        '--config',
        default='new_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_512x512_10k.py',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--work-dir',
        default='work_dirs/electronic_component_pspnet',
        help='工作目录，用于保存日志和模型'
    )
    parser.add_argument(
        '--data-root',
        default='data/electronic_component',
        help='数据集根目录'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小（可选，覆盖配置文件）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='是否从最新检查点恢复训练'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='是否使用自动混合精度训练'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='随机种子'
    )
    
    args = parser.parse_args()
    return args


def setup_config(args):
    """
    设置配置文件
    
    Args:
        args: 命令行参数
        
    Returns:
        Config: 配置对象
    """
    print("=" * 50)
    print("设置训练配置")
    print("=" * 50)
    
    # 加载配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    print(f"加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 设置工作目录
    cfg.work_dir = args.work_dir
    print(f"工作目录: {cfg.work_dir}")
    
    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 数据根目录
    print(f"数据根目录: {args.data_root}")
    
    # 设置批次大小
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
        print(f"批次大小: {args.batch_size}")
    
    # 设置随机种子
    cfg.randomness = dict(seed=args.seed)
    print(f"随机种子: {args.seed}")
    
    # 设置自动混合精度
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
        print("启用自动混合精度训练")
    
    # 设置恢复训练
    if args.resume:
        cfg.resume = True
        print("将从最新检查点恢复训练")
    
    return cfg


def check_data_availability(data_root):
    """
    检查Electronic Component数据集的可用性
    
    Args:
        data_root: 数据根目录
        
    Returns:
        bool: 数据是否可用
    """
    print("\n" + "=" * 50)
    print("检查数据可用性")
    print("=" * 50)
    
    train_img_dir = os.path.join(data_root, 'images/train')
    train_mask_dir = os.path.join(data_root, 'mask/train')
    val_img_dir = os.path.join(data_root, 'images/val')
    val_mask_dir = os.path.join(data_root, 'mask/val')
    test_img_dir = os.path.join(data_root, 'images/test')
    test_mask_dir = os.path.join(data_root, 'mask/test')
    
    # 检查训练数据
    if not os.path.exists(train_img_dir):
        print(f"❌ 训练图像目录不存在: {train_img_dir}")
        return False
    print(f"✅ 检查训练数据目录: {train_img_dir}")
    train_img_count = len([f for f in os.listdir(train_img_dir) if f.endswith('.png')])
    print(f"   训练图像数量: {train_img_count}")
    
    if not os.path.exists(train_mask_dir):
        print(f"❌ 训练掩码目录不存在: {train_mask_dir}")
        return False
    print(f"✅ 检查训练标签目录: {train_mask_dir}")
    train_mask_count = len([f for f in os.listdir(train_mask_dir) if f.endswith('.png')])
    print(f"   训练掩码数量: {train_mask_count}")
    
    if train_img_count != train_mask_count:
        print(f"⚠️  警告：训练图像和掩码数量不匹配！")
    
    # 检查验证数据
    if not os.path.exists(val_img_dir):
        print(f"❌ 验证图像目录不存在: {val_img_dir}")
        return False
    print(f"✅ 检查验证数据目录: {val_img_dir}")
    val_img_count = len([f for f in os.listdir(val_img_dir) if f.endswith('.png')])
    print(f"   验证图像数量: {val_img_count}")
    
    if not os.path.exists(val_mask_dir):
        print(f"❌ 验证掩码目录不存在: {val_mask_dir}")
        return False
    print(f"✅ 检查验证标签目录: {val_mask_dir}")
    val_mask_count = len([f for f in os.listdir(val_mask_dir) if f.endswith('.png')])
    print(f"   验证掩码数量: {val_mask_count}")
    
    # 检查测试数据
    if not os.path.exists(test_img_dir):
        print(f"❌ 测试图像目录不存在: {test_img_dir}")
        return False
    print(f"✅ 检查测试数据目录: {test_img_dir}")
    test_img_count = len([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    print(f"   测试图像数量: {test_img_count}")
    
    if not os.path.exists(test_mask_dir):
        print(f"❌ 测试掩码目录不存在: {test_mask_dir}")
        return False
    print(f"✅ 检查测试标签目录: {test_mask_dir}")
    test_mask_count = len([f for f in os.listdir(test_mask_dir) if f.endswith('.png')])
    print(f"   测试掩码数量: {test_mask_count}")
    
    print("\n✅ 数据检查完成！")
    print(f"总计: {train_img_count + val_img_count + test_img_count} 张图像")
    
    return True


def main():
    """
    主函数
    """
    # 切换到项目根目录，确保相对路径正确
    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    # 解析命令行参数
    args = parse_args()
    
    print("\n" + "=" * 50)
    print("Electronic Component数据集训练")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"工作目录: {args.work_dir}")
    print(f"数据根目录: {args.data_root}")
    print(f"运行目录: {PROJECT_ROOT}")
    print("=" * 50 + "\n")
    
    # 检查数据可用性
    if not check_data_availability(args.data_root):
        print("\n❌ 数据检查失败，请检查数据集路径和文件！")
        os.chdir(original_dir)
        sys.exit(1)
    
    # 设置配置
    cfg = setup_config(args)
    
    # 初始化默认作用域
    init_default_scope('mmseg')
    
    # 创建训练运行器
    print("\n" + "=" * 50)
    print("创建训练运行器")
    print("=" * 50)
    
    try:
        runner = Runner.from_cfg(cfg)
    except Exception as e:
        print(f"\n❌ 创建运行器时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 开始训练
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)
    
    try:
        runner.train()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_dir)
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ 训练完成！")
    print("=" * 50)
    print(f"模型保存在: {cfg.work_dir}")
    print(f"日志文件: {cfg.work_dir}/*.log")
    print("=" * 50 + "\n")
    
    # 恢复原始目录
    os.chdir(original_dir)


if __name__ == '__main__':
    main()








