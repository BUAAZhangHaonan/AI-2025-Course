#!/usr/bin/env python3
"""
DeepCrack数据集优化训练脚本
针对小数据集（300样本）和简单任务（二分类）优化
"""

import os
import sys
from pathlib import Path

# 获取项目根目录并切换工作目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # new_projects/deepcrack/scripts/ -> work_1/
os.chdir(PROJECT_ROOT)  # 切换到项目根目录，所有路径相对于此

import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepCrack数据集优化训练')
    parser.add_argument('--config', 
                       default='new_projects/deepcrack/configs/pspnet_r50-deepcrack_512x512_40k_strong_reg.py',
                       help='配置文件路径')
    parser.add_argument('--work-dir', 
                        default='work_dirs/deepcrack_pspnet_optimized_v2',
                        help='工作目录（v2：针对早期过拟合优化，1000轮+每10轮验证）')
    parser.add_argument('--data-root', 
                       default='data/DeepCrack',
                       help='数据根目录')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=8,
                       help='批次大小（针对小数据集优化）')
    parser.add_argument('--learning-rate', 
                       type=float, 
                       default=0.01,
                       help='学习率')
    parser.add_argument('--max-iters', 
                       type=int, 
                       default=1000,
                       help='最大迭代次数（针对小数据集优化）')
    parser.add_argument('--val-interval', 
                       type=int, 
                       default=10,
                       help='验证间隔（更频繁验证）')
    parser.add_argument('--resume', 
                       action='store_true',
                       help='从检查点恢复训练')
    parser.add_argument('--load-from', 
                       default=None,
                       help='预训练模型路径')
    
    return parser.parse_args()

def setup_config(args):
    """设置配置文件"""
    print("=" * 60)
    print("DeepCrack优化训练配置（针对小数据集）")
    print("=" * 60)
    
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
    
    # 设置批次大小
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
        print(f"批次大小: {args.batch_size}")
    
    # 设置学习率
    if args.learning_rate:
        cfg.optim_wrapper.optimizer.lr = args.learning_rate
        print(f"学习率: {args.learning_rate}")
    
    # 设置最大迭代次数
    if args.max_iters:
        cfg.train_cfg.max_iters = args.max_iters
        cfg.param_scheduler[0].end = args.max_iters
        print(f"最大迭代次数: {args.max_iters}")
    
    # 设置验证间隔
    if args.val_interval:
        cfg.train_cfg.val_interval = args.val_interval
        print(f"验证间隔: {args.val_interval}")
    
    # 设置检查点保存间隔
    cfg.default_hooks.checkpoint.interval = args.val_interval
    print(f"检查点保存间隔: {args.val_interval}")
    
    # 设置恢复训练
    if args.resume:
        cfg.resume = True
        print("从检查点恢复训练")
    
    # 设置预训练模型
    if args.load_from:
        cfg.load_from = args.load_from
        print(f"预训练模型: {args.load_from}")
    
    return cfg

def check_data_availability(cfg):
    """检查数据可用性"""
    print("\n" + "=" * 60)
    print("检查数据可用性")
    print("=" * 60)
    
    data_root = 'data/DeepCrack/'
    train_img_dir = os.path.join(data_root, 'train_img')
    train_lab_dir = os.path.join(data_root, 'train_lab')
    test_img_dir = os.path.join(data_root, 'test_img')
    test_lab_dir = os.path.join(data_root, 'test_lab')
    
    # 检查训练数据
    if os.path.exists(train_img_dir):
        train_img_count = len([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
        print(f"✓ 训练图像: {train_img_count} 张")
    else:
        raise FileNotFoundError(f"训练图像目录不存在: {train_img_dir}")
    
    if os.path.exists(train_lab_dir):
        train_lab_count = len([f for f in os.listdir(train_lab_dir) if f.endswith('.png')])
        print(f"✓ 训练标签: {train_lab_count} 张")
    else:
        raise FileNotFoundError(f"训练标签目录不存在: {train_lab_dir}")
    
    # 检查测试数据
    if os.path.exists(test_img_dir):
        test_img_count = len([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
        print(f"✓ 测试图像: {test_img_count} 张")
    else:
        raise FileNotFoundError(f"测试图像目录不存在: {test_img_dir}")
    
    if os.path.exists(test_lab_dir):
        test_lab_count = len([f for f in os.listdir(test_lab_dir) if f.endswith('.png')])
        print(f"✓ 测试标签: {test_lab_count} 张")
    else:
        raise FileNotFoundError(f"测试标签目录不存在: {test_lab_dir}")
    
    # 计算训练信息
    batch_size = cfg.train_dataloader.batch_size
    steps_per_epoch = train_img_count // batch_size
    total_epochs = cfg.train_cfg.max_iters // steps_per_epoch
    
    print(f"\n训练信息:")
    print(f"  - 每个epoch步数: {steps_per_epoch}")
    print(f"  - 总epoch数: ~{total_epochs}")
    print(f"  - 验证频率: 每 {cfg.train_cfg.val_interval} 次迭代 (~{cfg.train_cfg.val_interval // steps_per_epoch} epochs)")
    
    print("\n数据检查完成！")
    return True

def main():
    """主函数"""
    # 切换到项目根目录，确保相对路径正确
    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "=" * 60)
    print("DeepCrack数据集优化训练脚本")
    print("针对小数据集（300样本）和简单任务（二分类）")
    print("=" * 60 + "\n")
    print(f"工作目录: {PROJECT_ROOT}\n")
    
    # 解析参数
    args = parse_args()
    
    # 设置配置
    cfg = setup_config(args)
    
    # 检查数据可用性
    check_data_availability(cfg)
    
    # 初始化默认作用域
    init_default_scope('mmseg')
    
    # 创建训练运行器
    print("\n" + "=" * 60)
    print("创建训练运行器")
    print("=" * 60)
    
    try:
        runner = Runner.from_cfg(cfg)
        
        print("\n" + "=" * 60)
        print("开始优化训练...")
        print("=" * 60)
        print(f"工作目录: {cfg.work_dir}")
        print(f"配置文件: {args.config}")
        print(f"批次大小: {cfg.train_dataloader.batch_size}")
        print(f"学习率: {cfg.optim_wrapper.optimizer.lr}")
        print(f"最大迭代次数: {cfg.train_cfg.max_iters}")
        print(f"验证间隔: {cfg.train_cfg.val_interval}")
        print(f"检查点保存间隔: {cfg.default_hooks.checkpoint.interval}")
        print("=" * 60 + "\n")
        
        # 开始训练
        runner.train()
        
        print("\n" + "=" * 60)
        print("✓ 训练完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_dir)
        return 1
    finally:
        # 恢复原始目录
        os.chdir(original_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

