#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练日志可视化脚本
功能：读取MMSegmentation训练日志，可视化loss、mIoU、准确率、学习率等指标
作者：Claude Code
日期：2025-10-29
"""

import re
import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class LogParser:
    """日志解析器"""

    def __init__(self, log_file):
        self.log_file = log_file
        self.train_data = {
            'iter': [],
            'loss': [],
            'lr': [],
            'acc_seg': [],
            'decode_loss': [],
            'aux_loss': []
        }
        self.val_data = {
            'iter': [],
            'mIoU': [],
            'aAcc': [],
            'mAcc': []
        }

    def parse(self):
        """解析日志文件"""
        print(f"📖 正在读取日志文件: {self.log_file}")

        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        checkpoint_iter = 0

        for line in lines:
            # 解析训练迭代数据
            if 'Iter(train)' in line and 'loss:' in line:
                self._parse_train_line(line)

            # 解析验证迭代数据
            if 'Iter(val)' in line and 'mIoU:' in line:
                self._parse_val_line(line, checkpoint_iter)

            # 记录checkpoint迭代数
            if 'Saving checkpoint at' in line:
                match = re.search(r'Saving checkpoint at (\d+) iterations', line)
                if match:
                    checkpoint_iter = int(match.group(1))

        print(f"✅ 解析完成:")
        print(f"   - 训练数据点: {len(self.train_data['iter'])}")
        print(f"   - 验证数据点: {len(self.val_data['iter'])}")

        return self.train_data, self.val_data

    def _parse_train_line(self, line):
        """解析训练行"""
        # 示例: Iter(train) [  50/10000]  lr: 9.9500e-03  loss: 0.5123  decode.loss_ce: 0.3234  decode.acc_seg: 85.12

        # 提取迭代数
        iter_match = re.search(r'Iter\(train\)\s+\[\s*(\d+)/\d+\]', line)
        if not iter_match:
            return
        iter_num = int(iter_match.group(1))

        # 提取学习率
        lr_match = re.search(r'lr:\s+([\d.e+-]+)', line)
        lr = float(lr_match.group(1)) if lr_match else None

        # 提取总loss
        loss_match = re.search(r'(?<!decode\.)(?<!aux\.)loss:\s+([\d.]+)', line)
        loss = float(loss_match.group(1)) if loss_match else None

        # 提取decode准确率
        acc_match = re.search(r'decode\.acc_seg:\s+([\d.]+)', line)
        acc = float(acc_match.group(1)) if acc_match else None

        # 提取decode loss
        decode_loss_match = re.search(r'decode\.loss_ce:\s+([\d.]+)', line)
        decode_loss = float(decode_loss_match.group(1)) if decode_loss_match else None

        # 提取aux loss
        aux_loss_match = re.search(r'aux\.loss_ce:\s+([\d.]+)', line)
        aux_loss = float(aux_loss_match.group(1)) if aux_loss_match else None

        # 记录数据
        if loss is not None:
            self.train_data['iter'].append(iter_num)
            self.train_data['loss'].append(loss)
            self.train_data['lr'].append(lr)
            self.train_data['acc_seg'].append(acc)
            self.train_data['decode_loss'].append(decode_loss)
            self.train_data['aux_loss'].append(aux_loss)

    def _parse_val_line(self, line, checkpoint_iter):
        """解析验证行"""
        # 示例: Iter(val) [110/110]    aAcc: 93.0900  mIoU: 82.1300  mAcc: 87.0900

        if checkpoint_iter == 0:
            return

        # 提取指标
        miou_match = re.search(r'mIoU:\s+([\d.]+)', line)
        aacc_match = re.search(r'aAcc:\s+([\d.]+)', line)
        macc_match = re.search(r'mAcc:\s+([\d.]+)', line)

        if miou_match and aacc_match and macc_match:
            self.val_data['iter'].append(checkpoint_iter)
            self.val_data['mIoU'].append(float(miou_match.group(1)))
            self.val_data['aAcc'].append(float(aacc_match.group(1)))
            self.val_data['mAcc'].append(float(macc_match.group(1)))


class LogVisualizer:
    """日志可视化器"""

    def __init__(self, train_data, val_data, output_dir):
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self):
        """生成所有图表"""
        print("\n📊 正在生成可视化图表...")

        # 1. Loss曲线
        self.plot_loss()

        # 2. 学习率曲线
        self.plot_lr()

        # 3. mIoU曲线
        self.plot_miou()

        # 4. 准确率曲线
        self.plot_accuracy()

        # 5. 综合图表
        self.plot_summary()

        print(f"\n✅ 所有图表已保存到: {self.output_dir}")
        print(f"   - loss_curve.png")
        print(f"   - lr_curve.png")
        print(f"   - miou_curve.png")
        print(f"   - accuracy_curve.png")
        print(f"   - training_summary.png")

    def plot_loss(self):
        """绘制Loss曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        if self.train_data['loss']:
            ax.plot(self.train_data['iter'], self.train_data['loss'],
                   label='Total Loss', linewidth=2, alpha=0.8)

        if self.train_data['decode_loss'] and any(x is not None for x in self.train_data['decode_loss']):
            decode_loss = [x if x is not None else 0 for x in self.train_data['decode_loss']]
            ax.plot(self.train_data['iter'], decode_loss,
                   label='Decode Loss', linewidth=1.5, alpha=0.7)

        if self.train_data['aux_loss'] and any(x is not None for x in self.train_data['aux_loss']):
            aux_loss = [x if x is not None else 0 for x in self.train_data['aux_loss']]
            ax.plot(self.train_data['iter'], aux_loss,
                   label='Auxiliary Loss', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curve.png', dpi=150)
        plt.close()

    def plot_lr(self):
        """绘制学习率曲线"""
        if not self.train_data['lr'] or all(x is None for x in self.train_data['lr']):
            print("⚠️  未找到学习率数据，跳过学习率曲线")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        lr_data = [x if x is not None else 0 for x in self.train_data['lr']]
        ax.plot(self.train_data['iter'], lr_data,
               linewidth=2, color='orange', alpha=0.8)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 使用科学计数法
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'lr_curve.png', dpi=150)
        plt.close()

    def plot_miou(self):
        """绘制mIoU曲线"""
        if not self.val_data['mIoU']:
            print("⚠️  未找到验证数据，跳过mIoU曲线")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.val_data['iter'], self.val_data['mIoU'],
               marker='o', linewidth=2, markersize=6, color='green', alpha=0.8)

        # 标注最大值
        max_miou = max(self.val_data['mIoU'])
        max_iter = self.val_data['iter'][self.val_data['mIoU'].index(max_miou)]
        ax.axhline(y=max_miou, color='red', linestyle='--', alpha=0.5,
                  label=f'Best mIoU: {max_miou:.2f}% @ iter {max_iter}')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('mIoU (%)', fontsize=12)
        ax.set_title('Validation mIoU Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'miou_curve.png', dpi=150)
        plt.close()

    def plot_accuracy(self):
        """绘制准确率曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 训练准确率
        if self.train_data['acc_seg'] and any(x is not None for x in self.train_data['acc_seg']):
            acc_data = [x if x is not None else 0 for x in self.train_data['acc_seg']]
            ax1.plot(self.train_data['iter'], acc_data,
                    linewidth=1.5, color='blue', alpha=0.6)
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.set_title('Training Segmentation Accuracy', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 验证准确率
        if self.val_data['aAcc'] and self.val_data['mAcc']:
            ax2.plot(self.val_data['iter'], self.val_data['aAcc'],
                    marker='o', label='aAcc (Overall)', linewidth=2, markersize=5)
            ax2.plot(self.val_data['iter'], self.val_data['mAcc'],
                    marker='s', label='mAcc (Mean)', linewidth=2, markersize=5)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_curve.png', dpi=150)
        plt.close()

    def plot_summary(self):
        """绘制综合图表（2x2）"""
        fig = plt.figure(figsize=(16, 12))

        # 1. Loss
        ax1 = plt.subplot(2, 2, 1)
        if self.train_data['loss']:
            ax1.plot(self.train_data['iter'], self.train_data['loss'],
                    linewidth=2, color='red', alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Learning Rate
        ax2 = plt.subplot(2, 2, 2)
        if self.train_data['lr'] and any(x is not None for x in self.train_data['lr']):
            lr_data = [x if x is not None else 0 for x in self.train_data['lr']]
            ax2.plot(self.train_data['iter'], lr_data,
                    linewidth=2, color='orange', alpha=0.8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate', fontweight='bold')
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.grid(True, alpha=0.3)

        # 3. mIoU
        ax3 = plt.subplot(2, 2, 3)
        if self.val_data['mIoU']:
            ax3.plot(self.val_data['iter'], self.val_data['mIoU'],
                    marker='o', linewidth=2, markersize=6, color='green', alpha=0.8)
            max_miou = max(self.val_data['mIoU'])
            ax3.axhline(y=max_miou, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('mIoU (%)')
        ax3.set_title('Validation mIoU', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Accuracy
        ax4 = plt.subplot(2, 2, 4)
        if self.val_data['aAcc'] and self.val_data['mAcc']:
            ax4.plot(self.val_data['iter'], self.val_data['aAcc'],
                    marker='o', label='aAcc', linewidth=2, markersize=5)
            ax4.plot(self.val_data['iter'], self.val_data['mAcc'],
                    marker='s', label='mAcc', linewidth=2, markersize=5)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Validation Accuracy', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Training Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_summary.png', dpi=150)
        plt.close()


def find_latest_log(work_dir):
    """查找最新的日志文件"""
    work_path = Path(work_dir)
    if not work_path.exists():
        return None

    log_files = list(work_path.glob('*/*.log'))
    if not log_files:
        return None

    # 按修改时间排序，返回最新的
    latest_log = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return str(latest_log)


def main():
    parser = argparse.ArgumentParser(description='MMSegmentation训练日志可视化工具')
    parser.add_argument('--log', type=str, help='日志文件路径')
    parser.add_argument('--work-dir', type=str, help='工作目录（自动查找最新日志）')
    parser.add_argument('--output', type=str, default='visualizations', help='输出目录')

    args = parser.parse_args()

    # 确定日志文件
    log_file = None
    if args.log:
        log_file = args.log
    elif args.work_dir:
        log_file = find_latest_log(args.work_dir)
        if log_file:
            print(f"🔍 找到最新日志: {log_file}")

    if not log_file:
        print("❌ 错误: 请指定日志文件或工作目录")
        print("\n使用方法:")
        print("  python visualize_log.py --log path/to/xxx.log")
        print("  python visualize_log.py --work-dir work_dirs/xxx")
        sys.exit(1)

    if not os.path.exists(log_file):
        print(f"❌ 错误: 日志文件不存在: {log_file}")
        sys.exit(1)

    # 解析日志
    parser = LogParser(log_file)
    train_data, val_data = parser.parse()

    # 可视化
    visualizer = LogVisualizer(train_data, val_data, args.output)
    visualizer.plot_all()

    print("\n🎉 完成！")


if __name__ == '__main__':
    main()
