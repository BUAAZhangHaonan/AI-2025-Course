#!/usr/bin/env python3
"""
训练日志文本报告生成器（无需matplotlib）
"""
import re
import sys

def parse_and_report(log_file, output_file='training_report.txt'):
    print(f"📖 正在读取: {log_file}")

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # 解析数据
    train_data = []
    val_data = []
    checkpoint_iter = 0

    for line in lines:
        # 训练数据
        if 'Iter(train)' in line and 'loss:' in line:
            iter_match = re.search(r'Iter\(train\)\s+\[\s*(\d+)/\d+\]', line)
            lr_match = re.search(r'lr:\s+([\d.e+-]+)', line)
            loss_match = re.search(r'(?<!decode\.)(?<!aux\.)loss:\s+([\d.]+)', line)

            if iter_match and loss_match:
                train_data.append({
                    'iter': int(iter_match.group(1)),
                    'lr': float(lr_match.group(1)) if lr_match else 0,
                    'loss': float(loss_match.group(1))
                })

        # 验证数据
        if 'Iter(val)' in line and 'mIoU:' in line:
            miou_match = re.search(r'mIoU:\s+([\d.]+)', line)
            aacc_match = re.search(r'aAcc:\s+([\d.]+)', line)
            macc_match = re.search(r'mAcc:\s+([\d.]+)', line)

            if miou_match and checkpoint_iter > 0:
                val_data.append({
                    'iter': checkpoint_iter,
                    'mIoU': float(miou_match.group(1)),
                    'aAcc': float(aacc_match.group(1)),
                    'mAcc': float(macc_match.group(1))
                })
                checkpoint_iter = 0

        # Checkpoint
        if 'Saving checkpoint at' in line:
            match = re.search(r'Saving checkpoint at (\d+) iterations', line)
            if match:
                checkpoint_iter = int(match.group(1))

    # 生成报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("训练日志分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"日志文件: {log_file}\n")
        f.write(f"训练数据点: {len(train_data)}\n")
        f.write(f"验证数据点: {len(val_data)}\n\n")

        # 训练损失表
        f.write("-" * 80 + "\n")
        f.write("训练Loss统计 (每50次迭代)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Iter':<10} {'Learning Rate':<15} {'Loss':<10}\n")
        f.write("-" * 80 + "\n")

        for data in train_data[::10]:  # 每隔10个取一个
            f.write(f"{data['iter']:<10} {data['lr']:<15.6e} {data['loss']:<10.4f}\n")

        if train_data:
            f.write("-" * 80 + "\n")
            f.write(f"初始Loss: {train_data[0]['loss']:.4f} @ iter {train_data[0]['iter']}\n")
            f.write(f"最终Loss: {train_data[-1]['loss']:.4f} @ iter {train_data[-1]['iter']}\n")
            min_loss = min(train_data, key=lambda x: x['loss'])
            f.write(f"最小Loss: {min_loss['loss']:.4f} @ iter {min_loss['iter']}\n\n")

        # 验证mIoU表
        f.write("-" * 80 + "\n")
        f.write("验证指标统计 (每500次迭代)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Iter':<10} {'mIoU (%)':<12} {'aAcc (%)':<12} {'mAcc (%)':<12}\n")
        f.write("-" * 80 + "\n")

        for data in val_data:
            f.write(f"{data['iter']:<10} {data['mIoU']:<12.2f} {data['aAcc']:<12.2f} {data['mAcc']:<12.2f}\n")

        if val_data:
            f.write("-" * 80 + "\n")
            best_miou = max(val_data, key=lambda x: x['mIoU'])
            f.write(f"最佳mIoU: {best_miou['mIoU']:.2f}% @ iter {best_miou['iter']}\n")
            f.write(f"初始mIoU: {val_data[0]['mIoU']:.2f}% @ iter {val_data[0]['iter']}\n")
            f.write(f"最终mIoU: {val_data[-1]['mIoU']:.2f}% @ iter {val_data[-1]['iter']}\n")
            f.write(f"提升幅度: +{val_data[-1]['mIoU'] - val_data[0]['mIoU']:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")

    print(f"✅ 报告已生成: {output_file}")

    # 也打印到屏幕
    with open(output_file, 'r') as f:
        print("\n" + f.read())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 text_report.py <log_file> [output_file]")
        sys.exit(1)

    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'training_report.txt'

    parse_and_report(log_file, output_file)
