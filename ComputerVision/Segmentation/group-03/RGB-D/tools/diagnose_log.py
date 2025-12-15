#!/usr/bin/env python3
"""
日志解析诊断脚本
检查日志文件格式和解析结果
"""
import re
import sys

def diagnose_log(log_file):
    print("=" * 60)
    print("日志文件诊断")
    print("=" * 60)
    print(f"\n📁 文件: {log_file}\n")

    with open(log_file, 'r') as f:
        lines = f.readlines()

    print(f"✅ 总行数: {len(lines)}\n")

    # 1. 检测项目类型
    print("-" * 60)
    print("1. 项目类型检测")
    print("-" * 60)
    for line in lines[:200]:
        if 'resnet_rgbd' in line:
            print("✅ 项目类型: RGBD拼接")
            print(f"   证据: {line.strip()}")
            break
        elif 'resnet_depth_attention_v2' in line:
            print("✅ 项目类型: 深度注意力V2")
            print(f"   证据: {line.strip()}")
            break
        elif 'resnet_depth_attention' in line:
            print("✅ 项目类型: 深度注意力V1")
            print(f"   证据: {line.strip()}")
            break
    else:
        print("⚠️  未检测到明确项目类型，可能是RGB基线")

    # 2. 检测训练数据
    print("\n" + "-" * 60)
    print("2. 训练数据检测")
    print("-" * 60)

    train_count = 0
    train_samples = []
    for line in lines:
        if 'Iter(train)' in line and 'loss:' in line:
            train_count += 1
            if len(train_samples) < 3:
                train_samples.append(line.strip())

    print(f"✅ 找到训练数据: {train_count}条")
    if train_samples:
        print(f"\n示例:")
        for sample in train_samples:
            print(f"   {sample}")

    # 3. 检测验证数据
    print("\n" + "-" * 60)
    print("3. 验证数据检测")
    print("-" * 60)

    val_count = 0
    val_samples = []
    for line in lines:
        if 'Iter(val)' in line and 'mIoU:' in line:
            val_count += 1
            if len(val_samples) < 3:
                val_samples.append(line.strip())

    print(f"✅ 找到验证数据: {val_count}条")
    if val_samples:
        print(f"\n示例:")
        for sample in val_samples:
            print(f"   {sample}")

    # 4. 解析测试
    print("\n" + "-" * 60)
    print("4. 解析测试")
    print("-" * 60)

    # 测试解析训练数据
    if train_samples:
        line = train_samples[0]
        iter_match = re.search(r'Iter\(train\)\s+\[\s*(\d+)/\d+\]', line)
        lr_match = re.search(r'lr:\s+([\d.e+-]+)', line)
        loss_match = re.search(r'(?<!decode\.)(?<!aux\.)loss:\s+([\d.]+)', line)

        if iter_match:
            print(f"   ✅ 迭代数: {iter_match.group(1)}")
        if lr_match:
            print(f"   ✅ 学习率: {lr_match.group(1)}")
        if loss_match:
            print(f"   ✅ Loss: {loss_match.group(1)}")

    # 测试解析验证数据
    if val_samples:
        line = val_samples[0]
        miou_match = re.search(r'mIoU:\s+([\d.]+)', line)
        aacc_match = re.search(r'aAcc:\s+([\d.]+)', line)
        macc_match = re.search(r'mAcc:\s+([\d.]+)', line)

        if miou_match:
            print(f"   ✅ mIoU: {miou_match.group(1)}")
        if aacc_match:
            print(f"   ✅ aAcc: {aacc_match.group(1)}")
        if macc_match:
            print(f"   ✅ mAcc: {macc_match.group(1)}")

    # 5. 检测checkpoint
    print("\n" + "-" * 60)
    print("5. Checkpoint检测")
    print("-" * 60)

    checkpoint_count = 0
    checkpoint_samples = []
    for line in lines:
        if 'Saving checkpoint at' in line:
            checkpoint_count += 1
            if len(checkpoint_samples) < 3:
                checkpoint_samples.append(line.strip())

    print(f"✅ 找到Checkpoint: {checkpoint_count}个")
    if checkpoint_samples:
        print(f"\n示例:")
        for sample in checkpoint_samples:
            print(f"   {sample}")

    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    if train_count > 0 and val_count > 0:
        print("✅ 日志格式正常，可以进行可视化")
    elif train_count > 0:
        print("⚠️  只有训练数据，没有验证数据")
    elif val_count > 0:
        print("⚠️  只有验证数据，没有训练数据")
    else:
        print("❌ 未找到训练或验证数据")

    print(f"\n数据统计:")
    print(f"   训练数据点: {train_count}")
    print(f"   验证数据点: {val_count}")
    print(f"   Checkpoint: {checkpoint_count}")
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 diagnose_log.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    diagnose_log(log_file)
