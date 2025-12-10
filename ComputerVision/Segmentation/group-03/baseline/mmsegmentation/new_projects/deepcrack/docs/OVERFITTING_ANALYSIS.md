# 模型过拟合问题分析

## 🔍 问题现象

### 测试结果可视化

测试保存的图片包含三部分（从左到右）：
1. **左边**：原始图像
2. **中间**：预测掩码（黑色=背景0，白色=裂缝1）
3. **右边**：叠加图像（红色标记预测的裂缝）

### 异常现象

- ❌ **中间全黑**：模型预测所有像素都是背景（类别0）
- ❌ **左右相同**：因为没有检测到裂缝，叠加图=原图
- ❌ **完全失效**：模型在测试集上完全无法检测裂缝

### 训练vs测试对比

| 指标 | 训练集 | 验证集（测试集） | 实际测试 |
|------|--------|------------------|----------|
| mIoU | 100% | 100% | 0% |
| 准确率 | 100% | 100% | 100%（全预测背景） |
| 裂缝检测 | ✅ 完美 | ✅ 完美（假象） | ❌ 完全失败 |

## 🐛 根本原因

### 1. 验证集mIoU=100%是假象

**为什么验证mIoU=100%但测试失败？**

可能的原因：
1. **验证集标签问题**：测试集的标签可能全是背景
2. **评估逻辑问题**：我们修改的resize逻辑可能有bug
3. **数据加载问题**：验证时可能没有正确加载标签

让我检查一下验证集的标签：

```python
# 检查测试集标签是否全是0
import cv2
import numpy as np

test_lab_dir = 'data/DeepCrack/test_lab'
for f in os.listdir(test_lab_dir)[:5]:
    label = cv2.imread(os.path.join(test_lab_dir, f), 0)
    unique_values = np.unique(label)
    print(f"{f}: {unique_values}")
```

### 2. 严重过拟合

**训练指标：**
- 训练损失：0.0002（极低）
- 训练准确率：100%
- decode.loss_ce：0.0000

**原因：**
1. **数据集太小**：仅300个训练样本
2. **模型太大**：PSPNet-ResNet50（2500万参数）
3. **训练太久**：5000次迭代 ≈ 132 epochs
4. **任务简单**：二分类（裂缝/背景）

### 3. 模型学习策略问题

模型可能学到了一个简单策略：
- **训练集**：记住每张图的裂缝位置
- **测试集**：不认识新图，默认全预测背景

这是典型的**记忆而非学习**。

## 📊 数据集分析

### DeepCrack数据集统计

```
训练集：300张图像
测试集：237张图像
图像尺寸：不固定（384x544, 512x512等）
任务：二分类语义分割
```

### 裂缝像素比例

裂缝通常只占图像的很小一部分（<5%），这导致：
1. **类别不平衡**：背景像素 >> 裂缝像素
2. **简单策略**：全预测背景也能获得95%+准确率
3. **mIoU误导**：如果测试集标签全是背景，预测全背景就是100%

## 🔧 问题验证

### 检查1：测试集标签是否正常

```bash
# 查看测试集标签
cd data/DeepCrack/test_lab
python3 << EOF
import cv2
import numpy as np
import os

for f in sorted(os.listdir('.'))[0:10]:
    if f.endswith('.png'):
        label = cv2.imread(f, 0)
        unique = np.unique(label)
        crack_pixels = np.sum(label > 0)
        total_pixels = label.size
        ratio = crack_pixels / total_pixels * 100
        print(f"{f}: unique={unique}, crack={ratio:.2f}%")
EOF
```

### 检查2：模型实际输出

```python
# 查看模型输出的原始logits
import torch
from mmseg.apis import init_model, inference_model

model = init_model('configs/...', 'work_dirs/.../best_mIoU_iter_500.pth')
result = inference_model(model, 'data/DeepCrack/test_img/xxx.jpg')

# 查看预测分布
pred = result.pred_sem_seg.data
print(f"预测值分布: {torch.unique(pred, return_counts=True)}")
print(f"预测为裂缝的像素数: {torch.sum(pred == 1).item()}")
```

### 检查3：损失函数权重

```python
# 检查是否使用了类别权重
model.decode_head.loss_decode
# 应该看到是否有class_weight参数
```

## ✅ 解决方案

### 方案1：使用类别权重（推荐）

```python
# 在配置文件中添加类别权重
decode_head=dict(
    ...
    loss_decode=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        class_weight=[1.0, 10.0]  # 背景:1, 裂缝:10
    )
)
```

### 方案2：使用Focal Loss

```python
decode_head=dict(
    ...
    loss_decode=dict(
        type='FocalLoss',
        use_sigmoid=False,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0
    )
)
```

### 方案3：减小模型容量

```python
# 使用更小的backbone
backbone=dict(
    type='ResNetV1c',
    depth=18,  # 从50改为18
    ...
)
```

### 方案4：增加数据增强

```python
pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=10),  # 新增
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCrop', crop_size=(448, 448)),  # 新增
    dict(type='PackSegInputs')
]
```

### 方案5：减少训练迭代

```python
# 进一步减少迭代次数
train_cfg = dict(
    max_iters=2000,  # 从5000改为2000
    val_interval=200  # 从500改为200
)
```

## 🎯 推荐行动

### 立即行动

1. **检查测试集标签**
   ```bash
   python scripts/check_labels.py
   ```

2. **使用早期检查点测试**
   ```bash
   # 测试iter_500而不是iter_5000
   python scripts/test.py --checkpoint work_dirs/.../iter_500.pth
   ```

3. **添加类别权重重新训练**
   - 修改配置文件添加`class_weight=[1.0, 10.0]`
   - 重新训练

### 验证步骤

1. 确认测试集标签不是全0
2. 确认模型输出不是全0
3. 如果都正常，说明需要调整训练策略

## 📝 经验教训

### 1. mIoU=100%不一定是好事

- 可能是过拟合
- 可能是数据问题
- 需要实际测试验证

### 2. 类别不平衡很重要

- 裂缝检测是典型的不平衡问题
- 必须使用类别权重或Focal Loss
- 不能只看准确率，要看每个类别的recall

### 3. 小数据集训练困难

- 300样本对于深度学习太少
- 容易过拟合
- 需要强正则化和数据增强

### 4. 验证集要有代表性

- 如果验证集和测试集分布不同
- 验证指标会误导
- 需要确保数据划分合理

## 🔄 下一步

1. ✅ 分析问题原因
2. ⏳ 检查测试集标签
3. ⏳ 添加类别权重
4. ⏳ 重新训练
5. ⏳ 验证效果

---

**分析日期**: 2024-10-25  
**问题状态**: 🔴 严重过拟合，模型失效  
**优先级**: 🔥 高优先级

