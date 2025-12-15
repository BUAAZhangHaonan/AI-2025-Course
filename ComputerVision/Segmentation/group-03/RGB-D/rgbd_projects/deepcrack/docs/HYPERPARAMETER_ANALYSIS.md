# 超参数分析与调整建议

## 📊 当前参数分析

### 训练损失变化趋势

| 阶段 | 迭代 | Loss | 学习率 | 准确率 | 状态 |
|------|------|------|--------|--------|------|
| 早期 | 100 | 0.504 | 0.00982 | 96.7% | 快速下降 |
| 早期 | 200 | 0.438 | 0.00964 | 95.9% | 继续下降 |
| 中期 | 400 | 0.446 | 0.00929 | 93.4% | 🏆 最佳mIoU |
| 中期 | 1000 | 0.304 | 0.00820 | 92.8% | 稳定 |
| 后期 | 4900 | 0.111 | 0.00039 | 98.3% | ⚠️ 过拟合 |
| 后期 | 5000 | 0.101 | 0.00010 | 98.1% | ⚠️ 严重过拟合 |

### 关键观察

1. **损失下降正常**
   - 早期（0-200轮）：loss从0.5降到0.4 ✅
   - 中期（200-1000轮）：loss从0.4降到0.3 ✅
   - 后期（1000-5000轮）：loss从0.3降到0.1 ⚠️ 太低了！

2. **训练准确率过高**
   - 后期准确率98%+
   - 但测试mIoU只有73%
   - **典型的过拟合！**

3. **学习率衰减**
   - 初始：0.01
   - 400轮：0.00929（约93%）
   - 5000轮：0.0001（1%）
   - **衰减曲线正常** ✅

## 🎯 参数评估

### 1. 学习率 (Learning Rate)

#### 当前设置
```python
optimizer=dict(
    type='SGD', 
    lr=0.01,           # 初始学习率
    momentum=0.9, 
    weight_decay=0.0005
)

param_scheduler=[
    dict(
        type='PolyLR',
        eta_min=1e-4,      # 最小学习率
        power=0.9,
        end=1000
    )
]
```

#### 分析

✅ **初始学习率0.01合适**
- 损失下降稳定
- 没有震荡
- 收敛速度合理

✅ **学习率衰减策略合理**
- PolyLR适合语义分割
- power=0.9是标准设置
- 衰减到0.0001合适

❌ **但是！在400轮后学习率还是太高**
- 400轮时lr=0.00929
- 此时应该更慢地学习以保持泛化

#### 🔧 建议调整

**选项1：提前降低学习率（推荐）**
```python
# 在最佳点附近更激进地衰减
param_scheduler=[
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=1.2,        # ← 从0.9增加到1.2，更快衰减
        end=1000
    )
]
```

**选项2：使用CosineAnnealing（可选）**
```python
param_scheduler=[
    dict(
        type='CosineAnnealingLR',
        T_max=1000,
        eta_min=1e-4
    )
]
```

**选项3：多阶段学习率（高级）**
```python
param_scheduler=[
    dict(
        type='MultiStepLR',
        milestones=[300, 600, 900],  # 在最佳点前后调整
        gamma=0.1
    )
]
```

### 2. 批次大小 (Batch Size)

#### 当前设置
```python
train_dataloader = dict(
    batch_size=8,      # 每批8张图
    ...
)
```

#### 分析

✅ **batch_size=8是合适的**

理由：
1. **显存利用**
   - RTX 4090: 24GB
   - 当前使用：10GB
   - 还有余量但不需要增加

2. **样本数量**
   - 训练集：300张
   - 每个epoch：300/8=37.5步
   - 足够的迭代次数

3. **梯度估计**
   - batch_size=8对于300样本是合理的
   - 太大会导致梯度估计不准

❓ **可以尝试的调整**

**如果想更稳定（不推荐）**
```python
batch_size=16      # 增加到16
# 优点：梯度更稳定
# 缺点：收敛可能变慢，显存增加
```

**如果想更快收敛（不推荐）**
```python
batch_size=4       # 减小到4
# 优点：可能更快找到好的解
# 缺点：训练不稳定，噪声大
```

#### 🔧 建议

**保持batch_size=8** ✅

这是目前最优的选择，因为：
- 显存利用合理
- 梯度估计稳定
- 训练速度快

### 3. 权重衰减 (Weight Decay)

#### 当前设置
```python
optimizer=dict(
    ...
    weight_decay=0.0005
)
```

#### 分析

⚠️ **weight_decay=0.0005可能太小**

证据：
- 训练准确率98%+（过拟合）
- 测试mIoU只有73%
- 说明正则化不足

#### 🔧 建议调整（重要！）

**增加权重衰减以防止过拟合**
```python
optimizer=dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.001     # ← 从0.0005增加到0.001
)
```

理由：
- 增加正则化强度
- 减少过拟合
- 提高泛化能力

### 4. Dropout

#### 当前设置
```python
decode_head=dict(
    ...
    dropout_ratio=0.1,     # 10% dropout
    ...
)
```

#### 分析

⚠️ **dropout=0.1可能不够**

考虑到：
- 严重过拟合
- 小数据集（300样本）
- 需要更强的正则化

#### 🔧 建议调整

**增加Dropout比例**
```python
decode_head=dict(
    type='PSPHead',
    ...
    dropout_ratio=0.3,     # ← 从0.1增加到0.3
    ...
)

auxiliary_head=dict(
    type='FCNHead',
    ...
    dropout_ratio=0.3,     # ← 从0.1增加到0.3
    ...
)
```

### 5. 数据增强

#### 当前设置
```python
pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConvertDeepCrackLabels'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', ...),
    dict(type='PackSegInputs')
]
```

#### 分析

✅ **基本增强已包含**
- RandomFlip ✅
- PhotoMetricDistortion ✅

⚠️ **但可以增加更多**

#### 🔧 建议添加

**增加几何变换**
```python
pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConvertDeepCrackLabels'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    
    # 新增！
    dict(type='RandomRotate', 
         prob=0.5, 
         degree=10),           # 随机旋转±10度
    
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', ...),
    dict(type='PackSegInputs')
]
```

## 📋 推荐的参数调整方案

### 方案A：保守调整（推荐首选）

**只调整最关键的参数**

```python
# 1. 增加权重衰减
optimizer=dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.001        # ← 0.0005 → 0.001
)

# 2. 增加Dropout
decode_head=dict(
    ...
    dropout_ratio=0.3,        # ← 0.1 → 0.3
    ...
)

auxiliary_head=dict(
    ...
    dropout_ratio=0.3,        # ← 0.1 → 0.3
    ...
)

# 3. 保持其他参数不变
```

**预期效果：**
- 减少过拟合
- 提高泛化能力
- mIoU可能略有下降（75-76%），但测试效果更好

### 方案B：激进调整（如果A不够）

**在方案A基础上增加**

```python
# 4. 更强的学习率衰减
param_scheduler=[
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=1.2,            # ← 0.9 → 1.2
        end=1000
    )
]

# 5. 增加数据增强
pipeline=[
    ...
    dict(type='RandomRotate', prob=0.5, degree=10),
    ...
]

# 6. 进一步增加权重衰减
optimizer=dict(
    ...
    weight_decay=0.002,       # ← 0.001 → 0.002
)
```

### 方案C：当前参数不变（如果满意）

**如果您对76.52%的mIoU满意**

```python
# 不需要调整任何参数！
# 只需：
# 1. 减少训练轮次到1000
# 2. 增加验证频率到每10轮
# 3. 使用iter_400的模型
```

## 🎯 我的建议

### 第一次尝试：方案C（当前参数）

**理由：**
1. ✅ 当前参数已经work了
2. ✅ 76.52%是良好的结果
3. ✅ 主要问题是训练太久，不是参数问题

**行动：**
```bash
# 直接用优化后的配置训练
bash scripts/start_training.sh
```

### 如果结果不满意：方案A

**什么情况下使用：**
- 训练mIoU > 95%但测试效果差
- 明显过拟合
- 想要更好的泛化

**行动：**
```bash
# 修改配置文件
# 增加weight_decay和dropout
# 重新训练
```

## 📊 参数对比表

| 参数 | 当前值 | 方案A | 方案B | 影响 |
|------|--------|-------|-------|------|
| `lr` | 0.01 | 0.01 | 0.01 | ✅ 合适 |
| `batch_size` | 8 | 8 | 8 | ✅ 合适 |
| `weight_decay` | 0.0005 | **0.001** | **0.002** | 🔥 需要增加 |
| `dropout_ratio` | 0.1 | **0.3** | **0.3** | 🔥 需要增加 |
| `power` | 0.9 | 0.9 | **1.2** | ⚡ 可选 |
| `max_iters` | 5000 → 1000 | 1000 | 1000 | ✅ 已优化 |
| `val_interval` | 200 → 10 | 10 | 10 | ✅ 已优化 |

## 💡 总结

### 必须调整
1. ✅ **训练轮次**：5000 → 1000（已完成）
2. ✅ **验证频率**：200 → 10（已完成）

### 建议调整（如果过拟合）
3. 🔧 **权重衰减**：0.0005 → 0.001
4. 🔧 **Dropout**：0.1 → 0.3

### 可选调整（进一步优化）
5. ⚡ **学习率衰减**：power 0.9 → 1.2
6. ⚡ **数据增强**：添加RandomRotate

### 保持不变
7. ✅ **学习率**：0.01
8. ✅ **批次大小**：8
9. ✅ **优化器**：SGD + momentum=0.9

---

**我的最终建议：**

**先用当前参数训练一次（方案C），如果过拟合明显再调整！**

因为：
- 上次训练已经达到76.52%（很好的结果）
- 主要问题是训练太久
- 现在只训练1000轮，可能就不会过拟合了

