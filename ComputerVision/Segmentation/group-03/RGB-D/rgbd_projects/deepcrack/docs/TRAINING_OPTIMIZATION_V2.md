# 训练参数优化 V2 - 防止过拟合

## 📊 训练曲线分析

### 上一次训练（5000轮）的mIoU变化

```
迭代次数   |  mIoU   |  状态
-----------|---------|-------------
  200      | 76.02%  | 上升中
  400      | 76.52%  | 🏆 最佳点
  600      | 74.61%  | 开始下降
  800      | 72.93%  | 继续下降
 1000      | 74.44%  | 波动
 1200      | 74.19%  | 波动
 ...       | ...     | ...
 5000      | 73.40%  | 过拟合
```

### 关键发现

1. **最佳性能在第400轮**
   - mIoU达到76.52%
   - 这是整个训练过程的峰值

2. **400轮后开始过拟合**
   - mIoU从76.52%下降到71-75%
   - 波动范围约5%
   - 模型泛化能力下降

3. **5000轮的模型效果差**
   - 虽然训练集性能很好
   - 但测试集表现不如400轮
   - 典型的过拟合现象

## 🎯 优化策略

### 核心思路

既然最佳点在400轮，我们应该：
1. **减少总训练轮次**：避免训练过度
2. **增加验证频率**：及时发现并保存最佳模型
3. **更频繁保存**：不错过任何好模型

### 具体改进

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `max_iters` | 5000 | 1000 | 最佳点在400轮，1000轮足够 |
| `val_interval` | 200 | 10 | 每10轮验证，不错过最佳点 |
| `checkpoint_interval` | 200 | 10 | 每10轮检查并保存最佳模型 |
| `max_keep_ckpts` | 3 | 5 | 保留更多候选模型 |
| `logger_interval` | 20 | 5 | 更频繁记录日志 |
| `learning_rate_end` | 5000 | 1000 | 与max_iters一致 |

## 🔧 配置文件修改

### 1. 训练配置

```python
# 训练配置（针对小数据集和早期过拟合优化）
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1000,      # ← 5000 → 1000
    val_interval=10)     # ← 200 → 10
```

### 2. 学习率调度

```python
# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=1000,        # ← 5000 → 1000
        by_epoch=False)
]
```

### 3. 检查点保存

```python
# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', 
                interval=5,      # ← 20 → 5
                log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=10,         # ← 200 → 10 🔥
        max_keep_ckpts=5,    # ← 3 → 5
        save_best='mIoU',
        rule='greater',
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
```

## 📈 预期效果

### 训练时间

```
总迭代次数: 1000
每次迭代: ~0.3秒
验证次数: 100次 (每10轮)
预计总时间: ~6分钟 (原~30分钟)
```

### 模型质量

我们期望：
1. **捕获最佳点**：通过每10轮验证，不会错过峰值
2. **避免过拟合**：在1000轮停止，防止性能下降
3. **保留多个候选**：max_keep_ckpts=5，可以对比不同轮次

### 预期最佳mIoU

基于上次训练曲线：
- 预期在300-500轮之间出现峰值
- mIoU应该在75-77%之间
- 理想情况：复现76.52%或更好

## 🚀 如何使用

### 快速开始

```bash
# 清理旧结果并开始新训练
rm -rf work_dirs/deepcrack_pspnet_optimized_v2
bash scripts/start_training.sh
```

### 监控训练

```bash
# 实时查看日志
tail -f work_dirs/deepcrack_pspnet_optimized_v2/*/training_*.log

# 查看验证结果
grep "mIoU" work_dirs/deepcrack_pspnet_optimized_v2/*/*.log

# 查看最佳模型
ls -lh work_dirs/deepcrack_pspnet_optimized_v2/best_mIoU_iter_*.pth
```

## 📊 如何判断成功

### 训练过程中

观察日志中的验证结果：
```
Iter(val) [237/237]  mIoU: 74.50%  ← 迭代200
Iter(val) [237/237]  mIoU: 76.20%  ← 迭代300
Iter(val) [237/237]  mIoU: 76.80%  ← 迭代400 🏆
Iter(val) [237/237]  mIoU: 76.50%  ← 迭代500
Iter(val) [237/237]  mIoU: 75.30%  ← 迭代600
```

### 成功标志

- ✅ mIoU在某个点达到峰值（75-77%）
- ✅ 峰值后略有下降
- ✅ best_mIoU模型被正确保存

### 失败标志

- ❌ mIoU持续上升但未达到75%
- ❌ mIoU剧烈波动（±10%）
- ❌ 训练loss过低（<0.01）

## 🔍 与上一版本对比

### V1 (5000轮, 每200轮验证)

```
优点:
- 训练充分

缺点:
- ❌ 过拟合严重（400轮后性能下降）
- ❌ 验证间隔太大（200轮），可能错过最佳点
- ❌ 训练时间长（~30分钟）
- ❌ 浪费计算资源（后4600轮无意义）

结果:
- best_mIoU_iter_400: 76.52%
- iter_5000: 表现不佳
```

### V2 (1000轮, 每10轮验证) ← 当前

```
优点:
- ✅ 避免过拟合（适时停止）
- ✅ 密集验证（每10轮），不错过最佳点
- ✅ 训练时间短（~6分钟）
- ✅ 资源利用高效

预期结果:
- best_mIoU_iter_XXX: 75-77%
- 应该在300-500轮之间
```

## 💡 进一步优化建议

如果V2的结果还不理想，可以考虑：

### 1. 早停策略

```python
# 添加EarlyStoppingHook
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        patience=50,      # 50轮不提升就停止
        min_delta=0.001)  # 提升小于0.1%视为不提升
]
```

### 2. 更强的正则化

```python
# 增加权重衰减
optimizer=dict(
    type='SGD', 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.001)  # 从0.0005增加到0.001
```

### 3. 数据增强加强

```python
# 添加更多增强
dict(type='RandomRotate', prob=0.5, degree=10),
dict(type='RandomCrop', crop_size=(448, 448)),
dict(type='ColorJitter', brightness=0.5, contrast=0.5),
```

### 4. Dropout

```python
# PSPHead中增加dropout
decode_head=dict(
    ...
    dropout_ratio=0.3,  # 从0.1增加到0.3
    ...
)
```

## 📝 总结

### mIoU是什么？

**mIoU = mean Intersection over Union（平均交并比）**

- 语义分割最重要的评估指标
- 衡量预测和真实标签的重叠程度
- 范围：0-100%，越高越好
- 对于DeepCrack：76.52%属于良好水平

### 为什么400轮最好？

1. **学习充分**：模型已经学到了裂缝特征
2. **未过拟合**：还没有开始记忆训练样本的噪声
3. **泛化能力强**：对测试集表现最好

### 优化核心

**不是训练越久越好，而是要在最佳点及时停止！**

---

**现在开始训练 V2！**

```bash
bash scripts/start_training.sh
```

