# 验证阶段形状不匹配问题修复

## 🐛 问题描述

### 错误信息
```
IndexError: The shape of the mask [512, 512] at index 0 does not match 
the shape of the indexed tensor [384, 544] at index 0
```

### 发生时间
- 训练第500步保存检查点时
- 验证阶段评估时

### 错误堆栈
```python
File "mmseg/evaluation/metrics/iou_metric.py", line 186, in intersect_and_union
    pred_label = pred_label[mask]
IndexError: ...
```

## 🔍 问题分析

### 根本原因

虽然我们在数据加载器中设置了`keep_ratio=False`，但模型的`test_cfg`配置为`mode='whole'`，这导致：

1. **数据加载阶段**：图像被resize到512x512（标签也是512x512）
2. **模型推理阶段**：模型使用`mode='whole'`，会将图像恢复到原始尺寸进行推理
3. **评估阶段**：预测结果是原始尺寸（384x544），但标签是resize后的尺寸（512x512）
4. **结果**：形状不匹配错误

### 配置冲突

```python
# 数据加载器配置 ✅
val_dataloader = dict(
    pipeline=[
        dict(type='Resize', scale=(512, 512), keep_ratio=False),  # 标签: 512x512
        ...
    ]
)

# 模型配置 ❌
model = dict(
    test_cfg=dict(mode='whole')  # 推理时恢复原始尺寸，预测: 384x544
)
```

## ✅ 解决方案

### 修改test_cfg配置

将`mode='whole'`改为`mode='slide'`，并指定crop_size和stride：

```python
model = dict(
    test_cfg=dict(
        mode='slide',           # 使用滑动窗口模式
        crop_size=(512, 512),   # 窗口大小
        stride=(341, 341)       # 滑动步长
    )
)
```

### 为什么这样修复有效？

1. **slide模式**：使用滑动窗口在resize后的图像上进行推理
2. **crop_size=(512, 512)**：确保推理窗口大小与训练时一致
3. **stride=(341, 341)**：窗口重叠，提高预测质量
4. **结果**：预测输出和标签都是512x512，形状匹配

## 📊 修复前后对比

### 修复前

| 阶段 | 图像尺寸 | 标签尺寸 | 预测尺寸 |
|------|----------|----------|----------|
| 数据加载 | 512x512 | 512x512 | - |
| 模型推理 | 恢复原始 | - | 384x544 |
| 评估 | - | 512x512 | 384x544 |
| **结果** | - | - | ❌ 不匹配 |

### 修复后

| 阶段 | 图像尺寸 | 标签尺寸 | 预测尺寸 |
|------|----------|----------|----------|
| 数据加载 | 512x512 | 512x512 | - |
| 模型推理 | 512x512 | - | 512x512 |
| 评估 | - | 512x512 | 512x512 |
| **结果** | - | - | ✅ 匹配 |

## 🔧 完整修改

### 文件：`configs/deepcrack/pspnet_r50-deepcrack_512x512_40k.py`

```python
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(512, 512),  # 固定输入尺寸
        ...
    ),
    ...
    # 修改前
    # test_cfg=dict(mode='whole')
    
    # 修改后
    test_cfg=dict(
        mode='slide',           # 滑动窗口模式
        crop_size=(512, 512),   # 窗口大小
        stride=(341, 341)       # 滑动步长
    )
)
```

## 📝 测试模式说明

### mode='whole'
- **特点**：对整张图像进行一次推理
- **优点**：速度快
- **缺点**：
  - 会将图像恢复到原始尺寸
  - 导致与resize后的标签不匹配
  - 不适合固定尺寸训练的模型

### mode='slide'
- **特点**：使用滑动窗口进行推理
- **优点**：
  - 保持resize后的尺寸
  - 窗口重叠提高预测质量
  - 适合固定尺寸训练的模型
- **缺点**：速度稍慢（但更准确）

## ⚠️ 注意事项

### 1. crop_size选择
- 应该与训练时的输入尺寸一致
- 本项目：`crop_size=(512, 512)`

### 2. stride选择
- stride越小，窗口重叠越多，预测越准确，但速度越慢
- 推荐：`stride = crop_size * 2/3`
- 本项目：`stride=(341, 341)` ≈ 512 * 2/3

### 3. 内存占用
- slide模式比whole模式占用更多内存
- 如果内存不足，可以增大stride

## 🎯 验证修复

### 预期结果

训练应该能够顺利通过第500步的验证，不再出现形状不匹配错误。

### 监控日志

```bash
# 查看训练日志
tail -f training_final.log

# 搜索验证结果
grep "Iter(val)" training_final.log

# 搜索mIoU
grep "mIoU" training_final.log
```

### 成功标志

```
10/25 15:XX:XX - mmengine - INFO - Iter(train) [ 500/5000] ...
10/25 15:XX:XX - mmengine - INFO - Saving checkpoint at 500 iterations
10/25 15:XX:XX - mmengine - INFO - Iter(val) [  1/237] ...
10/25 15:XX:XX - mmengine - INFO - Iter(val) [237/237] ...
10/25 15:XX:XX - mmengine - INFO - mIoU: 0.XXXX
```

## 📚 相关文档

- [MMSegmentation测试模式](https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#test)
- [滑动窗口推理](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/models.html#test-time-augmentation)

## 🔄 历史修复

这是第三次尝试修复此问题：

1. **第一次（版本3.1）**：修改`keep_ratio=True` → `False`
   - 结果：仍然失败，因为test_cfg问题

2. **第二次（版本3.1）**：添加test_pipeline配置
   - 结果：推理成功，但验证仍失败

3. **第三次（当前）**：修改test_cfg模式
   - 结果：应该成功 ✅

## ✅ 总结

**问题**：验证阶段预测和标签形状不匹配  
**原因**：test_cfg使用whole模式恢复原始尺寸  
**解决**：改用slide模式保持resize后的尺寸  
**状态**：已修复，等待验证

---

**修复日期**: 2024-10-25 15:15  
**修复版本**: 4.1  
**修复人员**: AI Assistant

