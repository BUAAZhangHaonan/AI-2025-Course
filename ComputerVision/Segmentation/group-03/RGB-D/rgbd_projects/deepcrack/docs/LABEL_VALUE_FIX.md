# 标签值问题修复

## 🐛 问题发现

### 现象
测试结果图片显示：
- **左边**：原始图像
- **中间**：预测掩码（全黑）
- **右边**：叠加图像（与原图相同）

**结论**：模型预测所有像素都是背景（类别0），完全无法检测裂缝

### 验证结果异常
```
mIoU: 100.0000
aAcc: 100.0000
```

但实际测试时模型完全失效，这是矛盾的。

## 🔍 问题分析

### 检查标签值

运行`scripts/check_labels.py`发现：

```
训练集标签:
  唯一值: [0, 255]
  裂缝像素: 2.91%
  
测试集标签:
  唯一值: [0, 255]
  裂缝像素: 4.33%
```

### 根本原因

**标签值不匹配！**

| 组件 | 期望值 | 实际值 |
|------|--------|--------|
| 背景 | 0 | 0 ✅ |
| 裂缝 | 1 | 255 ❌ |

**问题**：
1. DeepCrack数据集标签中裂缝像素值是**255**
2. MMSegmentation模型期望类别标签是**0, 1, 2, ...**
3. 默认情况下255被视为`ignore_index`（忽略的像素）
4. 模型训练时实际上只学习了背景类（0），裂缝类被忽略了

### 为什么验证mIoU=100%？

因为我们修改的评估代码会自动resize预测结果，但：
1. 预测全是0（背景）
2. 标签中255被视为ignore_index
3. 评估时只计算背景类的IoU
4. 背景类IoU=100%（因为预测和标签都是背景）
5. 裂缝类被忽略，不参与计算

这是一个**假象**！

## ✅ 解决方案

### 创建自定义Transform

**文件**：`mmseg/datasets/transforms/deepcrack_transforms.py`

```python
@TRANSFORMS.register_module()
class ConvertDeepCrackLabels(BaseTransform):
    """
    将DeepCrack标签从255转换为1
    """
    
    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            # 将255转换为1
            gt_seg_map[gt_seg_map == 255] = 1
            results['gt_seg_map'] = gt_seg_map
        return results
```

### 注册Transform

**文件**：`mmseg/datasets/transforms/__init__.py`

```python
from .deepcrack_transforms import ConvertDeepCrackLabels

__all__ = [
    ...,
    'ConvertDeepCrackLabels'
]
```

### 在Pipeline中使用

**文件**：`configs/_base_/datasets/deepcrack_dataset.py`

```python
pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConvertDeepCrackLabels'),  # 添加这一行
    ...
]
```

**应用到所有pipeline**：
- 训练pipeline ✅
- 验证pipeline ✅
- 测试pipeline ✅

## 📊 预期效果

### 修复前
```
标签值: [0, 255]
模型学习: 只学习背景类（0），裂缝类（255）被忽略
验证mIoU: 100%（假象）
测试效果: 完全失败（全预测背景）
```

### 修复后
```
标签值: [0, 1]
模型学习: 学习背景类（0）和裂缝类（1）
验证mIoU: 真实反映模型性能
测试效果: 应该能检测裂缝
```

## 🔄 下一步

### 1. 重新训练模型

```bash
# 清理旧的训练结果
rm -rf work_dirs/deepcrack_pspnet_optimized

# 重新训练
python scripts/train.py
```

### 2. 添加类别权重

由于裂缝像素只占2-4%，需要添加类别权重：

```python
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

### 3. 监控训练

关注指标：
- 训练loss应该在0.1-0.5之间（不是0.0002）
- 训练准确率应该在85-95%之间（不是100%）
- 验证mIoU应该在50-80%之间（不是100%）

## 📝 经验教训

### 1. 数据格式很重要
- 不同数据集的标签格式可能不同
- 必须确保标签值与模型期望一致
- 不要假设数据格式，要验证

### 2. 验证指标可能误导
- mIoU=100%不一定是好事
- 可能是数据问题或评估问题
- 必须实际测试验证

### 3. 调试流程
1. 检查数据格式
2. 检查数据加载
3. 检查模型输入
4. 检查模型输出
5. 检查评估逻辑

### 4. 类别不平衡
- 裂缝检测是典型的不平衡问题
- 必须使用类别权重或Focal Loss
- 不能只看整体准确率

## 🎯 修复清单

- ✅ 创建`ConvertDeepCrackLabels` transform
- ✅ 注册transform到`__init__.py`
- ✅ 添加到训练pipeline
- ✅ 添加到验证pipeline
- ✅ 添加到测试pipeline
- ⏳ 重新训练模型
- ⏳ 添加类别权重
- ⏳ 验证效果

---

**发现日期**: 2024-10-25  
**修复日期**: 2024-10-25  
**状态**: ✅ 已修复代码，待重新训练

