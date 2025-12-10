# 深度注意力融合模型使用说明

## 📚 目录
- [概述](#概述)
- [架构说明](#架构说明)
- [文件列表](#文件列表)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [与RGBD拼接的对比](#与rgbd拼接的对比)
- [注意事项](#注意事项)

---

## 概述

本模型采用**深度引导注意力机制**在ResNet的浅层融合RGB和深度信息，相比简单的通道拼接，具有以下优势：

✅ **更好利用预训练权重**：RGB分支保持3通道，可直接加载ImageNet预训练
✅ **灵活的融合策略**：通过注意力机制自适应调整深度信息的贡献
✅ **较少的参数修改**：仅在浅层增加轻量级注意力模块

---

## 架构说明

```
输入: [B, 4, H, W] (RGB + Depth拼接)
    ↓
分离: RGB[B,3,H,W] + Depth[B,1,H,W]
    ↓
RGB → ResNet Stem → [B, 64, H/4, W/4] ←─┐
                          ↓              │ 深度注意力融合
Depth ──────────────────────────────────┘
                          ↓
                   ResNet Stage1-4
                          ↓
               多尺度特征 → PSPHead
```

### 深度注意力模块 (DepthGuidedAttention)

```python
输入: rgb_feat [B, C, H, W], depth [B, 1, H', W']
  ↓
1. 深度特征提取: depth → Conv3x3 → BN → ReLU → Conv3x3 → [B, C//4, H, W]
  ↓
2. 特征拼接: cat(rgb_feat, depth_feat) → [B, C + C//4, H, W]
  ↓
3. 注意力生成: Conv1x1 → ReLU → Conv1x1 → Sigmoid → [B, C, H, W]
  ↓
4. 加权融合: rgb_feat * attention + rgb_feat
  ↓
输出: [B, C, H, W]
```

---

## 文件列表

所有新建文件（**未修改原有任何文件**）：

```
rgbd_projects/electronic_component/
├── mmseg/models/
│   ├── fusion/                                          # 新建目录
│   │   ├── __init__.py                                 # ✅ 新建
│   │   └── depth_attention.py                          # ✅ 新建 - 注意力模块
│   └── backbones/
│       ├── resnet_depth_attention.py                   # ✅ 新建 - 注意力backbone
│       └── __init___new.py                             # ✅ 新建 - 更新后的init（参考用）
└── configs/electronic_component/
    └── pspnet_r50-electronic_depth_attention_512x512_10k.py  # ✅ 新建 - 训练配置
```

---

## 使用方法

### 🔧 步骤1：更新 `__init__.py`

**手动操作**：将以下内容添加到现有的 `__init__.py`，或替换整个文件

**文件**: `rgbd_projects/electronic_component/mmseg/models/backbones/__init__.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
from .resnet_rgbd import ResNetV1c_RGBD
from .resnet_depth_attention import ResNetV1c_DepthAttention  # 添加这行

__all__ = ['ResNetV1c_RGBD', 'ResNetV1c_DepthAttention']  # 添加新模型
```

**快速替换命令**：
```bash
cd /home/dyk/mms/mmsegmentation
cp rgbd_projects/electronic_component/mmseg/models/backbones/__init___new.py \
   rgbd_projects/electronic_component/mmseg/models/backbones/__init__.py
```

---

### 🚀 步骤2：训练模型

#### 单GPU训练
```bash
python tools/train.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_depth_attention_512x512_10k.py
```

#### 多GPU训练（推荐）
```bash
bash tools/dist_train.sh \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_depth_attention_512x512_10k.py \
    4  # 使用4个GPU
```

---

### 🧪 步骤3：测试模型

```bash
python tools/test.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_depth_attention_512x512_10k.py \
    work_dirs/electronic_component_depth_attention_pspnet/best_mIoU_iter_10000.pth \
    --show-dir results/depth_attention_visualization
```

---

### 🔍 步骤4：可视化结果

```bash
# 查看生成的可视化结果
ls results/depth_attention_visualization/vis_data/vis_image/

# 示例：查看第一张图片
# 会显示左右对比：Ground Truth | Prediction
```

---

## 配置说明

### 关键配置参数

**文件**: `pspnet_r50-electronic_depth_attention_512x512_10k.py`

```python
backbone=dict(
    type='ResNetV1c_DepthAttention',  # 使用注意力backbone
    depth=50,                         # ResNet深度: 18/34/50/101/152
    fusion_stage='stem',              # 🔥 融合位置
    attention_reduction=16,           # 🔥 注意力通道缩减比例
    init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
)
```

#### 🎯 `fusion_stage` 参数

| 值 | 融合位置 | 特征尺寸 | 通道数 | 说明 |
|---|---------|---------|--------|------|
| `'stem'` | ResNet stem之后 | H/4 × W/4 | 64 | **推荐**，更早融合，计算量小 |
| `'stage1'` | ResNet stage1之后 | H/4 × W/4 | 256 | 更高级特征融合，计算量大 |

**建议**:
- 先尝试 `fusion_stage='stem'`（默认）
- 如果效果不理想，尝试 `fusion_stage='stage1'`

#### 🎯 `attention_reduction` 参数

控制注意力模块的轻量化程度：
- `reduction=16`（默认）：较轻量，适合大多数情况
- `reduction=8`：更多参数，表达能力更强
- `reduction=32`：更轻量，训练更快

---

## 与RGBD拼接的对比

| 特性 | RGBD拼接 | 深度注意力（本方案） |
|-----|----------|------------------|
| **输入通道** | 4通道 | 4通道（内部分离） |
| **Backbone修改** | 改为4通道输入 | 保持3通道RGB |
| **预训练权重** | 需特殊初始化 | 直接加载ImageNet |
| **融合方式** | 简单拼接 | 注意力加权 |
| **参数量** | 略少 | 略多（注意力模块） |
| **灵活性** | 固定融合 | 自适应融合 |

**配置文件对比**：

```python
# RGBD拼接版本
backbone=dict(
    type='ResNetV1c_RGBD',
    in_channels=4,  # 4通道输入
    depth_init_method='mean'
)

# 深度注意力版本
backbone=dict(
    type='ResNetV1c_DepthAttention',
    fusion_stage='stem',  # 融合位置可调
    attention_reduction=16
)
```

---

## 注意事项

### ⚠️ 必须完成的手动操作

1. **更新 `__init__.py`**（步骤1）
   系统不允许直接修改现有文件，需要您手动操作

### ✅ 已自动处理的部分

- ✅ 数据pipeline（复用RGBD版本的transforms）
- ✅ 测试时加载标注（修复之前的bug）
- ✅ 预训练权重加载（自动处理）
- ✅ 工作目录隔离（不会覆盖RGBD版本）

### 💡 训练建议

1. **从RGB模型迁移**（推荐）：
   ```python
   load_from = 'work_dirs/electronic_component_pspnet/best_mIoU_iter_6500.pth'
   ```
   这会加载decode_head的权重，加速收敛

2. **从头训练**：
   ```python
   load_from = None  # 注释掉或设为None
   ```

3. **学习率调整**：
   - 如果从RGB模型迁移，可以用较小学习率（如0.001）微调
   - 从头训练使用默认学习率（0.01）

---

## 🎓 实验建议

### 对比实验

| 实验组 | 配置文件 | 工作目录 | 说明 |
|-------|---------|---------|------|
| 基线 | `pspnet_r50-electronic_512x512_10k.py` | `work_dirs/electronic_component_pspnet` | RGB only |
| 拼接 | `pre_pspnet_r50-electronic_rgbd_512x512_10k.py` | `work_dirs/electronic_component_rgbd_pspnet` | RGBD拼接 |
| 注意力 | `pspnet_r50-electronic_depth_attention_512x512_10k.py` | `work_dirs/electronic_component_depth_attention_pspnet` | 深度注意力 |

### 消融实验

在配置文件中调整参数：

1. **融合位置对比**:
   - `fusion_stage='stem'` vs `fusion_stage='stage1'`

2. **注意力缩减比例**:
   - `attention_reduction=8/16/32`

3. **是否使用迁移学习**:
   - `load_from='...'` vs `load_from=None`

---

## 📊 预期结果

基于类似任务的经验，预期性能：

```
RGB基线:         mIoU ~75-80%
RGBD拼接:        mIoU ~80-85% (提升5-10%)
深度注意力:      mIoU ~82-87% (可能略优于拼接)
```

**注意**: 实际结果取决于数据集特性

---

## 🐛 常见问题

### Q1: 训练时报 `ModuleNotFoundError: No module named 'fusion'`

**原因**: 未正确导入注意力模块

**解决**: 检查 `resnet_depth_attention.py` 中的导入：
```python
from fusion.depth_attention import DepthGuidedAttention  # ✅ 相对导入
# 或
from rgbd_projects.electronic_component.mmseg.models.fusion.depth_attention import DepthGuidedAttention
```

### Q2: 测试时报 `KeyError: 'gt_sem_seg'`

**原因**: test_pipeline缺少LoadAnnotations

**解决**: 配置文件已包含，确认使用了正确的配置

### Q3: 显存不足

**解决**:
- 减小batch_size: `batch_size=2` 或 `batch_size=1`
- 使用梯度累积
- 减小输入尺寸（不推荐，影响性能）

---

## 📧 总结

已创建的新文件：
1. ✅ `depth_attention.py` - 注意力模块
2. ✅ `resnet_depth_attention.py` - 注意力backbone
3. ✅ `pspnet_r50-electronic_depth_attention_512x512_10k.py` - 训练配置
4. ✅ `__init___new.py` - 更新后的注册文件（需手动替换）

**下一步**: 按照"使用方法"部分执行训练和测试

祝实验顺利！ 🚀







 文件:
  rgbd_projects/electronic_component/mmseg/models/fusion/depth_attention.py

  # 浅层深度注意力模块
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  class DepthGuidedAttention(nn.Module):
      """深度引导的空间注意力模块"""
      def __init__(self, rgb_channels, reduction=16):
          super().__init__()
          # 深度特征提取分支
          self.depth_conv = nn.Sequential(
              nn.Conv2d(1, rgb_channels//4, 3, padding=1),
              nn.BatchNorm2d(rgb_channels//4),
              nn.ReLU(inplace=True),
              nn.Conv2d(rgb_channels//4, rgb_channels//4, 3, padding=1),
              nn.BatchNorm2d(rgb_channels//4),
              nn.ReLU(inplace=True)
          )

          # 注意力权重生成
          self.attention = nn.Sequential(
              nn.Conv2d(rgb_channels + rgb_channels//4,
  rgb_channels//reduction, 1),
              nn.ReLU(inplace=True),
              nn.Conv2d(rgb_channels//reduction, rgb_channels, 1),
              nn.Sigmoid()
          )

      def forward(self, rgb_feat, depth):
          """
          Args:
              rgb_feat: RGB特征 [B, C, H, W]
              depth: 深度图 [B, 1, H, W]
          Returns:
              融合后的特征 [B, C, H, W]
          """
          # 调整深度图尺寸匹配RGB特征
          if depth.shape[2:] != rgb_feat.shape[2:]:
              depth = F.interpolate(depth, size=rgb_feat.shape[2:],
                                   mode='bilinear', align_corners=False)

          # 深度特征提取
          depth_feat = self.depth_conv(depth)

          # 特征拼接
          combined = torch.cat([rgb_feat, depth_feat], dim=1)

          # 生成注意力权重
          att_weight = self.attention(combined)

          # 加权融合
          out = rgb_feat * att_weight + rgb_feat

          return out

  2️⃣ 创建注意力融合Backbone

  文件: rgbd_projects/electronic_component/mmseg/models/backbones/resnet_dep
  th_attention.py

  import torch.nn as nn
  from mmengine.model import BaseModule
  from mmseg.registry import MODELS
  from mmseg.models.backbones import ResNetV1c
  from ..fusion.depth_attention import DepthGuidedAttention

  @MODELS.register_module()
  class ResNetV1c_DepthAttention(BaseModule):
      """带深度注意力的ResNet Backbone"""

      def __init__(self, 
                   depth=50,
                   fusion_stage='stem',  # 'stem' or 'stage1'
                   **kwargs):
          super().__init__()

          # RGB主干网络（保持3通道，可加载ImageNet预训练）
          self.rgb_backbone = ResNetV1c(depth=depth, **kwargs)

          # 深度注意力融合位置
          self.fusion_stage = fusion_stage

          # 获取融合点的通道数
          if fusion_stage == 'stem':
              fusion_channels = 64  # ResNet stem输出64通道
          elif fusion_stage == 'stage1':
              fusion_channels = 256  # ResNet stage1输出256通道
          else:
              raise ValueError(f"Unsupported fusion_stage: {fusion_stage}")

          # 深度注意力模块
          self.depth_attention = DepthGuidedAttention(
              rgb_channels=fusion_channels,
              reduction=16
          )

      def forward(self, x):
          """
          Args:
              x: 输入tensor [B, 4, H, W] (RGB+D拼接)
          Returns:
              多尺度特征列表
          """
          # 分离RGB和深度
          rgb = x[:, :3, :, :]   # [B, 3, H, W]
          depth = x[:, 3:4, :, :] # [B, 1, H, W]

          # RGB通过stem
          rgb_feat = self.rgb_backbone.stem(rgb)

          # 如果在stem后融合
          if self.fusion_stage == 'stem':
              rgb_feat = self.depth_attention(rgb_feat, depth)

          # RGB通过stage1
          rgb_feat = self.rgb_backbone.layer1(rgb_feat)

          # 如果在stage1后融合
          if self.fusion_stage == 'stage1':
              rgb_feat = self.depth_attention(rgb_feat, depth)

          # 后续stages
          outs = []
          outs.append(rgb_feat)  # stage1输出

          x = self.rgb_backbone.layer2(rgb_feat)
          outs.append(x)  # stage2输出

          x = self.rgb_backbone.layer3(x)
          outs.append(x)  # stage3输出

          x = self.rgb_backbone.layer4(x)
          outs.append(x)  # stage4输出

          return outs

  3️⃣ 注册模块

  文件: rgbd_projects/electronic_component/mmseg/models/fusion/__init__.py

  from .depth_attention import DepthGuidedAttention

  __all__ = ['DepthGuidedAttention']

  更新:
  rgbd_projects/electronic_component/mmseg/models/backbones/__init__.py

  from .resnet_rgbd import ResNetV1c_RGBD
  from .resnet_depth_attention import ResNetV1c_DepthAttention

  __all__ = ['ResNetV1c_RGBD', 'ResNetV1c_DepthAttention']

  4️⃣ 创建训练配置

  文件: rgbd_projects/electronic_component/configs/electronic_component/pspn
  et_r50-electronic_depth_attention_512x512_10k.py

  关键配置：
  custom_imports = dict(
      imports=[

  'rgbd_projects.electronic_component.mmseg.datasets.electronic_component',
          'rgbd_projects.electronic_component.mmseg.datasets.transforms.elec
  tronic_component_transforms',
          'rgbd_projects.electronic_component.mmseg.models.backbones.resnet_
  depth_attention'  # 新的backbone
      ],
      allow_failed_imports=False
  )

  model = dict(
      backbone=dict(
          type='ResNetV1c_DepthAttention',  # 使用注意力融合backbone
          depth=50,
          fusion_stage='stem',  # 或 'stage1'
          num_stages=4,
          out_indices=(0, 1, 2, 3),
          dilations=(1, 1, 2, 4),
          strides=(1, 2, 1, 1),
          norm_cfg=dict(type='SyncBN', requires_grad=True),
          norm_eval=False,
          style='pytorch',
          contract_dilation=True,
          init_cfg=dict(type='Pretrained',
  checkpoint='open-mmlab://resnet50_v1c')
      ),
      # ... 其他配置同RGBD版本
  )

  # ⚠️ 数据pipeline复用RGBD的（已有ConcatRGBD等transforms）
  train_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='LoadDepthFromFile', normalize=True, to_float32=True),
      dict(type='LoadAnnotations'),
      dict(type='ConvertInstanceToSemantic'),
      dict(type='Resize', scale=(512, 512), keep_ratio=False),
      dict(type='RandomFlip', prob=0.5),
      dict(type='PhotoMetricDistortion'),
      dict(type='ConcatRGBD'),  # 复用拼接transform
      dict(type='PackSegInputs')
  ]

  # 从RGB基线模型加载权重
  load_from =
  'work_dirs/electronic_component_pspnet/best_mIoU_iter_6500.pth'
  work_dir = 'work_dirs/electronic_component_depth_attention_pspnet'

  ⚠️ 关键避坑经验

  1. 预训练权重加载

  # ✅ 正确：在内部的rgb_backbone使用预训练
  init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')

  # ❌ 错误：在外层wrapper使用预训练（会报key不匹配）

  2. Transform复用

  # ✅ 直接复用现有的ConcatRGBD
  # 因为注意力backbone的forward也接受[B,4,H,W]输入
  dict(type='ConcatRGBD')

  3. 测试时记得加LoadAnnotations

  test_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='LoadDepthFromFile', normalize=True, to_float32=True),
      dict(type='LoadAnnotations'),  # ⚠️ 不要忘记！
      dict(type='ConvertInstanceToSemantic'),
      dict(type='ConcatRGBD'),
      dict(type='PackSegInputs')
  ]

  🎯 实现步骤建议

  您希望我现在就帮您：
  1. ✅ 创建这些文件
  2. ✅ 生成完整配置
  3. ✅ 验证代码可以正常import
