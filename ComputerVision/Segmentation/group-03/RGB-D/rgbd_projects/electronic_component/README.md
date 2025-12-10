# RGB-D 语义分割项目 - Electronic Component

基于 MMSegmentation 的 RGB-D 融合语义分割实现，用于电子元件分割任务。

## 📁 项目结构

```
rgbd_projects/electronic_component/
├── configs/
│   └── electronic_component/
│       └── pspnet_r50-electronic_rgbd_512x512_10k.py  # RGBD配置文件
├── mmseg/
│   ├── datasets/
│   │   ├── electronic_component.py                    # 数据集类（包含RGBD版本）
│   │   └── transforms/
│   │       └── electronic_component_transforms.py     # 自定义transforms
│   └── models/
│       └── backbones/
│           └── resnet_rgbd.py                         # 4通道ResNet
└── scripts/
    └── train_rgbd.py                                  # 训练脚本
```

## 🔧 实现方案

### 1. RGB-D 融合策略
采用**直接拼接**方法：将深度图作为第4个通道直接拼接到RGB图像后，形成4通道输入 (H, W, 4)

### 2. 核心组件

#### 2.1 数据集类 (`ElectronicComponentRGBDDataset`)
- 位置: `mmseg/datasets/electronic_component.py`
- 功能: 自动匹配 RGB 图像、深度图和分割标签
- 路径配置:
  ```python
  data_prefix=dict(
      img_path='images/train',
      seg_map_path='mask/train',
      depth_path='depth/depth_npy/train'
  )
  ```

#### 2.2 数据加载 Transforms
位置: `mmseg/datasets/transforms/electronic_component_transforms.py`

- **LoadDepthFromFile**: 加载 .npy 深度图并归一化
  ```python
  dict(type='LoadDepthFromFile',
       normalize=True,      # 归一化到 [0, 1]
       to_float32=True)
  ```

- **ConcatRGBD**: 拼接 RGB (H,W,3) 和 Depth (H,W,1) 为 RGBD (H,W,4)
  ```python
  dict(type='ConcatRGBD')
  ```

#### 2.3 4通道 Backbone (`ResNetV1c_RGBD`)
- 位置: `mmseg/models/backbones/resnet_rgbd.py`
- 特性:
  - 支持 4 通道输入 (`in_channels=4`)
  - 从预训练 RGB 模型初始化前3个通道
  - 深度通道初始化方法:
    - `mean`: 使用 RGB 三通道权重的平均值（推荐）
    - `zero`: 初始化为零
    - `copy_red`: 复制红色通道权重

### 3. 数据管道

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),           # 加载RGB
    dict(type='LoadDepthFromFile',            # 加载深度
         normalize=True, to_float32=True),
    dict(type='LoadAnnotations'),             # 加载标签
    dict(type='ConvertInstanceToSemantic'),   # 实例→语义
    dict(type='Resize', scale=(512, 512)),    # 调整大小
    dict(type='RandomFlip', prob=0.5),        # 数据增强
    dict(type='PhotoMetricDistortion'),       # 光度变换
    dict(type='ConcatRGBD'),                  # RGB+D拼接
    dict(type='PackSegInputs')                # 打包
]
```

## 🚀 使用方法

### 1. 数据准备
确保数据结构如下:
```
data/electronic_component/
├── images/
│   ├── train/        # RGB 图像 (.png)
│   ├── val/
│   └── test/
├── depth/
│   └── depth_npy/
│       ├── train/    # 深度图 (.npy)
│       ├── val/
│       └── test/
└── mask/
    ├── train/        # 分割标签 (.png)
    ├── val/
    └── test/
```

**注意**: RGB 图像和深度图文件名必须完全匹配（除扩展名外）

### 2. 训练模型

#### 单 GPU 训练
```bash
python rgbd_projects/electronic_component/scripts/train_rgbd.py
```

#### 多 GPU 训练 (推荐)
```bash
# 使用2个GPU
bash tools/dist_train.sh \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py \
    2
```

#### 使用标准训练脚本
```bash
python tools/train.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py \
    --work-dir work_dirs/electronic_component_rgbd_pspnet
```

### 3. 测试模型
```bash
python tools/test.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py \
    work_dirs/electronic_component_rgbd_pspnet/best_mIoU_iter_10000.pth \
    --show-dir results/rgbd_visualization
```

### 4. 可视化预测
```bash
python tools/test.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py \
    work_dirs/electronic_component_rgbd_pspnet/best_mIoU_iter_xxxx.pth \
    --show-dir results/rgbd_vis \
    --opacity 0.5
```

## ⚙️ 配置说明

### 关键配置项
```python
# 模型配置
model = dict(
    backbone=dict(
        type='ResNetV1c_RGBD',
        in_channels=4,                    # RGBD 输入
        depth_init_method='mean',         # 深度通道初始化
    ),
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53, 127.5],  # RGBD均值
        std=[58.395, 57.12, 57.375, 50.0],      # RGBD标准差
    )
)

# 数据集配置
train_dataloader = dict(
    dataset=dict(
        type='ElectronicComponentRGBDDataset',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='mask/train',
            depth_path='depth/depth_npy/train'
        )
    )
)

# 迁移学习：从RGB基线模型初始化
load_from = 'work_dirs/electronic_component_pspnet/best_mIoU_iter_6500.pth'
```

## 📊 模型对比

| 模型 | 输入 | mIoU | 配置文件 |
|------|------|------|----------|
| PSPNet (RGB) | RGB (3通道) | 基线 | `pspnet_r50-electronic_512x512_10k.py` |
| PSPNet (RGBD) | RGBD (4通道) | 待测试 | `pspnet_r50-electronic_rgbd_512x512_10k.py` |

## 🔬 实验设置

- **Backbone**: ResNet-50 + PSPNet
- **输入尺寸**: 512×512
- **训练迭代**: 10,000 次
- **批次大小**: 4
- **学习率**: 0.01 (PolyLR, power=0.9)
- **优化器**: SGD (momentum=0.9, weight_decay=0.0005)
- **数据增强**: RandomFlip, PhotoMetricDistortion
- **验证间隔**: 每 500 次迭代

## 📝 文件路径说明

### 新建文件
1. **配置文件**: `rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py`
2. **RGBD数据集**: `rgbd_projects/electronic_component/mmseg/datasets/electronic_component.py:183-312`
3. **深度加载**: `rgbd_projects/electronic_component/mmseg/datasets/transforms/electronic_component_transforms.py:72-205`
4. **RGBD Backbone**: `rgbd_projects/electronic_component/mmseg/models/backbones/resnet_rgbd.py`
5. **训练脚本**: `rgbd_projects/electronic_component/scripts/train_rgbd.py`

### 修改文件
- `electronic_component.py`: 添加了 `ElectronicComponentRGBDDataset` 类
- `electronic_component_transforms.py`: 添加了 `LoadDepthFromFile` 和 `ConcatRGBD`

## 🐛 常见问题

### 1. 找不到模块错误
确保在配置文件中添加了自定义导入:
```python
custom_imports = dict(
    imports=[
        'rgbd_projects.electronic_component.mmseg.datasets.electronic_component',
        'rgbd_projects.electronic_component.mmseg.datasets.transforms.electronic_component_transforms',
        'rgbd_projects.electronic_component.mmseg.models.backbones.resnet_rgbd'
    ],
    allow_failed_imports=False
)
```

### 2. 深度图文件缺失
检查深度图路径和文件名是否与RGB图像匹配:
```bash
ls data/electronic_component/images/train/ | head -5
ls data/electronic_component/depth/depth_npy/train/ | head -5
```

### 3. 形状不匹配错误
确保RGB图像和深度图尺寸一致，`ConcatRGBD` transform 会检查尺寸匹配。

### 4. 预训练权重加载警告
从RGB模型加载RGBD模型时，第一层卷积的深度通道权重不匹配是正常的，会使用 `depth_init_method` 进行初始化。

## 🎯 下一步改进方向

1. **其他融合方法**:
   - Early fusion: 在 backbone 早期阶段融合
   - Late fusion: 双分支网络，后期特征融合
   - Attention fusion: 使用注意力机制自适应融合

2. **深度预处理**:
   - 深度图归一化策略优化
   - 深度图数据增强

3. **模型架构**:
   - 尝试其他 backbone (Swin Transformer, ConvNeXt)
   - 尝试其他 decoder (DeepLabV3+, SegFormer)

## 📚 参考资料

- [MMSegmentation 文档](https://mmsegmentation.readthedocs.io/)
- [PSPNet 论文](https://arxiv.org/abs/1612.01105)
- RGB-D 分割相关论文

---

**作者**: Claude Code
**日期**: 2025-10-28
**基线模型**: `work_dirs/electronic_component_pspnet/best_mIoU_iter_6500.pth`


  python rgbd_projects/electronic_component/scripts/quick_test_rgbd.py
