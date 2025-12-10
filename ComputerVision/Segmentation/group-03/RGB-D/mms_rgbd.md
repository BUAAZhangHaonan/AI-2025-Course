#  RGB_D

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

### RGB-D 融合策略
采用**直接拼接**方法：将深度图作为第4个通道直接拼接到RGB图像后，形成4通道输入 (H, W, 4)



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






# 深度注意力融合V2（稳定版）使用说明


## 📁 新建文件列表

```
rgbd_projects/electronic_component/
├── mmseg/models/
│   ├── fusion/
│   │   └── depth_attention_v2.py              ✅ 新建 - 改进版注意力模块
│   │       ├── DepthGuidedAttentionV2         (完整版，带LayerNorm)
│   │       └── DepthGuidedAttentionV2Light    (轻量版)
│   └── backbones/
│       └── resnet_depth_attention_v2.py       ✅ 新建 - 改进版backbone
└── configs/electronic_component/
    └── pspnet_r50-electronic_depth_attention_v2_512x512_10k.py  ✅ 新建 - V2配置
```

---

## 🔬 融合公式对比

### V2版本（改进）

#### 模式1: Residual（推荐）
```python
# 深度特征内容通过attention加权后，以残差方式加到RGB上
out = rgb_feat + alpha * (depth_feat * attention)
```
- ✅ 保留RGB特征的稳定性
- ✅ 深度特征内容参与融合
- ✅ alpha可学习，初始值小（0.1），逐渐增大

#### 模式2: Weighted
```python
# RGB和深度特征按attention权重加权平均
out = rgb_feat * (1 - attention) + depth_feat * attention
```
- ✅ 完全数据驱动的权重分配
- ⚠️  训练初期可能不稳定（两个特征空间差异大）

#### 模式3: Adaptive
```python
# 与residual类似，但使用normalized depth特征
out = rgb_feat + alpha * (depth_feat_norm * attention)
```

---

## 🚀 快速开始

### 步骤1：更新模块注册

**方式A：手动添加**
编辑 `rgbd_projects/electronic_component/mmseg/models/backbones/__init__.py`:
```python
from .resnet_rgbd import ResNetV1c_RGBD
from .resnet_depth_attention import ResNetV1c_DepthAttention
from .resnet_depth_attention_v2 import ResNetV1c_DepthAttentionV2  # 添加这行

__all__ = ['ResNetV1c_RGBD', 'ResNetV1c_DepthAttention', 'ResNetV1c_DepthAttentionV2']
```

**方式B：使用脚本**
```bash
cd /home/dyk/mms/mmsegmentation
# 创建备份
cp rgbd_projects/electronic_component/mmseg/models/backbones/__init__.py \
   rgbd_projects/electronic_component/mmseg/models/backbones/__init__.py.backup
# 手动添加上述内容
```

同样更新fusion模块的 `__init__.py`:
```python
# rgbd_projects/electronic_component/mmseg/models/fusion/__init__.py
from .depth_attention import DepthGuidedAttention
from .depth_attention_v2 import DepthGuidedAttentionV2, DepthGuidedAttentionV2Light

__all__ = ['DepthGuidedAttention', 'DepthGuidedAttentionV2', 'DepthGuidedAttentionV2Light']
```

### 步骤2：训练模型

```bash
# 单GPU
python tools/train.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_depth_attention_v2_512x512_10k.py

# 多GPU（推荐）
bash tools/dist_train.sh \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_depth_attention_v2_512x512_10k.py \
    4
```

### 步骤3：测试模型

```bash
python tools/test.py \
    rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_depth_attention_v2_512x512_10k.py \
    work_dirs/electronic_component_depth_attention_v2_pspnet/best_mIoU_iter_10000.pth \
    --show-dir results/depth_attention_v2_visualization
```

---



# 可视化脚本

# 训练日志可视化工具

## ✅ 已创建文件

```
tools/
├── visualize_log.py              ✅ 主脚本（完整功能）
└── VISUALIZE_LOG_README.md       ✅ 详细使用说明

run_visualize.sh                  ✅ 快速使用脚本（自动安装依赖）
```

---

## 🚀 最简单的使用方法（推荐）

### 一键运行

```bash
./run_visualize.sh
```

这会：
- ✅ 自动检查并安装matplotlib
- ✅ 自动查找`work_dirs/electronic_component_rgbd_pspnet`的最新日志
- ✅ 生成图表到`visualizations`目录

### 指定工作目录

```bash
./run_visualize.sh work_dirs/electronic_component_pspnet visualizations/rgb
```

---

## 📊 生成的图表

运行后会生成5张图片：

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `loss_curve.png` | Loss曲线 | Total Loss、Decode Loss、Aux Loss |
| `lr_curve.png` | 学习率曲线 | 学习率调度变化 |
| `miou_curve.png` | mIoU曲线 | 验证mIoU，标注最佳值 |
| `accuracy_curve.png` | 准确率曲线 | 训练准确率、验证aAcc和mAcc |
| `training_summary.png` | 综合图表 | 2x2四合一图表 |

---

## 💡 其他使用方式

### 方式1：使用主脚本（指定日志文件）

```bash
python3 tools/visualize_log.py \
    --log work_dirs/electronic_component_rgbd_pspnet/20251028_231641/20251028_231641.log \
    --output visualizations
```

### 方式2：使用主脚本（自动查找最新日志）

```bash
python3 tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_rgbd_pspnet \
    --output visualizations
```

---

## 🆚 对比多个模型

### 方法A：批量可视化

```bash
# RGB基线
./run_visualize.sh work_dirs/electronic_component_pspnet visualizations/rgb

# RGBD拼接
./run_visualize.sh work_dirs/electronic_component_rgbd_pspnet visualizations/rgbd

# 深度注意力V2
./run_visualize.sh work_dirs/electronic_component_depth_attention_v2_pspnet visualizations/attention_v2
```

然后对比三个目录的图表。

### 方法B：手动合并对比

查看各模型的 `training_summary.png` 进行对比。

---
