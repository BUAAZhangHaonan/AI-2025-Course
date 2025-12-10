# DeepCrack裂缝检测项目

基于MMSegmentation框架的裂缝检测项目，支持DeepCrack数据集的训练和测试。

## 📋 项目概述

本项目使用MMSegmentation框架实现了DeepCrack裂缝检测数据集的支持，采用PSPNet模型进行二分类语义分割任务。

### 特点

- ✅ 完整的DeepCrack数据集支持
- ✅ 针对小数据集优化的训练策略
- ✅ 自动保存最佳模型
- ✅ 可视化测试结果
- ✅ 详细的训练和测试文档

### 数据集

- **DeepCrack**: 裂缝检测数据集
  - 训练集: 300张图像
  - 测试集: 237张图像
  - 任务: 二分类（裂缝/背景）

- **Electronic Component**: 电子元件数据集（待实现）
  - 训练集: 886张图像
  - 验证集: 110张图像
  - 测试集: 110张图像

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.13.0
CUDA >= 11.7
```

### 安装依赖

```bash
# 创建虚拟环境
conda create -n mmseg python=3.8 -y
conda activate mmseg

# 安装PyTorch
pip install torch torchvision

# 安装MMSegmentation依赖
pip install mmengine mmcv opencv-python

# 安装MMSegmentation
cd /home/szw/segmentation_ws/work_1
pip install -e .
```

### 数据准备

确保数据集位于正确位置：

```
work_1/
└── data/
    └── DeepCrack/
        ├── train_img/    # 300张训练图像
        ├── train_lab/    # 300张训练标签
        ├── test_img/     # 237张测试图像
        └── test_lab/     # 237张测试标签
```

## 📖 使用方法

### 1. 训练模型

#### 方法1：使用启动脚本（推荐）

```bash
cd /home/szw/segmentation_ws/work_1
bash scripts/start_training.sh
```

#### 方法2：直接运行Python脚本

```bash
cd /home/szw/segmentation_ws/work_1
python scripts/train.py
```

#### 方法3：自定义参数

```bash
python scripts/train.py \
    --batch-size 8 \
    --max-iters 5000 \
    --val-interval 1000 \
    --learning-rate 0.01
```

#### 方法4：后台运行

```bash
nohup python scripts/train.py > training.log 2>&1 &
tail -f training.log
```

### 2. 测试模型

```bash
# 测试指定检查点
python scripts/test.py \
    --checkpoint work_dirs/deepcrack_pspnet_optimized/iter_1000.pth \
    --num-samples 10

# 测试最佳模型
python scripts/test.py \
    --checkpoint work_dirs/deepcrack_pspnet_optimized/best_mIoU_iter_*.pth \
    --num-samples 20
```

### 3. 查看结果

训练结果保存在：
- 检查点: `work_dirs/deepcrack_pspnet_optimized/iter_*.pth`
- 最佳模型: `work_dirs/deepcrack_pspnet_optimized/best_mIoU_iter_*.pth`
- 训练日志: `work_dirs/deepcrack_pspnet_optimized/*.log`

测试结果保存在：
- 可视化图像: `work_dirs/test_results/*_result.jpg`
- 格式: 原图 | 预测掩码 | 叠加图

## 📁 项目结构

```
work_1/
├── data/                          # 数据集目录
│   ├── DeepCrack/                # DeepCrack数据集
│   └── electronic_component_dataset_0831_1k/  # 电子元件数据集
│
├── configs/                       # 配置文件
│   ├── _base_/                   # 基础配置
│   │   └── datasets/
│   │       └── deepcrack_dataset.py  # DeepCrack数据集配置
│   └── deepcrack/                # DeepCrack模型配置
│       └── pspnet_r50-deepcrack_512x512_40k.py
│
├── mmseg/                        # MMSegmentation核心代码
│   └── datasets/
│       └── deepcrack.py         # DeepCrack数据集类
│
├── scripts/                      # 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── test.py                  # 测试脚本
│   └── start_training.sh        # 快速启动脚本
│
├── docs/                         # 文档目录
│   ├── README.md                # 项目总览（本文件）
│   ├── TRAINING_GUIDE.md        # 详细训练指南
│   └── CHANGELOG.md             # 修改记录
│
└── work_dirs/                    # 训练输出目录
    ├── deepcrack_pspnet_optimized/  # 训练结果
    └── test_results/            # 测试结果
```

## 📊 训练配置

### 针对小数据集的优化

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大迭代次数 | 5000 | 约132 epochs |
| 验证间隔 | 1000 | 每26 epochs验证一次 |
| 批次大小 | 8 | 适应小数据集 |
| 学习率 | 0.01 | 初始学习率 |
| 优化器 | SGD | momentum=0.9 |
| 学习率策略 | PolyLR | power=0.9 |

### 数据增强

- RandomFlip (p=0.5)
- PhotoMetricDistortion
- Resize (512x512)

## 📈 性能指标

### 预期结果

- **训练时间**: ~48分钟（RTX 4090）
- **验证mIoU**: > 0.7
- **训练损失**: 0.01-0.1
- **训练准确率**: 90-98%

### 监控训练

```bash
# 查看训练日志
tail -f work_dirs/deepcrack_pspnet_optimized/*.log

# 查看验证结果
grep "mIoU" work_dirs/deepcrack_pspnet_optimized/*.log

# 查看训练进程
ps aux | grep "python.*train.py"
```

## 🔧 故障排查

### 常见问题

#### 1. 形状不匹配错误
```
IndexError: The shape of the mask does not match the tensor
```
**解决方案**: 确保配置文件中所有Resize都使用`keep_ratio=False`

#### 2. 找不到test_pipeline
```
AttributeError: 'ConfigDict' object has no attribute 'test_pipeline'
```
**解决方案**: 确保配置文件中定义了`test_pipeline`

#### 3. 训练过拟合
```
训练loss < 0.001, 准确率 = 100%
```
**解决方案**: 
- 减少训练迭代次数
- 增加数据增强
- 使用更小的模型

### 获取帮助

查看详细文档：
- [训练指南](TRAINING_GUIDE.md)
- [修改记录](CHANGELOG.md)

## 📝 开发计划

### 已完成
- ✅ DeepCrack数据集支持
- ✅ PSPNet模型训练
- ✅ 模型测试和可视化
- ✅ 训练参数优化
- ✅ 文档完善

### 待完成
- ⏳ Electronic Component数据集支持
- ⏳ 多GPU训练支持
- ⏳ 模型部署
- ⏳ 实时推理接口

## 📄 许可证

本项目基于MMSegmentation框架开发，遵循Apache 2.0许可证。

## 🙏 致谢

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [DeepCrack Dataset](https://github.com/yhlleo/DeepCrack)

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件

---

**最后更新**: 2024-10-25
**版本**: 4.0

