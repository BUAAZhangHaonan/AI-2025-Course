# MMSegmentation 环境配置指南

## 📋 概述

本项目提供了两种环境配置方式：
1. **自动配置脚本（推荐）**: 完全自动化安装所有依赖
2. **手动配置**: 逐步安装各个组件

---

## 🚀 方式一：自动配置（推荐）

### Bash脚本（完整安装）

```bash
# 使用默认环境名称 'mmseg'
bash setup_environment.sh

# 或指定自定义环境名称
bash setup_environment.sh my_mmseg_env
```

#### 功能：
- ✅ 自动创建conda虚拟环境
- ✅ 安装PyTorch (支持CUDA 11.8/11.7/CPU)
- ✅ 安装MMCV和MMEngine
- ✅ 安装MMSegmentation
- ✅ 安装所有必要依赖
- ✅ 验证安装是否成功
- ✅ 生成快速激活脚本

#### 配置参数（可在脚本中修改）：
```bash
ENV_NAME="mmseg"           # 环境名称
PYTHON_VERSION="3.8"       # Python版本
PYTORCH_VERSION="2.0.0"    # PyTorch版本
CUDA_VERSION="11.8"        # CUDA版本 (11.8/11.7/cpu)
```

### Python脚本（仅检查）

```bash
# 检查当前环境配置
python setup_environment.py --check-only

# 检查并安装缺失的依赖
python setup_environment.py
```

#### 检查内容：
- ✅ Python版本
- ✅ PyTorch和CUDA配置
- ✅ MMCV/MMEngine/MMSegmentation
- ✅ 自定义数据集文件
- ✅ 数据目录完整性

---

## 🛠️ 方式二：手动配置

### 步骤1：创建conda环境

```bash
conda create -n mmseg python=3.8 -y
conda activate mmseg
```

### 步骤2：安装PyTorch

#### CUDA 11.8:
```bash
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 11.7:
```bash
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu117
```

#### CPU版本:
```bash
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### 步骤3：安装OpenMMLab组件

```bash
# 安装MIM
pip install -U openmim

# 安装MMEngine
mim install mmengine

# 安装MMCV
mim install "mmcv>=2.0.0,<2.2.0"
```

### 步骤4：安装MMSegmentation

```bash
# 在项目根目录下
pip install -v -e .
```

### 步骤5：安装其他依赖

```bash
pip install -r requirements/runtime.txt
```

---

## ✅ 验证安装

### 快速验证

```bash
python -c "
import torch
import mmcv
import mmengine
import mmseg

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ MMCV:', mmcv.__version__)
print('✅ MMEngine:', mmengine.__version__)
print('✅ MMSegmentation:', mmseg.__version__)
"
```

### 详细检查

```bash
# 使用Python检查脚本
python setup_environment.py --check-only
```

---

## 📦 依赖列表

### 核心依赖

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.7 | 推荐3.8-3.10 |
| PyTorch | >= 1.8 | 推荐2.0+ |
| MMCV | >= 2.0.0, < 2.2.0 | OpenMMLab基础库 |
| MMEngine | >= 0.5.0, < 1.0.0 | 训练引擎 |

### 运行时依赖

```
matplotlib      # 可视化
numpy          # 数值计算
scipy          # 科学计算
prettytable    # 表格输出
opencv-python  # 图像处理
```

### 可选依赖

```
ftfy           # 文本处理
regex          # 正则表达式
cityscapesscripts  # Cityscapes数据集工具
timm           # 预训练模型
```

---

## 🎯 项目特定配置

### DeepCrack项目

```bash
# 激活环境
conda activate mmseg

# 进入项目目录
cd new_projects/deepcrack

# 运行训练
bash scripts/start_deepcrack_training.sh
```

### Electronic Component项目

```bash
# 激活环境
conda activate mmseg

# 进入项目目录
cd new_projects/electronic_component

# 运行训练
bash scripts/start_electronic_training.sh
```

---

## 🔧 常见问题

### 1. CUDA不可用

**问题**: `torch.cuda.is_available()` 返回 `False`

**解决方法**:
- 检查NVIDIA驱动是否安装: `nvidia-smi`
- 确认PyTorch版本与CUDA版本匹配
- 重新安装对应CUDA版本的PyTorch

### 2. MMCV导入错误

**问题**: `ImportError: cannot import name 'xxx' from 'mmcv'`

**解决方法**:
```bash
# 卸载所有MMCV版本
pip uninstall mmcv mmcv-full mmcv-lite -y

# 重新安装正确版本
mim install "mmcv>=2.0.0,<2.2.0"
```

### 3. 内存不足

**问题**: CUDA out of memory

**解决方法**:
- 减小batch_size（在配置文件中修改）
- 减小输入图像尺寸
- 使用混合精度训练（--amp参数）

### 4. 找不到自定义数据集

**问题**: `ModuleNotFoundError: No module named 'deepcrack'`

**解决方法**:
```bash
# 确保在项目根目录
cd /home/szw/segmentation_ws/work_1

# 重新安装
pip install -v -e .
```

---

## 📚 环境管理

### 快速激活环境

使用自动生成的激活脚本:
```bash
source activate_env.sh
```

### 导出环境配置

```bash
# 导出完整环境
conda env export > environment.yml

# 导出pip依赖
pip freeze > requirements_freeze.txt
```

### 删除环境

```bash
conda deactivate
conda env remove -n mmseg
```

---

## 🆘 获取帮助

### 查看脚本帮助

```bash
# Bash脚本
bash setup_environment.sh --help

# Python脚本
python setup_environment.py --help
```

### 相关文档

- [MMSegmentation官方文档](https://mmsegmentation.readthedocs.io/)
- [MMCV文档](https://mmcv.readthedocs.io/)
- [PyTorch官方文档](https://pytorch.org/docs/)

---

## 📝 更新日志

### v1.0.0 (2025-10-27)
- ✅ 创建自动化环境配置脚本
- ✅ 支持Bash和Python两种配置方式
- ✅ 添加完整的环境检查功能
- ✅ 生成快速激活脚本

