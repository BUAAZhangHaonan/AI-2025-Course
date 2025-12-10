# 🛠️ 环境配置工具说明

## 📁 已创建的文件

本次创建了以下环境配置相关文件：

### 1. 自动配置脚本

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `setup_environment.sh` | Bash脚本 | 完整的自动环境配置脚本 |
| `setup_environment.py` | Python脚本 | 环境检查和验证脚本 |
| `activate_env.sh` | Bash脚本 | 快速激活脚本（运行setup后自动生成） |

### 2. 依赖列表

| 文件名 | 说明 |
|--------|------|
| `requirements_full.txt` | 完整的依赖包列表（包含注释说明） |
| `requirements.txt` | 基础依赖引用文件（已存在） |
| `requirements/runtime.txt` | 运行时依赖（已存在） |
| `requirements/optional.txt` | 可选依赖（已存在） |

### 3. 文档

| 文件名 | 说明 |
|--------|------|
| `ENVIRONMENT_SETUP_GUIDE.md` | 详细的环境配置指南 |
| `QUICK_START.md` | 快速开始指南 |
| `SETUP_TOOLS_README.md` | 本文件（工具说明） |

---

## 🚀 使用指南

### 场景1: 首次配置环境（完整安装）

```bash
# 1. 运行自动配置脚本
bash setup_environment.sh

# 脚本将自动完成：
# ✅ 创建conda环境 (mmseg)
# ✅ 安装PyTorch + CUDA
# ✅ 安装MMCV + MMEngine
# ✅ 安装MMSegmentation
# ✅ 验证所有组件
# ✅ 生成快速激活脚本

# 2. 激活环境
conda activate mmseg

# 或使用快速激活脚本
source activate_env.sh
```

**预计时间**: 10-20分钟（取决于网络速度）

---

### 场景2: 仅检查当前环境

```bash
# 检查环境配置状态
python setup_environment.py --check-only

# 输出示例：
# ✅ Python版本正常
# ✅ PyTorch已安装
# ✅ CUDA可用
# ✅ 数据集文件完整
# ⚠️  某些可选依赖未安装
```

**用途**: 
- 快速诊断环境问题
- 验证安装是否成功
- 检查数据集完整性

---

### 场景3: 手动配置（逐步安装）

参考 [ENVIRONMENT_SETUP_GUIDE.md](ENVIRONMENT_SETUP_GUIDE.md) 中的"方式二：手动配置"部分。

---

## 📋 功能特性

### setup_environment.sh（Bash脚本）

✅ **功能全面**
- 自动检测并创建conda环境
- 智能处理已存在的环境
- 支持多种CUDA版本
- 自动验证安装结果
- 生成快速激活脚本

✅ **用户友好**
- 彩色输出，清晰易读
- 进度提示，每步都有反馈
- 错误处理，遇到问题立即停止
- 交互式确认，避免误操作

✅ **灵活配置**
```bash
# 在脚本中可调整的参数：
ENV_NAME="mmseg"           # 环境名称
PYTHON_VERSION="3.8"       # Python版本
PYTORCH_VERSION="2.0.0"    # PyTorch版本
CUDA_VERSION="11.8"        # CUDA版本
```

---

### setup_environment.py（Python脚本）

✅ **检查项目**
- Python版本兼容性
- PyTorch和CUDA配置
- OpenMMLab组件版本
- 自定义数据集文件
- 数据目录完整性

✅ **输出报告**
```
环境配置报告:
- ✅ 成功: X项
- ⚠️  警告: Y项
- ❌ 错误: Z项
```

✅ **使用模式**
```bash
# 仅检查（不安装）
python setup_environment.py --check-only

# 检查并安装缺失依赖
python setup_environment.py
```

---

## 🎯 典型工作流程

### 新用户工作流

```bash
# 第1步: 克隆或下载项目
cd /path/to/project

# 第2步: 运行自动配置
bash setup_environment.sh

# 第3步: 验证环境
python setup_environment.py --check-only

# 第4步: 开始使用
source activate_env.sh
cd new_projects/deepcrack
bash scripts/start_deepcrack_training.sh
```

---

### 调试工作流

```bash
# 遇到问题时：

# 1. 检查环境状态
python setup_environment.py --check-only

# 2. 查看详细错误
python -c "import torch; print(torch.__version__)"
python -c "import mmcv; print(mmcv.__version__)"

# 3. 重新安装问题组件
pip uninstall torch torchvision -y
bash setup_environment.sh  # 重新运行

# 4. 查看文档
cat ENVIRONMENT_SETUP_GUIDE.md
```

---

## 📊 环境检查示例输出

### ✅ 成功示例

```
======================================================================
  环境配置验证报告
======================================================================
Python版本: 3.8.13
PyTorch版本: 2.0.0+cu118
CUDA可用: True
CUDA版本: 11.8
GPU数量: 2
  GPU 0: NVIDIA RTX 4090
  GPU 1: NVIDIA RTX 4090
MMCV版本: 2.0.0
MMEngine版本: 0.8.4
MMSegmentation版本: 1.2.2
======================================================================
✅ 所有组件验证成功！
======================================================================
```

### ⚠️ 警告示例

```
⚠️  警告:
  1. Python 3.13.5 可能不完全兼容，推荐使用 3.8-3.10
  2. prettytable: 未安装（可选依赖）
  3. ftfy: 未安装（可选依赖）

✅ 环境配置基本正常，但有 3 个警告。
```

### ❌ 错误示例

```
❌ 发现错误:
  1. PyTorch未安装，无法检查CUDA
  2. MMCV版本不兼容: 1.x (需要 >= 2.0.0)

❌ 环境配置存在问题: 2 个错误, 5 个警告。

建议运行完整的环境配置脚本:
  bash setup_environment.sh
```

---

## 🔧 自定义配置

### 修改CUDA版本

编辑 `setup_environment.sh`:

```bash
# 将第36行修改为您的CUDA版本
CUDA_VERSION="11.7"  # 或 "11.8", "cpu"
```

### 修改环境名称

```bash
# 方法1: 通过参数指定
bash setup_environment.sh my_custom_env

# 方法2: 修改脚本默认值
ENV_NAME=${1:-my_custom_env}
```

### 添加额外依赖

编辑 `requirements_full.txt` 或创建自己的依赖文件：

```bash
# 安装额外依赖
pip install -r my_requirements.txt
```

---

## 📚 相关文档链接

| 文档 | 内容 |
|------|------|
| [ENVIRONMENT_SETUP_GUIDE.md](ENVIRONMENT_SETUP_GUIDE.md) | 完整的环境配置指南 |
| [QUICK_START.md](QUICK_START.md) | 快速开始指南 |
| [new_projects/README.md](new_projects/README.md) | 项目概览 |
| [new_projects/MIGRATION_SUMMARY.md](new_projects/MIGRATION_SUMMARY.md) | 项目迁移总结 |

---

## ⚡ 常见问题快速解决

### Q: 脚本执行权限不足

```bash
chmod +x setup_environment.sh setup_environment.py
```

### Q: conda命令找不到

```bash
# 安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Q: CUDA版本不匹配

```bash
# 检查CUDA版本
nvidia-smi

# 在setup_environment.sh中修改CUDA_VERSION
# 然后重新运行
```

### Q: 网络连接问题

```bash
# 使用国内镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或在pip install命令中指定
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

---

## 📝 版本信息

- **创建日期**: 2025-10-27
- **适用版本**: MMSegmentation 1.2.2
- **Python要求**: >= 3.7 (推荐 3.8-3.10)
- **PyTorch要求**: >= 1.8 (推荐 2.0+)
- **MMCV要求**: >= 2.0.0, < 2.2.0

---

## 🙏 致谢

这些工具脚本基于：
- MMSegmentation官方安装指南
- OpenMMLab最佳实践
- 社区反馈和经验总结

