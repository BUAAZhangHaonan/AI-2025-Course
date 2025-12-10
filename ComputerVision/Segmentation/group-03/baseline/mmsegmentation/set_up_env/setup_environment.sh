#!/bin/bash
################################################################################
# MMSegmentation 环境自动配置脚本
# 
# 功能：
# 1. 创建conda虚拟环境
# 2. 安装PyTorch和相关依赖
# 3. 安装MMCV和MMEngine
# 4. 安装MMSegmentation
# 5. 验证安装
#
# 使用方法：
#   bash setup_environment.sh [环境名称]
#   默认环境名称: mmseg
################################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}$1${NC}"
    echo "======================================================================"
    echo ""
}

# 配置参数
ENV_NAME=${1:-mmseg}
PYTHON_VERSION="3.8"
PYTORCH_VERSION="2.0.0"
CUDA_VERSION="11.8"  # 根据您的GPU调整
TORCHVISION_VERSION="0.15.0"
TORCHAUDIO_VERSION="2.0.0"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_header "MMSegmentation 环境自动配置"

print_info "配置参数："
print_info "  - 环境名称: ${ENV_NAME}"
print_info "  - Python版本: ${PYTHON_VERSION}"
print_info "  - PyTorch版本: ${PYTORCH_VERSION}"
print_info "  - CUDA版本: ${CUDA_VERSION}"
print_info "  - 工作目录: ${SCRIPT_DIR}"

# 检查conda是否安装
print_header "步骤 1/6: 检查Conda环境"

if ! command -v conda &> /dev/null; then
    print_error "Conda未安装！请先安装Anaconda或Miniconda。"
    exit 1
fi
print_success "Conda已安装: $(conda --version)"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建？(y/N): " -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
        print_success "环境已删除"
    else
        print_warning "使用现有环境，跳过创建步骤"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ${ENV_NAME}
        print_info "已激活环境: ${ENV_NAME}"
        # 跳转到依赖安装
        skip_env_creation=true
    fi
fi

# 创建conda环境
if [ "$skip_env_creation" != true ]; then
    print_header "步骤 2/6: 创建Conda环境"
    print_info "创建环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    print_success "环境创建成功"
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${ENV_NAME}
    print_success "环境已激活: ${ENV_NAME}"
fi

# 安装PyTorch
print_header "步骤 3/6: 安装PyTorch"

print_info "检查PyTorch是否已安装..."
if python -c "import torch" 2>/dev/null; then
    CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)")
    print_warning "PyTorch已安装: ${CURRENT_TORCH}"
    read -p "是否重新安装PyTorch ${PYTORCH_VERSION}？(y/N): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过PyTorch安装"
        skip_torch=true
    fi
fi

if [ "$skip_torch" != true ]; then
    print_info "安装PyTorch ${PYTORCH_VERSION} (CUDA ${CUDA_VERSION})"
    
    # 根据CUDA版本选择安装命令
    if [ "$CUDA_VERSION" == "11.8" ]; then
        pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu118
    elif [ "$CUDA_VERSION" == "11.7" ]; then
        pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu117
    elif [ "$CUDA_VERSION" == "cpu" ]; then
        pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu
    else
        print_error "不支持的CUDA版本: ${CUDA_VERSION}"
        print_info "请手动安装PyTorch: https://pytorch.org/get-started/locally/"
        exit 1
    fi
    
    print_success "PyTorch安装完成"
fi

# 验证PyTorch安装
print_info "验证PyTorch安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
print_success "PyTorch验证成功"

# 安装MMCV和MMEngine
print_header "步骤 4/6: 安装OpenMMLab依赖"

print_info "安装MMCV和MMEngine..."

# 安装MMEngine
print_info "安装MMEngine..."
pip install -U openmim
mim install mmengine

# 安装MMCV
print_info "安装MMCV (这可能需要几分钟)..."
mim install "mmcv>=2.0.0,<2.2.0"

print_success "OpenMMLab依赖安装完成"

# 验证MMCV和MMEngine
print_info "验证MMCV和MMEngine安装..."
python -c "import mmcv; import mmengine; print(f'MMCV版本: {mmcv.__version__}'); print(f'MMEngine版本: {mmengine.__version__}')"
print_success "MMCV和MMEngine验证成功"

# 安装MMSegmentation
print_header "步骤 5/6: 安装MMSegmentation"

print_info "从当前目录安装MMSegmentation..."
pip install -v -e .

print_success "MMSegmentation安装完成"

# 验证MMSegmentation
print_info "验证MMSegmentation安装..."
python -c "import mmseg; print(f'MMSegmentation版本: {mmseg.__version__}')"
print_success "MMSegmentation验证成功"

# 安装其他依赖
print_header "步骤 6/6: 安装其他依赖"

print_info "安装基础依赖..."
pip install -r requirements/runtime.txt

print_info "安装可选依赖（部分）..."
# 只安装常用的可选依赖，避免安装所有复杂依赖
pip install matplotlib scipy prettytable ftfy regex

print_success "依赖安装完成"

# 最终验证
print_header "环境配置完成 - 验证"

print_info "执行完整验证..."
python -c "
import sys
import torch
import mmcv
import mmengine
import mmseg

print('='*70)
print('环境配置验证报告')
print('='*70)
print(f'Python版本: {sys.version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'MMCV版本: {mmcv.__version__}')
print(f'MMEngine版本: {mmengine.__version__}')
print(f'MMSegmentation版本: {mmseg.__version__}')
print('='*70)
print('✅ 所有组件验证成功！')
print('='*70)
"

print_header "环境配置总结"

echo ""
echo "✅ 环境配置成功完成！"
echo ""
echo "环境信息："
echo "  - 环境名称: ${ENV_NAME}"
echo "  - 激活命令: conda activate ${ENV_NAME}"
echo ""
echo "DeepCrack项目使用："
echo "  cd new_projects/deepcrack"
echo "  bash scripts/start_deepcrack_training.sh"
echo ""
echo "Electronic Component项目使用："
echo "  cd new_projects/electronic_component"
echo "  bash scripts/start_electronic_training.sh"
echo ""
echo "======================================================================"
echo ""

# 生成环境激活提示
cat > activate_env.sh << EOF
#!/bin/bash
# 快速激活MMSegmentation环境
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo "✅ 已激活环境: ${ENV_NAME}"
EOF

chmod +x activate_env.sh
print_info "已创建快速激活脚本: activate_env.sh"
print_info "使用方法: source activate_env.sh"

