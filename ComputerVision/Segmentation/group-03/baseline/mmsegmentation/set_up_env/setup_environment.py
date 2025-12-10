#!/usr/bin/env python3
"""
MMSegmentation 环境自动配置脚本 (Python版本)

功能：
1. 检查并安装依赖
2. 验证环境配置
3. 生成环境报告

使用方法：
    python setup_environment.py [--check-only]
"""

import sys
import subprocess
import argparse
from pathlib import Path


class EnvironmentSetup:
    """环境配置类"""
    
    def __init__(self, check_only=False):
        self.check_only = check_only
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        
    def print_header(self, text):
        """打印标题"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70 + "\n")
    
    def print_info(self, text):
        """打印信息"""
        print(f"[INFO] {text}")
    
    def print_success(self, text):
        """打印成功信息"""
        print(f"[✓] {text}")
    
    def print_warning(self, text):
        """打印警告"""
        print(f"[WARNING] {text}")
        self.warnings.append(text)
    
    def print_error(self, text):
        """打印错误"""
        print(f"[ERROR] {text}")
        self.errors.append(text)
    
    def check_python_version(self):
        """检查Python版本"""
        self.print_header("检查Python版本")
        
        version_info = sys.version_info
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        self.print_info(f"Python版本: {python_version}")
        
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 7):
            self.print_error("Python版本过低，需要 >= 3.7")
            return False
        elif version_info.major == 3 and version_info.minor >= 11:
            self.print_warning(f"Python {python_version} 可能不完全兼容，推荐使用 3.8-3.10")
        else:
            self.print_success(f"Python版本符合要求: {python_version}")
        
        return True
    
    def check_package(self, package_name, import_name=None):
        """检查包是否已安装"""
        import_name = import_name or package_name
        
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            self.print_success(f"{package_name}: {version}")
            return True, version
        except ImportError:
            self.print_warning(f"{package_name}: 未安装")
            return False, None
    
    def check_dependencies(self):
        """检查所有依赖"""
        self.print_header("检查依赖包")
        
        # 核心依赖
        core_deps = {
            'PyTorch': 'torch',
            'torchvision': 'torchvision',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'scipy': 'scipy',
            'opencv-python': 'cv2',
        }
        
        # OpenMMLab依赖
        mim_deps = {
            'MMCV': 'mmcv',
            'MMEngine': 'mmengine',
            'MMSegmentation': 'mmseg',
        }
        
        print("\n核心依赖:")
        print("-" * 70)
        for name, import_name in core_deps.items():
            self.check_package(name, import_name)
        
        print("\nOpenMMLab依赖:")
        print("-" * 70)
        for name, import_name in mim_deps.items():
            self.check_package(name, import_name)
        
        # 可选依赖
        optional_deps = {
            'prettytable': 'prettytable',
            'ftfy': 'ftfy',
        }
        
        print("\n可选依赖:")
        print("-" * 70)
        for name, import_name in optional_deps.items():
            self.check_package(name, import_name)
    
    def check_cuda(self):
        """检查CUDA配置"""
        self.print_header("检查CUDA配置")
        
        try:
            import torch
            
            print(f"PyTorch版本: {torch.__version__}")
            print(f"CUDA是否可用: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"cuDNN版本: {torch.backends.cudnn.version()}")
                print(f"GPU数量: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                self.print_success("CUDA配置正常")
            else:
                self.print_warning("CUDA不可用，将使用CPU模式")
                
        except ImportError:
            self.print_error("PyTorch未安装，无法检查CUDA")
    
    def check_mmseg_datasets(self):
        """检查自定义数据集"""
        self.print_header("检查自定义数据集")
        
        # 检查DeepCrack数据集
        deepcrack_dataset = self.project_root / "new_projects/deepcrack/mmseg/datasets/deepcrack.py"
        if deepcrack_dataset.exists():
            self.print_success(f"DeepCrack数据集: {deepcrack_dataset}")
        else:
            self.print_warning(f"DeepCrack数据集未找到: {deepcrack_dataset}")
        
        # 检查Electronic Component数据集
        electronic_dataset = self.project_root / "new_projects/electronic_component/mmseg/datasets/electronic_component.py"
        if electronic_dataset.exists():
            self.print_success(f"Electronic Component数据集: {electronic_dataset}")
        else:
            self.print_warning(f"Electronic Component数据集未找到: {electronic_dataset}")
    
    def check_data_directories(self):
        """检查数据目录"""
        self.print_header("检查数据目录")
        
        # DeepCrack数据
        deepcrack_data = self.project_root / "data/DeepCrack"
        if deepcrack_data.exists():
            train_img = deepcrack_data / "train_img"
            train_lab = deepcrack_data / "train_lab"
            test_img = deepcrack_data / "test_img"
            test_lab = deepcrack_data / "test_lab"
            
            if all(d.exists() for d in [train_img, train_lab, test_img, test_lab]):
                train_count = len(list(train_img.glob("*.jpg")))
                test_count = len(list(test_img.glob("*.jpg")))
                self.print_success(f"DeepCrack数据: {train_count} 训练 + {test_count} 测试")
            else:
                self.print_warning("DeepCrack数据目录不完整")
        else:
            self.print_warning(f"DeepCrack数据未找到: {deepcrack_data}")
        
        # Electronic Component数据
        electronic_data = self.project_root / "data/electronic_component"
        if electronic_data.exists():
            train_img = electronic_data / "images/train"
            val_img = electronic_data / "images/val"
            test_img = electronic_data / "images/test"
            
            if all(d.exists() for d in [train_img, val_img, test_img]):
                train_count = len(list(train_img.glob("*.png")))
                val_count = len(list(val_img.glob("*.png")))
                test_count = len(list(test_img.glob("*.png")))
                self.print_success(f"Electronic Component数据: {train_count} 训练 + {val_count} 验证 + {test_count} 测试")
            else:
                self.print_warning("Electronic Component数据目录不完整")
        else:
            self.print_warning(f"Electronic Component数据未找到: {electronic_data}")
    
    def generate_report(self):
        """生成环境报告"""
        self.print_header("环境配置报告")
        
        if self.errors:
            print("\n❌ 发现错误:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\n⚠️  警告:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ 环境配置完美！所有检查通过。")
        elif not self.errors:
            print(f"\n✅ 环境配置基本正常，但有 {len(self.warnings)} 个警告。")
        else:
            print(f"\n❌ 环境配置存在问题: {len(self.errors)} 个错误, {len(self.warnings)} 个警告。")
            return False
        
        return True
    
    def install_dependencies(self):
        """安装依赖"""
        if self.check_only:
            self.print_warning("仅检查模式，跳过安装")
            return
        
        self.print_header("安装依赖")
        
        # 安装基础依赖
        self.print_info("安装基础依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/runtime.txt"])
        
        # 安装MMSegmentation
        self.print_info("安装MMSegmentation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-v", "-e", "."])
        
        self.print_success("依赖安装完成")
    
    def run(self):
        """运行环境配置检查"""
        print("\n" + "=" * 70)
        print("  MMSegmentation 环境配置检查")
        print("=" * 70)
        
        # 执行所有检查
        self.check_python_version()
        self.check_dependencies()
        self.check_cuda()
        self.check_mmseg_datasets()
        self.check_data_directories()
        
        # 生成报告
        success = self.generate_report()
        
        # 如果有错误且不是仅检查模式，提示安装
        if not success and not self.check_only:
            print("\n建议运行完整的环境配置脚本:")
            print("  bash setup_environment.sh")
        
        return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MMSegmentation 环境配置检查')
    parser.add_argument('--check-only', action='store_true', 
                       help='仅检查环境，不安装依赖')
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(check_only=args.check_only)
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

