# Electronic Component 项目

## 项目描述
Electronic Component电子元件分割数据集的模型训练和测试项目。

## 目录结构
```
electronic_component/
├── configs/                    # 配置文件
│   ├── electronic_component/
│   │   └── pspnet_r50-electronic_512x512_10k.py
│   └── base/
│       └── electronic_component_dataset.py  # 数据集基础配置
├── scripts/                    # 训练和测试脚本
│   ├── train_electronic.py     # 训练脚本
│   └── test_electronic.py      # 测试脚本
└── mmseg/                      # 数据集和转换模块
    └── datasets/
        ├── electronic_component.py      # Electronic Component数据集类
        └── transforms/
            └── electronic_component_transforms.py  # 自定义transforms
```

## 数据集位置
- 数据集: `data/electronic_component/`
- 训练结果: `work_dirs/electronic_component_pspnet/`

## 使用方法
```bash
# 训练
python new_projects/electronic_component/scripts/train_electronic.py

# 测试
python new_projects/electronic_component/scripts/test_electronic.py
```

