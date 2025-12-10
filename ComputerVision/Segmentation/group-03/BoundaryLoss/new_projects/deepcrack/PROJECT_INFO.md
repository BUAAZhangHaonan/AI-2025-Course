# DeepCrack 项目

## 项目描述
DeepCrack裂缝检测数据集的模型训练和测试项目。

## 目录结构
```
deepcrack/
├── configs/                    # 配置文件
│   ├── pspnet_r50-deepcrack_512x512_40k.py
│   └── pspnet_r50-deepcrack_512x512_40k_strong_reg.py    # 稍微缓解过拟合
├── scripts/                    # 训练和测试脚本
│   ├── train_deepcrack.py      # 训练脚本
│   └── test_deepcrack.py       # 测试脚本
├── mmseg/                      # 数据集和转换模块
│   └── datasets/
│       ├── deepcrack.py        # DeepCrack数据集类
│       └── transforms/
│           └── deepcrack_transforms.py  # 自定义transforms
└── docs/                       # 项目文档（一些杂物可以不看）
```

## 数据集位置
- 数据集: `data/DeepCrack/`
- 训练结果: `work_dirs/deepcrack_pspnet_optimized_v2/`

## 使用方法
```bash
# 训练
python new_projects/deepcrack/scripts/train_deepcrack.py

# 测试
python new_projects/deepcrack/scripts/test_deepcrack.py
```

