# Electronic Component Dataset Configuration
# 这个文件定义了Electronic Component数据集的基础配置

# 自定义模块导入
custom_imports = dict(
    imports=[
        'new_projects.electronic_component.mmseg.datasets.electronic_component',
        'new_projects.electronic_component.mmseg.datasets.transforms.electronic_component_transforms'
    ],
    allow_failed_imports=False
)

# 数据集类型
dataset_type = 'ElectronicComponentDataset'

# 训练数据配置
train_dataloader = dict(
    batch_size=4,  # 批次大小（1024x1024图像较大，使用4）
    num_workers=4,  # 数据加载器工作进程数
    persistent_workers=True,  # 保持工作进程活跃
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 无限采样器，支持随机打乱
    dataset=dict(
        type=dataset_type,
        data_root='data/electronic_component',  # 数据根目录
        data_prefix=dict(
            img_path='images/train',  # 训练图像目录
            seg_map_path='mask/train'  # 训练掩码目录
        ),
        pipeline=[
            # 数据加载管道
            dict(type='LoadImageFromFile'),  # 从文件加载图像
            dict(type='LoadAnnotations'),  # 加载标注信息
            dict(type='ConvertInstanceToSemantic'),  # 🔥 将实例标签转换为语义标签
            # 数据增强管道
            dict(type='Resize',  # 调整大小（从1024→512节省显存）
                 scale=(512, 512),  # 固定尺寸
                 keep_ratio=False),  # 不保持宽高比
            dict(type='RandomFlip',  # 随机翻转
                 prob=0.5),  # 翻转概率
            dict(type='PhotoMetricDistortion',  # 光度畸变
                 brightness_delta=32,  # 亮度变化
                 contrast_range=(0.5, 1.5),  # 对比度范围
                 saturation_range=(0.5, 1.5),  # 饱和度范围
                 hue_delta=18),  # 色调变化
            dict(type='PackSegInputs')  # 打包分割输入
        ]
    )
)

# 验证数据配置
val_dataloader = dict(
    batch_size=1,  # 验证时使用批次大小为1
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # 默认采样器，不打乱
    dataset=dict(
        type=dataset_type,
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/val',  # 验证图像目录
            seg_map_path='mask/val'  # 验证掩码目录
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),  # 🔥 转换标签
            dict(type='Resize',  # 调整大小
                 scale=(512, 512),  # 固定尺寸
                 keep_ratio=False),  # 强制resize
            dict(type='PackSegInputs')
        ]
    )
)

# 测试数据配置
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/test',  # 测试图像目录
            seg_map_path='mask/test'  # 测试掩码目录
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),  # 🔥 转换标签
            dict(type='Resize',
                 scale=(512, 512),
                 keep_ratio=False),
            dict(type='PackSegInputs')
        ]
    )
)

# 评估配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # IoU评估指标
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])








