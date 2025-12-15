# DeepCrack Dataset Configuration
# 这个文件定义了DeepCrack数据集的基础配置

# 数据集类型
dataset_type = 'DeepCrackDataset'

# 训练数据配置
train_dataloader = dict(
    batch_size=4,  # 批次大小，根据GPU内存调整
    num_workers=4,  # 数据加载器工作进程数
    persistent_workers=True,  # 保持工作进程活跃
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 无限采样器，支持随机打乱
    dataset=dict(
        type=dataset_type,
        data_root='data/DeepCrack/',
        data_prefix=dict(
            img_path='train_img',  # 训练图像目录
            seg_map_path='train_lab'  # 训练标签目录
        ),
                pipeline=[
                    # 数据加载管道
                    dict(type='LoadImageFromFile'),  # 从文件加载图像
                    dict(type='LoadAnnotations'),  # 加载标注信息
                    dict(type='ConvertDeepCrackLabels'),  # 将标签255转换为1
                    # 数据增强管道
                    dict(type='RandomResize',  # 随机调整大小
                         scale=(512, 1024),  # 缩放范围
                         ratio_range=(0.5, 2.0),  # 宽高比范围
                         keep_ratio=True),  # 保持宽高比
                    dict(type='RandomCrop',  # 随机裁剪
                         crop_size=(512, 512),  # 裁剪尺寸
                         cat_max_ratio=0.75),  # 最大类别比例
                    dict(type='RandomFlip',  # 随机翻转
                         prob=0.5),  # 翻转概率
                    dict(type='PhotoMetricDistortion'),  # 光度畸变
                    dict(type='Resize',  # 确保所有图像都是相同尺寸
                         scale=(512, 512),  # 固定尺寸
                         keep_ratio=False),  # 不保持宽高比
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
        data_root='data/DeepCrack/',
        data_prefix=dict(
            img_path='test_img',  # 测试图像目录
            seg_map_path='test_lab'  # 测试标签目录
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertDeepCrackLabels'),  # 将标签255转换为1
            dict(type='Resize',  # 调整大小
                 scale=(512, 512),  # 固定尺寸
                 keep_ratio=True),
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
        data_root='data/DeepCrack/',
        data_prefix=dict(
            img_path='test_img',
            seg_map_path='test_lab'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertDeepCrackLabels'),  # 将标签255转换为1
            dict(type='Resize',
                 scale=(512, 512),
                 keep_ratio=True),
            dict(type='PackSegInputs')
        ]
    )
)

# 评估配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # IoU评估指标
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
