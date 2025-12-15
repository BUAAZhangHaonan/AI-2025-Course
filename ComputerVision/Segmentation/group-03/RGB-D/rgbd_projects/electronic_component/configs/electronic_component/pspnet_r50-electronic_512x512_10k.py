# PSPNet配置文件 - Electronic Component数据集
# 使用ResNet50作为骨干网络，输入尺寸512x512，训练10000次迭代

_base_ = [
    '../../../../configs/_base_/models/pspnet_r50-d8.py',  # PSPNet模型配置
    '../base/electronic_component_dataset.py',  # Electronic Component数据集配置（使用本地副本）
    '../../../../configs/_base_/default_runtime.py',  # 默认运行时配置
    '../../../../configs/_base_/schedules/schedule_20k.py'  # 20k步数的训练计划（我们会修改为10k）
]

# 模型配置
model = dict(
    type='EncoderDecoder',  # 编码器-解码器架构
    data_preprocessor=dict(
        type='SegDataPreProcessor',  # 分割数据预处理器
        mean=[123.675, 116.28, 103.53],  # ImageNet均值
        std=[58.395, 57.12, 57.375],  # ImageNet标准差
        size=(512, 512),  # 固定输入尺寸
        bgr_to_rgb=True,  # BGR转RGB
        pad_val=0,  # 填充值
        seg_pad_val=255),  # 分割图填充值
    pretrained='open-mmlab://resnet50_v1c',  # 预训练权重
    backbone=dict(
        type='ResNetV1c',  # ResNet骨干网络
        depth=50,  # ResNet50
        num_stages=4,  # 4个阶段
        out_indices=(0, 1, 2, 3),  # 输出索引
        dilations=(1, 1, 2, 4),  # 膨胀率
        strides=(1, 2, 1, 1),  # 步长
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # 同步批归一化
        norm_eval=False,  # 训练时批归一化
        style='pytorch',  # PyTorch风格
        contract_dilation=True),  # 收缩膨胀
    decode_head=dict(
        type='PSPHead',  # PSPNet头部
        in_channels=2048,  # 输入通道数
        in_index=3,  # 输入索引
        channels=512,  # 通道数
        pool_scales=(1, 2, 3, 6),  # 池化尺度
        dropout_ratio=0.1,  # Dropout比例
        num_classes=2,  # 类别数：背景和电子元件
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # 同步批归一化
        align_corners=False,  # 不对齐角点
        loss_decode=dict(
            type='CrossEntropyLoss',  # 交叉熵损失
            use_sigmoid=False,  # 不使用sigmoid
            loss_weight=1.0)),  # 损失权重
    auxiliary_head=dict(
        type='FCNHead',  # FCN辅助头部
        in_channels=1024,  # 输入通道数
        in_index=2,  # 输入索引
        channels=256,  # 通道数
        num_convs=1,  # 卷积层数
        concat_input=False,  # 不连接输入
        dropout_ratio=0.1,  # Dropout比例
        num_classes=2,  # 类别数
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # 同步批归一化
        align_corners=False,  # 不对齐角点
        loss_decode=dict(
            type='CrossEntropyLoss',  # 交叉熵损失
            use_sigmoid=False,  # 不使用sigmoid
            loss_weight=0.4)),  # 辅助损失权重
    # 模型训练和测试配置
    train_cfg=dict(),  # 训练配置
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))  # 测试模式：滑动窗口推理

# 数据集配置覆盖（确保参数正确）
train_dataloader = dict(
    batch_size=4,  # 批次大小
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ElectronicComponentDataset',
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='mask/train'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),  # 🔥 关键转换
            dict(type='Resize',
                 scale=(512, 512),
                 keep_ratio=False),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion',
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18),
            dict(type='PackSegInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ElectronicComponentDataset',
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='mask/val'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),  # 🔥 关键转换
            dict(type='Resize',
                 scale=(512, 512),
                 keep_ratio=False),
            dict(type='PackSegInputs')
        ]
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ElectronicComponentDataset',
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='mask/test'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),  # 🔥 关键转换
            dict(type='Resize',
                 scale=(512, 512),
                 keep_ratio=False),
            dict(type='PackSegInputs')
        ]
    )
)

# 评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # IoU评估指标
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # IoU评估指标

# 测试管道配置（用于推理）
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 加载图像
    dict(type='Resize', scale=(512, 512), keep_ratio=False),  # 调整大小
    # LoadAnnotations 在推理时不需要
    dict(type='PackSegInputs')  # 打包输入
]

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',  # 基于迭代的训练循环
    max_iters=10000,  # 10000次迭代（886样本，约11.3个样本/iter，约885 epochs）
    val_interval=500)  # 每500次迭代验证一次

# 验证配置
val_cfg = dict(type='ValLoop')  # 验证循环

# 测试配置
test_cfg = dict(type='TestLoop')  # 测试循环

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',  # 优化器包装器
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),  # SGD优化器
    clip_grad=None)  # 梯度裁剪

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',  # 多项式学习率调度器
        eta_min=1e-4,  # 最小学习率
        power=0.9,  # 幂次
        begin=0,  # 开始迭代
        end=10000,  # 结束迭代
        by_epoch=False)  # 按迭代而非按epoch
]

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 迭代计时器
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),  # 日志记录器
    param_scheduler=dict(type='ParamSchedulerHook'),  # 参数调度器钩子
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=500,  # 每500次迭代保存一次
        max_keep_ckpts=5,  # 最多保留5个检查点
        save_best='mIoU',  # 保存最佳mIoU模型
        rule='greater',  # mIoU越大越好
        save_last=True),  # 保存最后一个检查点
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式采样器种子钩子
    visualization=dict(type='SegVisualizationHook'))  # 分割可视化钩子

# 环境配置
env_cfg = dict(
    cudnn_benchmark=True,  # CUDNN基准测试
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 多进程配置
    dist_cfg=dict(backend='nccl'))  # 分布式配置

# 日志配置
log_processor = dict(by_epoch=False)  # 按迭代记录日志

# 可视化器配置
vis_backends = [dict(type='LocalVisBackend')]  # 本地可视化后端
visualizer = dict(
    type='SegLocalVisualizer',  # 分割本地可视化器
    vis_backends=vis_backends,
    name='visualizer')  # 可视化器名称

# 随机性配置
randomness = dict(seed=0)  # 随机种子

# 工作目录
work_dir = 'work_dirs/electronic_component_pspnet'  # 工作目录

