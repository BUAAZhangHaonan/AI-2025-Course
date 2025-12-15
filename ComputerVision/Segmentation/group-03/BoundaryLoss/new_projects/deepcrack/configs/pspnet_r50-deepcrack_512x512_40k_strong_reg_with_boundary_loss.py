# PSPNet for DeepCrack Dataset
# 使用PSPNet模型在DeepCrack数据集上进行裂缝检测
# 相较于另一个配置文件，增加正则化项，防止过拟合

# 基础配置继承
_base_ = [
    '../../../configs/_base_/models/pspnet_r50-d8.py',  # PSPNet模型配置
    'base/deepcrack_dataset.py',  # DeepCrack数据集配置（使用本地副本）
    '../../../configs/_base_/default_runtime.py',  # 默认运行时配置
    '../../../configs/_base_/schedules/schedule_40k.py'  # 40k步数的训练计划
]

# 模型配置
model = dict(
    type='EncoderDecoder',  # 编码器-解码器架构
    data_preprocessor=dict(
        type='SegDataPreProcessor',  # 分割数据预处理器
        mean=[123.675, 116.28, 103.53],  # ImageNet预训练模型的均值
        std=[58.395, 57.12, 57.375],  # ImageNet预训练模型的标准差
        size=(512, 512),  # 固定输入尺寸
        bgr_to_rgb=True,  # BGR转RGB
        pad_val=0,  # 填充值
        seg_pad_val=255),  # 分割标签填充值
    pretrained='open-mmlab://resnet50_v1c',  # 预训练权重
    backbone=dict(
        type='ResNetV1c',  # ResNet V1c变体
        depth=50,  # 网络深度
        num_stages=4,  # 阶段数
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
        dropout_ratio=0.3,  # 🔥 增加Dropout：0.1 → 0.3（防止过拟合）
        num_classes=2,  # 类别数：背景和裂缝
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # 同步批归一化
        align_corners=False,  # 不对齐角点
        loss_decode=[dict(type='CrossEntropyLoss',  use_sigmoid=False,  loss_weight=1.0, class_weight=[1.0, 10.0]),
                     dict(type='SignedBoundaryLoss', loss_name='loss_boundary_signed', loss_weight=1.0, normalize_phi=True, use_abs=False),
                     ]),  
    auxiliary_head=dict(
        type='FCNHead',  # FCN辅助头部
        in_channels=1024,  # 输入通道数
        in_index=2,  # 输入索引
        channels=256,  # 通道数
        num_convs=1,  # 卷积层数
        concat_input=False,  # 不连接输入
        dropout_ratio=0.3,  # 🔥 增加Dropout：0.1 → 0.3（防止过拟合）
        num_classes=2,  # 类别数
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # 同步批归一化
        align_corners=False,  # 不对齐角点
        loss_decode=dict(
            type='CrossEntropyLoss',  # 交叉熵损失
            use_sigmoid=False,  # 不使用sigmoid
            loss_weight=0.4,
            class_weight=[1.0, 10.0])),  # 类别权重：背景1.0，裂缝10.0
    # 模型训练和测试配置
    train_cfg=dict(),  # 训练配置
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))  # 测试模式：滑动窗口推理

# 数据集配置覆盖
train_dataloader = dict(
    batch_size=2,  # 减小批次大小以适应小数据集（300样本）
    num_workers=4,  # 减少工作进程数
    persistent_workers=True,  # 保持工作进程
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 无限采样器
    dataset=dict(
        type='DeepCrackDataset',  # DeepCrack数据集
        data_root='data/DeepCrack/',  # 数据根目录
        data_prefix=dict(
            img_path='train_img',  # 训练图像路径
            seg_map_path='train_lab'  # 训练标签路径
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),  # 加载图像
            dict(type='LoadAnnotations'),  # 加载标注
            dict(type='ConvertDeepCrackLabels'),  # 🔥 将标签255转换为1
            dict(type='Resize',  # 首先调整到固定尺寸
                 scale=(512, 512),  # 固定尺寸
                 keep_ratio=False),  # 不保持宽高比
            dict(type='RandomFlip',  # 随机翻转
                 prob=0.5),
            dict(type='PhotoMetricDistortion',  # 光度畸变
                 brightness_delta=32,  # 亮度变化
                 contrast_range=(0.5, 1.5),  # 对比度范围
                 saturation_range=(0.5, 1.5),  # 饱和度范围
                 hue_delta=18),  # 色调变化
            dict(type='PackSegInputs')  # 打包输入
        ]
    )
)

val_dataloader = dict(
    batch_size=1,  # 验证批次大小
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # 默认采样器
    dataset=dict(
        type='DeepCrackDataset',  # DeepCrack数据集
        data_root='data/DeepCrack/',  # 数据根目录
        data_prefix=dict(
            img_path='test_img',  # 测试图像路径
            seg_map_path='test_lab'  # 测试标签路径
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),  # 加载图像
            dict(type='LoadAnnotations'),  # 加载标注
            dict(type='ConvertDeepCrackLabels'),  # 🔥 将标签255转换为1
            dict(type='Resize',  # 调整大小（修复形状不匹配问题）
                 scale=(512, 512),
                 keep_ratio=False),  # 强制resize到512x512，不保持宽高比
            dict(type='PackSegInputs')  # 打包输入
        ]
    )
)

test_dataloader = dict(
    batch_size=1,  # 测试批次大小
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # 默认采样器
    dataset=dict(
        type='DeepCrackDataset',  # DeepCrack数据集
        data_root='data/DeepCrack/',  # 数据根目录
        data_prefix=dict(
            img_path='test_img',  # 测试图像路径
            seg_map_path='test_lab'  # 测试标签路径
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),  # 加载图像
            dict(type='LoadAnnotations'),  # 加载标注
            dict(type='ConvertDeepCrackLabels'),  # 🔥 将标签255转换为1
            dict(type='Resize',  # 调整大小（修复形状不匹配问题）
                 scale=(512, 512),
                 keep_ratio=False),  # 强制resize到512x512，不保持宽高比
            dict(type='PackSegInputs')  # 打包输入
        ]
    )
)

# 测试管道配置（用于推理）
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 加载图像
    dict(type='Resize', scale=(512, 512), keep_ratio=False),  # 调整大小
    dict(type='LoadAnnotations'),  # 加载标注（可选）
    dict(type='ConvertDeepCrackLabels'),  # 🔥 将标签255转换为1
    dict(type='PackSegInputs')  # 打包输入
]

# 评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # IoU评估指标
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # IoU评估指标

# 训练配置（针对小数据集和早期过拟合优化）
# 根据训练曲线分析：最佳mIoU在400轮(76.52%)，之后开始过拟合
train_cfg = dict(
    type='IterBasedTrainLoop',  # 基于迭代的训练循环
    max_iters=1000,  # 进一步减少到1000次迭代（约27个epoch），因为400轮已经是最佳
    val_interval=10)  # 每10次迭代验证一次，及时捕捉最佳点（400轮前后）

# 验证配置
val_cfg = dict(type='ValLoop')  # 验证循环

# 测试配置
test_cfg = dict(type='TestLoop')  # 测试循环

# 优化器配置（增强正则化版本）
optim_wrapper = dict(
    type='OptimWrapper',  # 优化器包装器
    optimizer=dict(
        type='SGD', 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=0.001),  # 🔥 增加权重衰减：0.0005 → 0.001（防止过拟合）
    accumulative_counts=4,
    clip_grad=None)  # 梯度裁剪

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',  # 多项式学习率调度器
        eta_min=1e-4,  # 最小学习率
        power=0.9,  # 幂次
        begin=0,  # 开始迭代
        end=1000,  # 结束迭代（调整为1000，与max_iters一致）
        by_epoch=False)  # 按迭代而非按epoch
]

# 默认钩子配置（针对小数据集和早期过拟合优化）
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 迭代计时器
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),  # 更频繁的日志记录（每5次迭代）
    param_scheduler=dict(type='ParamSchedulerHook'),  # 参数调度器钩子
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=10,  # 🔥每10次迭代检查一次，与val_interval一致
        max_keep_ckpts=5,  # 增加到5个，保留更多候选模型
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

# 单GPU训练（注释掉多GPU配置）
# launcher = 'pytorch'  # 使用PyTorch启动器

# 日志配置
log_processor = dict(by_epoch=False)  # 按迭代记录日志

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]  # 本地可视化后端
visualizer = dict(
    type='SegLocalVisualizer',  # 分割本地可视化器
    vis_backends=vis_backends,
    name='visualizer')  # 可视化器名称

# 随机种子
randomness = dict(seed=0)  # 随机种子
