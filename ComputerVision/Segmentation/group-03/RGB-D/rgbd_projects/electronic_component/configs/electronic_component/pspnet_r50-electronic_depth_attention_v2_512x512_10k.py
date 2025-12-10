# PSPNet 深度注意力V2配置文件（稳定版）- Electronic Component 数据集
# 核心改进：
# 1. 深度特征内容参与融合
# 2. LayerNorm增强稳定性
# 3. 可学习残差权重alpha（初始0.1）
# 4. 残差融合模式

_base_ = [
    '../../../../configs/_base_/models/pspnet_r50-d8.py',
    '../../../../configs/_base_/default_runtime.py',
    '../../../../configs/_base_/schedules/schedule_20k.py'
]

# 自定义导入
custom_imports = dict(
    imports=[
        'rgbd_projects.electronic_component.mmseg.datasets.electronic_component',
        'rgbd_projects.electronic_component.mmseg.datasets.transforms.electronic_component_transforms',
        'rgbd_projects.electronic_component.mmseg.models.backbones.resnet_depth_attention_v2'  # V2版本
    ],
    allow_failed_imports=False
)

# 模型配置 - 使用改进版深度注意力融合
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53, 127.5],
        std=[58.395, 57.12, 57.375, 50.0],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=1
    ),
    backbone=dict(
        type='ResNetV1c_DepthAttentionV2',  # 改进版backbone
        depth=50,

        # === 核心配置参数 ===
        fusion_stage='stem',           # 融合位置: 'stem' 或 'stage1'
        fusion_mode='residual',        # 融合模式: 'residual' (推荐), 'weighted', 'adaptive'
        attention_reduction=16,        # 注意力通道缩减: 8, 16, 32
        use_layer_norm=True,           # 使用LayerNorm稳定训练 [推荐True]
        init_alpha=0.1,                # 残差权重初始值 [0.05-0.2都可以]
        use_light_version=False,       # 轻量级版本（去除LayerNorm）

        # ResNet标准配置
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,

        # RGB backbone加载ImageNet预训练
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
    ),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512))
)

# RGBD 数据管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile',
         normalize=True,
         to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='ConvertInstanceToSemantic'),
    dict(type='Resize',
         scale=(512, 512),
         keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='ConcatRGBD'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile',
         normalize=True,
         to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='ConvertInstanceToSemantic'),
    dict(type='ConcatRGBD'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile',
         normalize=True,
         to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='ConvertInstanceToSemantic'),
    dict(type='ConcatRGBD'),
    dict(type='PackSegInputs')
]

# 数据集配置
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ElectronicComponentRGBDDataset',
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='mask/train',
            depth_path='depth/depth_npy/train'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ElectronicComponentRGBDDataset',
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='mask/val',
            depth_path='depth/depth_npy/val'
        ),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ElectronicComponentRGBDDataset',
        data_root='data/electronic_component',
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='mask/test',
            depth_path='depth/depth_npy/test'
        ),
        pipeline=test_pipeline
    )
)

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=10000,
    val_interval=500
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
# 建议：使用稍小的学习率以配合LayerNorm和残差学习
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,  # 可以尝试0.005-0.01
        momentum=0.9,
        weight_decay=0.0005
    ),
    clip_grad=None
)

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=10000,
        by_epoch=False
    )
]

# 钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=5,
        save_best='mIoU',
        rule='greater',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# 日志配置
log_processor = dict(by_epoch=False)

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 随机性
randomness = dict(seed=0)

# 工作目录
work_dir = 'work_dirs/electronic_component_depth_attention_v2_pspnet'

# 从RGB基线模型加载权重（可选）
load_from = 'work_dirs/electronic_component_pspnet/best_mIoU_iter_6500.pth'
