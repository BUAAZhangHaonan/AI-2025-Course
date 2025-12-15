data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
dataset_type = 'ElectronicComponentDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=5,
        rule='greater',
        save_best='mIoU',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='PSPHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        341,
        341,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=10000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='images/test', seg_map_path='mask/test'),
        data_root='data/electronic_component',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        type='ElectronicComponentDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='images/train', seg_map_path='mask/train'),
        data_root='data/electronic_component',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.5,
                    1.5,
                ),
                hue_delta=18,
                saturation_range=(
                    0.5,
                    1.5,
                ),
                type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='ElectronicComponentDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='images/val', seg_map_path='mask/val'),
        data_root='data/electronic_component',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConvertInstanceToSemantic'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        type='ElectronicComponentDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/electronic_component_pspnet'
