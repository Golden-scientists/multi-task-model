auto_scale_lr = dict(base_batch_size=256)
dataset_type = 'MultiTaskDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        task_heads=dict(
            intel1=dict(num_classes=3, type='LinearClsHead'),
            intel2=dict(num_classes=3, type='LinearClsHead')),
        type='MultiTaskHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.045, momentum=0.9, type='SGD', weight_decay=4e-05))
param_scheduler = dict(by_epoch=True, gamma=0.98, step_size=1, type='StepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='test.json',
        data_root='../',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=(
                256,
                200,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(multi_task_fields=('gt_label', ), type='PackMultiTaskInputs'),
        ],
        type='MultiTaskDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
test_evaluator = dict(
    task_metrics=dict(
        intel1=[
            dict(topk=(1, ), type='Accuracy'),
        ],
        intel2=[
            dict(topk=(1, ), type='Accuracy'),
        ]),
    type='MultiTasksMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=(
        256,
        200,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(multi_task_fields=('gt_label', ), type='PackMultiTaskInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='train.json',
        data_root='../',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=(
                256,
                200,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(multi_task_fields=('gt_label', ), type='PackMultiTaskInputs'),
        ],
        type='MultiTaskDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=(
        256,
        200,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(multi_task_fields=('gt_label', ), type='PackMultiTaskInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='test.json',
        data_root='../',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=(
                256,
                200,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(multi_task_fields=('gt_label', ), type='PackMultiTaskInputs'),
        ],
        type='MultiTaskDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_evaluator = dict(
    task_metrics=dict(
        intel1=[
            dict(topk=(1, ), type='Accuracy'),
        ],
        intel2=[
            dict(topk=(1, ), type='Accuracy'),
        ]),
    type='MultiTasksMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '../tmp/multi_tasks'
