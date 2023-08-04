# dataset settings
dataset_type = 'MultiTaskDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 200), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs',multi_task_fields=('gt_label', )),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 200), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs',multi_task_fields=('gt_label', ))
]

train_dataloader = dict(
    batch_size=256,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        ann_file='train.json',
        data_root='../',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=256,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        ann_file='test.json',
        data_root='../',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
test_dataloader = dict(
    batch_size=128,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        ann_file='test.json',
        data_root='../',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics={
        'intel': [dict(type='Accuracy', topk=(1, ))],
        'indoor': [dict(type='Accuracy', topk=(1, ))]
    })

test_dataloader = val_dataloader
test_evaluator = val_evaluator
