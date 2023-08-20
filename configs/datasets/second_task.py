# dataset settings
dataset_type = 'ImageNet'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 200), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 200), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_prefix='..\data\intel\seg_train2',
        with_label=True,                # or False for unsupervised tasks
        classes=['forest', 'glacier', 'sea'],  # The name of every category.
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_prefix='..\data\intel\seg_test2',
        with_label=True,                # or False for unsupervised tasks
        classes=['forest', 'glacier', 'sea'],  # The name of every category.
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
test_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_prefix='..\data\intel\seg_test2',
        with_label=True,                # or False for unsupervised tasks
        classes=['forest', 'glacier', 'sea'],  # The name of every category.
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
