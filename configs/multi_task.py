_base_ = [
    './datasets/multi_task_full_mask.py',
    './models/mobilenet_multi_task.py',
    './_base_/schedules/imagenet_bs256_epochstep.py',
    './_base_/default_runtime.py'
]
