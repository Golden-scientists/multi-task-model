_base_ = [
    './datasets/second_task.py',
    './models/mobilenet_singl_task.py',
    './_base_/schedules/imagenet_bs256_epochstep.py',
    './_base_/default_runtime.py'
]
