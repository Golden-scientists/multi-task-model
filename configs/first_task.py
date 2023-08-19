_base_ = [
    './datasets/first_task.py',
    './models/mobilenet_singl_task.py',
    './_base_/schedules/imagenet_bs256_epochstep.py',
    './_base_/default_runtime.py'
]
