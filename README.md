# Introduction :
Testing the limits of a multi-task model with [Intel Image Classification ]([https://www.kaggle.com/datasets/puneet6060intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)) datasets ,we split this  dataset into two subsets (intel1 and intel 2), the first contains classes ('buildings', 'mountain', 'street') and the second contains ('forest', 'glacier', 'sea'), the idea is to build a single model for each task and compare it to the multitasking model with both tasks:

# Installation : 
Below are quick steps for installation:
```py
conda env create -f env.yaml
conda activate multi-tasking
mim install mmcv
```

---------------------------------------------------------------------------------------------------------
## what you need to know : 
    - configs/first_task.py : is the config for task 1 : intel 1
    - configs/second_task.py : is the config for task 2 : intel 2
    - configs/multi_task.py : is the config for multi tasking both task
    - annotations for multi_tasking : download data and use script/construct_coco_multi_tasking.py

----------------------------------------------------------------------------------------------------------

to train a model use the below command
```py
python3 tools/train.py path_to_config --work-dir path_save_logs

```