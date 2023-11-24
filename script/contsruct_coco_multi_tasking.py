import os
import json

def construct_coco(name:str,path_data:str):
    """
    function to construct a coco json(annotations)
    parameters:
        name : name of the coco json
        path_data : path to the intel data
    return:
        coco.json
    """
    coco = {
        "metainfo": {
            "tasks": [
                "intel1",
                "intel2"
            ]
        },
        "data_list":[]
    }

    class_intel1 = ['buildings', 'mountain', 'street']
    for file in class_intel1:
        path_class = os.path.join(path_data, file)
        image_names = os.listdir(path_class)
        if len(image_names)<100:
            raise Exception(file)
        for image in image_names:
            image_path=os.path.join(path_class,image)
            coco['data_list'].append({"img_path": image_path,"gt_label": {"intel1": class_intel1.index(file)}})


    class_intel2 = ['forest', 'glacier', 'sea']
    for file in class_intel2:
        path_class = os.path.join(path_data, file)
        image_names = os.listdir(path_class)
        if len(image_names)<100:
            raise Exception(file)
        for image in image_names:
            image_path=os.path.join(path_class,image)
            coco['data_list'].append({"img_path": image_path,"gt_label": {"intel2": class_intel2.index(file)}})

    json_object = json.dumps(coco, indent=4)
        
    with open(name, "w") as outfile:
        outfile.write(json_object)
construct_coco("../train.json","../data/intel/seg_train")
construct_coco("../test.json","../data/intel/seg_test")

