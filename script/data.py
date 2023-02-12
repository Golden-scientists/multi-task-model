import os
import json
def construct_coco(name,path,path_text,path2):
    coco = {
        "metainfo": {
            "tasks": [
                "indoor",
                "intel"
            ]
        },
        "data_list":[]
    }
    for file in os.listdir(path):
        d = os.path.join(path, file)
        for image in os.listdir(d):
            image_path=os.path.join(d,image)
            coco['data_list'].append({"img_path": image_path,"gt_label": {"intel": file}})

    fichier = open(path_text, "r")
    for f in fichier:
        coco['data_list'].append({"img_path": os.path.join(path2, f),"gt_label": {"Indoor": f.split('/')[0]}})

    # Serializing json
    json_object = json.dumps(coco, indent=4)
        
    # Writing to sample.json
    with open(name, "w") as outfile:
        outfile.write(json_object)
construct_coco("intel_train.json","./data/intel/seg_train","./data/indoor/TrainImages.txt","./data/indoor")
construct_coco("intel_test.json","./data/intel/seg_test","./data/indoor/TestImages.txt","./data/indoor")

