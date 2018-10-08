from PIL import Image
import cv2
import numpy as np
import json

kitti_id_map = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
kitti_category = [{"id": 24, "name": "person", "supercategory": "human"},
                  {"id": 25, "name": "rider", "supercategory": "human"},
                  {"id": 26, "name": "car", "supercategory": "vehicle"},
                  {"id": 27, "name": "truck", "supercategory": "vehicle"},
                  {"id": 28, "name": "bus", "supercategory": "vehicle"},
                  {"id": 29, "name": "carvanan", "supercategory": "vehicle"},
                  {"id": 30, "name": "trailer", "supercategory": "vehicle"},
                  {"id": 31, "name": "train", "supercategory": "vehicle"},
                  {"id": 32, "name": "motorcycle", "supercategory": "vehicle"},
                  {"id": 33, "name": "bicycle", "supercategory": "vehicle"}]


def anlyze_single_image():
    instance = Image.open("/nfs/project/libo_i/go_kitti/data/training/instance/000000_10.png")
    instance_array = np.array(instance, dtype=np.uint16)
    instance_label_array = np.array(instance_array // 256, dtype=np.uint8)
    instance_labels = np.unique(instance_label_array)

    for ins_label_i in instance_labels:
        if ins_label_i in kitti_id_map:
            ins_array_label_i_mask = instance_label_array == ins_label_i
            instance_array_label_i = instance_array * ins_array_label_i_mask
            instance_ids_array = np.array(instance_array_label_i % 256, dtype=np.uint8)
            back_value = instance_ids_array.max() + 1
            instance_ids_array[~ins_array_label_i_mask] = back_value
            ins_ids = np.unique(instance_ids_array)
            for ins_id in ins_ids[:-1]:
                instance_label_id_mask = instance_ids_array == ins_id
                instance_label_id_mask = np.array(instance_label_id_mask, dtype=np.uint8)


import os
import json

with open("/nfs/project/data/apolloscape/coco/train.json") as f:
    data = json.load(f)
    print(len(data['images']))
