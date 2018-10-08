# coding:utf-8

from PIL import Image
import cv2
import numpy as np
import os, sys
import pdb
import re
import pickle
import pprint
import traceback
from multiprocessing import Pool, Manager
# import datetime
import time
import json
from pycococreatortools import pycococreatortools
import matplotlib.pyplot as plt

Data_ROOT = '/nfs/project/libo_i/go_kitti/data'
path_to_ins = '{}/instance/'
path_to_img = '{}/image_2/'
path_to_save = '../vis/label/{}'


def gen_category_info(id, name, supercategory):
    cate_info = {}
    cate_info['id'] = id
    cate_info['name'] = name
    cate_info['supercategory'] = supercategory[-2]
    cate_info['supercategory_1'] = ''
    if len(supercategory) > 2:
        cate_info['supercategory_1'] = supercategory[-3]

    return cate_info


def gen_anno_info(anno_id, category_id, image_id, area, segmentation, bbox, iscrowd=0):
    annotation_info = {
        'id': anno_id,
        'category_id': category_id,
        'image_id': image_id,
        'area': area,
        'segmentation': segmentation,
        'bbox': bbox,
        'iscrowd': iscrowd}

    return annotation_info


def gen_image_info(image_id, file_name, image_size):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1]
        # "date_captured": date_captured,
        # "license": license_id,
        # "coco_url": coco_url,
        # "flickr_url": flickr_url
    }

    return image_info


########################
# type_id: 0, only for class with instance
# type_id: 1, for full class
########################
def gen_category(config_file, type_id=0):
    category = []
    orig_id = []
    with open(config_file, 'r') as cf:
        config = json.load(cf)
    labels = config['labels']
    label_id = 1
    if type_id == 0:
        for id, label in enumerate(labels):
            if label['instances'] and label['evaluate']:
                name = label['readable']
                supercategory = label['name'].split('--')
                category.append(gen_category_info(label_id, name, supercategory))
                label_id += 1
                orig_id.append(id)
    elif type_id == 1:
        for id, label in enumerate(labels):
            if label['evaluate']:
                name = label['readable']
                supercategory = label['name'].split('--')
                category.append(gen_category_info(label_id, name, supercategory))
                label_id += 1
                orig_id.append(id)
    return category, orig_id


def save_data(category, images, annotations, json_file):
    normal_images = [image for _, image in images.items()]
    normal_annotations = [anno for _, anno in annotations.items() if anno is not None]
    json_info = {
        'categories': category,
        'images': normal_images,
        'annotations': normal_annotations
    }
    with open(json_file, 'w') as sjf:
        json.dump(json_info, sjf)


def process_file(task, name, image_id, file_num, orig_id, images, annotations):
    no_suffix_name = os.path.splitext(name)[0]
    img_name = no_suffix_name + '.png'
    path_to_img_file = os.path.join(Data_ROOT, path_to_img.format(task), img_name)
    image = Image.open(path_to_img_file)
    image_info = gen_image_info(image_id, os.path.basename(path_to_img_file), image.size)
    images[image_id] = image_info
    try:
        path_to_instance_file = os.path.join(Data_ROOT, path_to_ins.format(task), name)
        instance = Image.open(path_to_instance_file)
        instance_array = np.array(instance, dtype=np.uint16)
        instance_label_array = np.array(instance_array // 256, dtype=np.uint8)
        instance_labels = np.unique(instance_label_array)
    except:
        print("ERROR")

    anno_id = image_id * 256
    for ins_label_i in instance_labels:
        if ins_label_i in orig_id:
            ins_array_label_i_mask = instance_label_array == ins_label_i
            instance_array_label_i = instance_array * ins_array_label_i_mask
            instance_ids_array = np.array(instance_array_label_i % 256, dtype=np.uint8)
            back_value = instance_ids_array.max() + 1
            instance_ids_array[~ins_array_label_i_mask] = back_value
            ins_ids = np.unique(instance_ids_array)
            for ins_id in ins_ids[:-1]:
                try:
                    instance_label_id_mask = instance_ids_array == ins_id
                    instance_label_id_mask = np.array(instance_label_id_mask, dtype=np.uint8)
                    # row_lst, col_lst = np.where(instance_label_id_mask == 1)
                    # y_min, x_min = np.min(row_lst), np.min(col_lst)
                    # y_max, x_max = np.max(row_lst), np.max(col_lst)
                    # bbox = [x_min, y_min, x_max - x_min + 1., y_max - y_min + 1.]
                    # mask_area = np.sum(instance_label_id_mask)
                    # contours = cv2.findContours(instance_label_id_mask, cv2.RETR_TREE,
                    #                            cv2.CHAIN_APPROX_SIMPLE)
                    # seg = []
                    # for cont in contours[0]:
                    #    [seg.append(i) for i in list(cont.flatten())]
                    cate_id = orig_id.index(ins_label_i) + 1
                    category_info = {'id': cate_id, 'is_crowd': 0}
                    # anno_info = gen_anno_info(anno_id, cate_id, image_id, mask_area, seg, bbox)
                    anno_info = pycococreatortools.create_annotation_info(
                        anno_id, image_id, category_info, instance_label_id_mask, tolerance=0)
                    annotations[anno_id] = anno_info
                    anno_id += 1
                except:
                    print("The area of the {}th instance of label {} in image {}. Please check it later.".format(
                        ins_id, ins_label_i, name))
    print("{}: Have Processed {}/{}".format(task, image_id, file_num))


def vis_file(task, name, label_ids, label_names, i, file_num):
    path_to_ins_file = os.path.join(Data_ROOT, path_to_ins.format(task), name)
    path_to_img_file = os.path.join(Data_ROOT, path_to_img.format(task), name)
    img = cv2.imread(path_to_img_file)
    instance = Image.open(path_to_ins_file)
    instance_array = np.array(instance, dtype=np.uint16)
    ins_shape = instance_array.shape
    im_size_min = np.min(ins_shape[0:2])
    im_size_max = np.max(ins_shape[0:2])
    instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    instance_labels = np.unique(instance_label_array)
    contours_lst = {}
    for ins_label_i in instance_labels:
        if int(ins_label_i) < 65:
            contours_lst[ins_label_i] = []
            ins_array_label_i_mask = instance_label_array == ins_label_i
    # ins_array_label_i_bin = np.array(ins_array_label_i_mask, dtype=np.uint8)
    instance_array_label_i = instance_array * ins_array_label_i_mask
    instance_ids_array = np.array(instance_array_label_i % 256, dtype=np.uint8)
    back_value = instance_ids_array.max() + 1
    instance_ids_array[~ins_array_label_i_mask] = back_value
    ins_ids = np.unique(instance_ids_array)
    for ins_id in ins_ids[:-1]:
        instance_label_id_mask = instance_ids_array == ins_id
        instance_label_id_bin = np.array(instance_label_id_mask, dtype=np.uint8)
        contours, _ = cv2.findContours(instance_label_id_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours_lst[ins_label_i].append(contours)
    # print("Something wrong in the {}th instance of label {}:{} in image {}. Please check it later.".format(ins_id,
    # ins_label_i,label_name, name))
    color_map = plt.cm.hsv(np.linspace(0, 1, 66)).tolist()
    for key, conts in contours_lst.items():
        cv2.drawContours(img, conts, -1, (0, 0, 255), 10)
        cv2.imread(path_to_save.format(name), img)
    print("Have Processed {}/{}".format(i + 1, file_num))


def statistic_bbox(p, task, label_ids, label_names, bboxes):
    data_path = os.path.join(Data_ROOT, path_to_ins.format(task))
    file_list = os.listdir(data_path)
    # pdb.set_trace()
    # file_list = file_list[:12]
    for i in range(len(file_list)):
        p.apply_async(vis_file,
                      args=(task, file_list[i], label_ids, label_names, bboxes, i, len(file_list),))
    p.close()
    p.join()


def vis_labels(p, task, label_ids, label_names):
    data_path = os.path.join(Data_ROOT, path_to_ins.format(task))
    file_list = os.listdir(data_path)
    # pdb.set_trace()
    file_list = file_list[:1]
    for i in range(len(file_list)):
        vis_file(task, file_list[i], label_ids, label_names, i, len(file_list))


# p.apply_async(vis_file, args=(task, file_list[i], label_ids, label_names, i, len(file_list),))
# p.close()
# p.join()


def gen_coco_data(p, orig_id, task, images, annotations, part):
    data_path = "/nfs/project/libo_i/go_kitti/data/training/image_2"
    file_list = os.listdir(data_path)

    num_img = len(part)

    for i in part:
        i = int(i)
        p.apply_async(process_file,
                      args=(task, file_list[i], i + 1, num_img, orig_id, images, annotations
                            ,))
    # p.apply_async(process_file, args=(task, file_list[select_list[i]], i+1, len(select_list), orig_id, images,
    # annotations,))
    p.close()
    p.join()


kitti_id_map = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
kitti_category = [{"id": 1, "name": "person", "supercategory": "human"},
                  {"id": 2, "name": "rider", "supercategory": "human"},
                  {"id": 3, "name": "car", "supercategory": "vehicle"},
                  {"id": 4, "name": "truck", "supercategory": "vehicle"},
                  {"id": 5, "name": "bus", "supercategory": "vehicle"},
                  {"id": 6, "name": "carvanan", "supercategory": "vehicle"},
                  {"id": 7, "name": "trailer", "supercategory": "vehicle"},
                  {"id": 8, "name": "train", "supercategory": "vehicle"},
                  {"id": 9, "name": "motorcycle", "supercategory": "vehicle"},
                  {"id": 10, "name": "bicycle", "supercategory": "vehicle"}]

if __name__ == '__main__':
    task = "training"
    config_file = os.path.join(Data_ROOT, 'config.json')

    import random

    lst = np.arange(0, 200)
    np.random.shuffle(lst)
    # 通过shuffle的操作将图片序号打混
    train_part = lst[:180]
    val_part = lst[180:]

    train_json_file = '/nfs/project/libo_i/go_kitti/data/annotations/{}_180_Part1.json'.format(task)
    # category, orig_id = gen_category(config_file)

    train_p = Pool()
    # task = 'validation'
    train_manager = Manager()
    train_images = train_manager.dict()
    train_annotations = train_manager.dict()

    gen_coco_data(train_p, kitti_id_map, task, train_images, train_annotations, train_part)
    save_data(kitti_category, train_images, train_annotations, train_json_file)
    print("Done!")

    val_json_file = '/nfs/project/libo_i/go_kitti/data/annotations/{}_20_Part1.json'.format(task)
    # category, orig_id = gen_category(config_file)

    val_p = Pool()
    # task = 'validation'
    val_manager = Manager()
    val_images = val_manager.dict()
    val_annotations = val_manager.dict()

    gen_coco_data(val_p, kitti_id_map, task, val_images, val_annotations, val_part)
    save_data(kitti_category, val_images, val_annotations, val_json_file)
    print("Done!")
