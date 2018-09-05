""" Perform the data statistics of the train set"""
# coding:utf-8
import os, sys
import json
from collections import defaultdict
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from pycocotools.coco import COCO
import gc
import copy

AVG = 100

pro_root = os.path.abspath('..')
os.chdir(pro_root)

MAX_NUM_PER_CLS = 10000


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


def gen_image_info(chip_id, image_name, image, save_dir):
    chip_name = image_name.replace('.jpg', '_{}.jpg'.format(chip_id))
    image_info = {
        "id": chip_id,
        "file_name": chip_name,
        "width": image.size[0],
        "height": image.size[1]
        # "date_captured": date_captured,
        # "license": license_id,
        # "coco_url": coco_url,
        # "flickr_url": flickr_url
    }
    return image_info


def statistic(path_to_ann):
    train_set = json.load(open(path_to_ann, 'r'))
    annotations = train_set['annotations']
    images = train_set['images']
    cls_to_img_ids = defaultdict(list)
    id_to_img = defaultdict(str)

    for img in images:
        id_to_img[img['id']] = img['file_name']
    for anno in annotations:
        cls_to_img_ids[anno['category_id']].append(anno['image_id'])
    return cls_to_img_ids, id_to_img


def aug_seq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([sometimes(iaa.OneOf([
        iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
        iaa.AverageBlur(k=(0.5, 3)),
        # blur image using local means with kernel sizes between 0.5 and 3
        iaa.MedianBlur(k=(1, 5)),
        # blur image using local medians with kernel sizes between 1 and 4
    ])),
        sometimes(iaa.OneOf([
            iaa.Multiply((0.8, 1.2)),
            iaa.Add((-35, 25)),
        ])),
        sometimes(iaa.ContrastNormalization((0.85, 1.60))),
    ], random_order=True)
    return seq


def save_data(annFile):
    annFile_new = annFile.replace('.json', '_new.json')
    with open(annFile_new, 'w') as af:
        json.dump(anno_json, af)


def add_annotation(aug_time, image_id, per, cur_max_img_id, cur_max_ann_id, coco):
    global anno_json
    anno_ids_to_img = coco.getAnnIds(imgIds=image_id)
    for i in range(aug_time):
        image_info = copy.deepcopy(coco.imgs[image_id])
        image_info['id'] = cur_max_img_id + i + 1
        image_info['file_name'] = image_info['file_name'].replace('.jpg', '_aug_{}.jpg'.format(per * aug_time + i))
        anno_json['images'].append(image_info)
        for id, anno_id in enumerate(anno_ids_to_img):
            anno = copy.deepcopy(coco.anns[anno_id])
            anno['id'] = cur_max_ann_id + i * aug_time + id + 1
            anno['image_id'] = image_info['id']
            anno_json['annotations'].append(anno)
    cur_max_img_id += aug_time
    cur_max_ann_id = cur_max_ann_id + aug_time * len(anno_ids_to_img)
    return cur_max_img_id, cur_max_ann_id


def get_max_id(item):
    max_id = 0
    for key, _ in item.items():
        if key > max_id:
            max_id = key
    return max_id


# image Augment
def imageaug(coco, image_dir, save_dir, ann_json, aug_time, period=300):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    catToImgs = coco.catToImgs
    imgToAnns = coco.imgToAnns
    images = coco.imgs
    cats = coco.cats

    cur_max_ann_id = get_max_id(coco.anns)
    cur_max_img_id = get_max_id(images)

    # catToImgLst = []
    # for cat, imgs in catToImgs.items():
    #     catToImgLst.append(len(imgs))
    #
    # sort_id = np.argsort(catToImgLst)
    # sortedCatToImgLst = np.sort(catToImgLst)
    #
    # most_cat_ids = sort_id[-3:]
    # most_img_ids = []
    # for most_id in most_cat_ids:
    #     most_img_ids.extend(coco.getImgIds(catIds=most_id))
    # most_img_ids = list(set(most_img_ids))
    #
    # min_cat_id = sort_id[0]
    # min_cat_cnt = sortedCatToImgLst[0]
    #
    # avg_cnt = float(np.sum(catToImgLst)) / len(catToImgLst)
    #
    # aug_time = avg_cnt / min_cat_cnt

    # to_process_img_ids = []
    # min_cat_ids = coco.getImgIds(catIds=min_cat_id)
    # for img_id in min_cat_ids:
    #     if img_id not in most_img_ids:
    #         to_process_img_ids.append(img_id)
    # aug_time = int(aug_time * (len(min_cat_ids) / float(len(to_process_img_ids))))
    to_process_img_ids = list(images.keys())
    #period = 300
    for img_id in to_process_img_ids:
        img_name = images[img_id]['file_name']
        img_path = os.path.join(image_dir, img_name)
        im = cv2.imread(img_path)
        h, w = im.shape[:2]
        im = im.reshape(1, h, w, 3)
        real_aug_time = period if aug_time > period else aug_time
        for per in range(int(aug_time / period + 1)):
            ims = im.repeat(real_aug_time, axis=0)
            seq = aug_seq()
            ims_aug = seq.augment_images(ims)
            for i in range(len(ims_aug)):
                aug_image_file = img_name.replace('.jpg', '_aug_{}.jpg'.format(per * period + i))
                save_path = os.path.join(save_dir, aug_image_file)
                cv2.imwrite(save_path, ims_aug[i])
            cur_max_img_id, cur_max_ann_id = add_annotation(real_aug_time, img_id, per, cur_max_img_id, cur_max_ann_id, coco)
            del ims_aug, ims
            gc.collect()
        del im
        gc.collect()


if __name__ == "__main__":
    task = 'train'
    annFile = '/nfs/project/data/kitti/annotations/training.json'

    image_dir = '/nfs/project/data/kitti/training/image_2'
    save_dir = '/nfs/project/data/kitti/aug'
    aug_time = 20
    period = 300
    anno_json = json.load(open(annFile))
    coco = COCO(annFile)
    imageaug(coco, image_dir, save_dir, anno_json, aug_time, period=300)
    save_data(annFile)
