from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import json
import os
import subprocess
import sys
from collections import defaultdict

# Use a non-interactive backend
import matplotlib

matplotlib.use('Agg')

import cv2

import torch

sys.path.insert(0, '/nfs/project/libo_i/mask-rcnn.pytorch/lib')
from core.config import cfg
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.json_vis as vis_utils
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')

    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    parser.add_argument(
        '--vis_segms_only', type=distutils.util.strtobool, default=True)

    parser.add_argument('--load_json', help='directory to save demo results', default="load_json")

    args = parser.parse_args()

    return args


def process_input_json(segms_json, im_list, cls_id_map):
    # cls_id_map: 66->38
    ret_segms_dict = {}

    for im_name in im_list:
        ret_segms_dict.update({im_name: []})
        # 放置空的框填数
        for i in range(0, 38):
            ret_segms_dict[im_name].append([])

    for id, im_name in enumerate(im_list):
        print(id, im_name)
        for item in segms_json:
            if item['image_id'] == im_name:
                try:
                    cls_id = cls_id_map[item['category_id']]
                except KeyError:
                    print("Key Error ", item['category_id'])
                else:
                    ret_segms_dict[im_name][cls_id].append(item['segmentation'])

    return ret_segms_dict


def process_score_json(submit_json, im_list, cls_id_map):
    ret_score_dict = {}

    for im_name in im_list:
        ret_score_dict.update({im_name: []})
        # 放置空的框填数
        for i in range(0, 38):
            ret_score_dict[im_name].append([])

    for id, im_name in enumerate(im_list):
        print(id, im_name)
        for item in submit_json:
            if item['image_id'] == im_name:
                try:
                    cls_id = cls_id_map[item['category_id']]
                except KeyError:
                    print("Key Error ", item['category_id'])
                else:
                    ret_score_dict[im_name][cls_id].append(item['score'])

    return ret_score_dict


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("mapillary"):
        dataset = datasets.get_mapillary_dataset()
        cfg.MODEL.NUM_CLASSES = 38
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    if args.image_dir:
        imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images

    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    im_list = []
    for i in range(num_images):
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        im_list.append(im_name)

    processed_save_path = "/nfs/project/libo_i/mask-rcnn.pytorch/vis_mid_result.json"
    submit_json_path = "/nfs/project/libo_i/mask-rcnn.pytorch/submit_json/map_trainval_29999/segmentations_coco_2017_test_multi_results.json"

    if os.path.exists(processed_save_path) is False:
        config_path = "/nfs/project/libo_i/mask-rcnn.pytorch/config.json"
        with open(config_path) as f:
            config_json = json.load(f)

        cls_id_map = {}

        cnt = 1
        for id, item in enumerate(config_json['labels']):
            if item['instances']:
                cls_id_map.update({id + 1: cnt})
                cnt += 1

        with open(submit_json_path) as f:
            segms_json = json.load(f)

        # processed_segms_json使用im_name索引具体的cls_segms
        processed_segms_list = process_input_json(segms_json, im_list, cls_id_map)

        with open(processed_save_path, 'w') as f:
            json.dump(processed_segms_list, f)
    else:
        with open(processed_save_path) as f:
            processed_segms_list = json.load(f)

    score_path = "/nfs/project/libo_i/mask-rcnn.pytorch/vis_mid_score.json"

    if os.path.exists(score_path) is False:

        config_path = "/nfs/project/libo_i/mask-rcnn.pytorch/config.json"
        with open(config_path) as f:
            config_json = json.load(f)

        cls_id_map = {}
        cnt = 1
        for id, item in enumerate(config_json['labels']):
            if item['instances']:
                cls_id_map.update({id + 1: cnt})
                cnt += 1

        with open(submit_json_path) as f:
            segms_json = json.load(f)

        # processed_segms_json使用im_name索引具体的cls_segms
        score_list = process_score_json(segms_json, im_list, cls_id_map)

        with open(score_path, 'w') as f:
            json.dump(score_list, f)

    else:
        with open(score_path) as f:
            score_list = json.load(f)

    # assert len(processed_segms_list) == num_images

    for i in range(num_images):
        print('img {} \ {}'.format(i, num_images))
        im = cv2.imread(imglist[i])
        assert im is not None

        timers = defaultdict(Timer)

        # cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)
        cls_boxes = None
        cls_keyps = None

        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        cls_segms = processed_segms_list[im_name]
        score = score_list[im_name]

        # tolist_box = [[v.tolist() for v in i] for i in cls_boxes]
        # all_segms[im_name].update({'boxes': tolist_box})

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            score,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

    if args.merge_pdfs and num_images > 1:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
