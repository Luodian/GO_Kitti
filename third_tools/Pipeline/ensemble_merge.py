# coding:utf-8
from __future__ import print_function

import json
import multiprocessing
import sys

import numpy as np

sys.path.insert(0, '/nfs/project/libo_i/mask-rcnn.pytorch/lib')
from utils.segms import rle_mask_nms
from utils.segms import rle_mask_voting
import utils.boxes as box_utils
import os
import time

# path to save multi-model output jsons
mask_json_dir = sys.argv[1].rstrip('/')
# path to save ensambe json
ensamble_json_path = os.path.join("/".join(mask_json_dir.split('/')[:-1]),
                                  mask_json_dir.split('/')[-1].rstrip('/') + '_ensamble.json')
print(ensamble_json_path)
BBOX_NMS = True
MEM_SHARE = False

all_boxes = []
all_segms = []


def merge(json_dir='/nfs/project/guyang/ensamble/Mapillary-Team/ensamble/ensamble_3'):
    '''
    func:
    merge multi-detection output to only one output
    inputs:
    json_dir: str, path to save multi-model output json
    returns:
    all_boxes: use in program
    all_segms: use in program
    '''
    for index, json_name in enumerate(os.listdir(json_dir)):
        json_path = os.path.join(json_dir, json_name)
        # print("process {}".format(json_path))
        if not json_name.split('.')[-1] == 'json':
            continue
        json_ = json.loads(open(json_path).read())
        json_boxes = [[np.array(v).reshape((-1, 5)) for v in i] for i in json_['all_boxes']]
        json_segms = json_['all_segms']
        if index == 0:
            all_boxes = json_boxes
            all_segms = json_segms
        else:
            for cls_id in range(len(json_boxes)):
                for img_id in range(len(json_boxes[cls_id])):
                    all_boxes[cls_id][img_id] = np.concatenate((all_boxes[cls_id][img_id], json_boxes[cls_id][img_id]),
                                                               axis=0)
                    all_segms[cls_id][img_id] = all_segms[cls_id][img_id] + json_segms[cls_id][img_id]
    return all_boxes, all_segms


def run_task(start_img_id, end_img_id, cls_id):
    # print("Task from {} to {}".format(start_img_id, end_img_id))
    ret_boxes = []
    ret_segms = []
    for i in range(11):
        ret_boxes.append([])
        ret_segms.append([])
        for j in range(end_img_id - start_img_id):
            ret_boxes[i].append([])
            ret_segms[i].append([])

    for img_id in range(start_img_id, end_img_id):
        if len(all_segms[cls_id]) != 0:
            if len(all_segms[cls_id][img_id]) != 0:
                segms = all_segms[cls_id][img_id]
                boxes = all_boxes[cls_id][img_id]
                # nms
                nms_start_time = time.time()
                if BBOX_NMS:
                    boxes = np.array(boxes).astype(np.float32, copy=False)
                    keep = box_utils.nms(boxes, 0.5)
                else:
                    keep = rle_mask_nms(segms, boxes, 0.5, mode='IOU')

                # nms_end_time = time.time()
                # print('nms spend {:.2f}s'.format(nms_end_time - nms_start_time))

                top_boxes = boxes[keep, :]
                top_segms = []
                for index in keep:
                    top_segms.append(segms[index])

                vote_start_time = time.time()
                # mask_vote
                # top_segms = rle_mask_voting(
                #     top_segms,
                #     segms,
                #     boxes,
                #     0.9,
                #     0.5
                # )

                # trans from byte to str for json format
                if not top_segms is None and len(top_segms) > 0:
                    for id, s in enumerate(top_segms):
                        if type(s['counts']) == str:
                            top_segms[id]['counts'] = s['counts']
                        else:
                            top_segms[id]['counts'] = str(s['counts'], 'utf-8')
                vote_end_time = time.time()
                print('Img:{} cls:{} vote spend {:.2f}s'.format(img_id, cls_id, vote_end_time - vote_start_time))
                ret_boxes[cls_id][img_id - start_img_id].append(top_boxes)
                ret_segms[cls_id][img_id - start_img_id].append(top_segms)

    return ret_boxes, ret_segms


def multi_process(process_num, pa_cls_id):
    p = multiprocessing.Pool(processes=process_num)
    p_res = []
    # 通过第1类获取img_length
    img_length = len(all_segms[1])
    lock = multiprocessing.Lock()
    for i in range(process_num):
        start_img_id = int(i * (img_length / process_num))
        end_img_id = min(int(start_img_id + (img_length / process_num)), img_length)
        p_res.append(p.apply_async(run_task, args=(start_img_id, end_img_id, pa_cls_id)))
    print("get ok")
    p.close()
    p.join()

    print("Processes done! Gathering data...")

    Interval = int(img_length / process_num)

    for id, item in enumerate(p_res):
        for cls_id in range(1, 11):
            start_img_id = int(id * (img_length / process_num))
            end_img_id = min(int(start_img_id + (img_length / process_num)), img_length)
            for img_id in range(start_img_id, end_img_id):
                all_boxes[cls_id][img_id] = item.get()[0][cls_id][img_id - start_img_id]
                all_segms[cls_id][img_id] = item.get()[1][cls_id][img_id - start_img_id]

    print("Mask_vote")

    output_json = {'all_boxes': [],
                   'all_segms': [],
                   }
    for i, item_i in enumerate(all_boxes):
        for j, item_j in enumerate(item_i):
            if type(item_j) != list:
                all_boxes[i] = item_j.tolist()
            for k, item_k in enumerate(item_j):
                if type(item_k) != list:
                    all_boxes[i][j] = item_k.tolist()

    for i, item_i in enumerate(all_segms):
        for j, item_j in enumerate(all_segms[i]):
            if len(item_j) == 0 and type(item_j) == list:
                all_segms[i][j] = []
            else:
                all_segms[i][j] = all_segms[i][j][0]

    # output_json['all_boxes'] = all_boxes
    # output_json['all_segms'] = all_segms
    return all_boxes, all_segms


print("Merge")
start_time = time.time()
output_json = None
for cls_id in range(0, 10):
    json_dir = os.path.join(mask_json_dir, "class{}".format(cls_id))
    all_boxes, all_segms = merge(json_dir)
    if output_json is None:
        output_json = {'all_boxes': [], 'all_segms': []}
        output_json['all_boxes'] = all_boxes
        output_json['all_segms'] = all_segms

    cls_boxes, cls_segms = multi_process(10, cls_id)
    output_json['all_boxes'][cls_id] = cls_boxes[cls_id]
    output_json['all_segms'][cls_id] = cls_segms[cls_id]
open(ensamble_json_path, 'w').write(json.dumps(output_json))
end_time = time.time()
print("Spend {:.2f}s".format(end_time - start_time))
print("Finish")
