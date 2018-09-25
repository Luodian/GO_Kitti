import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import json
import numpy as np
import os


def area_stat(path, save_path):
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    print("Task starts")
    print(save_path )

    id2name = dict()
    area_info = dict()
    bbox_info = dict()

    with open(path, 'r') as f:
        data = json.load(f)
        ann = data['annotations']
        cat_label = data['categories']
        num_imgs = len(data['images'])

        for item in cat_label:
            id2name.update({item['id']: item['name']})

        for v in id2name.values():
            area_info.update({v: list()})

        for v in id2name.values():
            bbox_info.update({v: list()})

        # do area stats
        for item in ann:
            area_info[id2name[item['category_id']]].append(item['area'])

        # do bbox stats
        for item in ann:
            bbox_info[id2name[item['category_id']]].append((item['bbox'][2], item['bbox'][3]))

        # plt area stat info
        for k, v in area_info.items():
            label_name = k
            num_list = v

            print(k, len(v), len(v) / num_imgs)

            plt.subplot(211)
            plt.title("{}-{}-{}".format(label_name, len(num_list), len(num_list) / num_imgs))
            plt.bar(range(len(num_list)), np.sort(num_list))

            plt.subplot(212)
            plt.title("bbox size")
            shape_list = bbox_info[k]
            x = [i[0] for i in shape_list]
            y = [i[1] for i in shape_list]
            plt.scatter(x, y)

            plt.savefig(save_path + "/" + label_name + "_area.png")
            plt.close()

        print("Done\n")


json_lists = ["/nfs/project/data/apolloscape/coco/train.json",
              "/nfs/project/data/cityscape/coco/instancesonly_filtered_gtFine_train.json",
              "/nfs/project/data/kitti/annotations/training_new.json",
              "/nfs/project/data/mapillary/annotation/training.json"]
dir_lists = ["/nfs/project/libo_i/go_kitti/stats_info/apollo",
             "/nfs/project/libo_i/go_kitti/stats_info/cityscape",
             "/nfs/project/libo_i/go_kitti/stats_info/kitti",
             "/nfs/project/libo_i/go_kitti/stats_info/mapillary"]

for i, item in enumerate(json_lists):
    json_path = item
    dir_path = dir_lists[i]
    area_stat(json_path, dir_path)
