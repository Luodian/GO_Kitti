import subprocess
import os
import re
import json
import numpy as np


# cmd: ps -ef | grep python
# item: suffix parameter
def subproc(cmd):
    print(cmd)
    try:
        out_bytes = subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        out_bytes = e.output  # Output generated before error
        code = e.returncode  # Return code


# extract initial mAP value from a line
# input:
#    25.31   49.22   22.86   18.87   38.07   87.72
# ret: 25.31
def extract_mAP(line):
    ret = ""
    for i in range(len(line)):
        if line[i] is not " ":
            for j in range(i, len(line)):
                if line[j] is not " ":
                    ret += line[j]
                else:
                    return ret


# read all_results.txt file sequentially and append average mAP value at last line
def collect_info(root_path, test_lists):
    parent_dict = dict()
    for item in test_lists:
        mAP_pool = []
        step_pool = []
        info_dict = dict()
        full_path = os.path.join(root_path, item, 'all_results.txt')
        print(full_path)

        lines = open(full_path).readlines()
        line_count = len(lines)

        for i in range(3, line_count, 4):
            current_line = lines[i]
            mAP = extract_mAP(current_line)
            print(mAP)
            mAP_pool.append(mAP)

        for i in range(0, line_count, 4):
            current_line = lines[i]
            steps = re.findall("step(\d+)\.pth", current_line)
            if len(steps) is not 0:
                step_pool.append(steps[0])

        assert len(mAP_pool) == len(step_pool)

        for i in range(len(mAP_pool)):
            mAP = mAP_pool[i]
            step = step_pool[i]
            info_dict.update({step: mAP})

        parent_dict.update({item: info_dict})

    average_dict = dict()
    union_keys = set()

    # 求子集之间的并集
    for item in parent_dict:
        union_keys = parent_dict[item].keys() | union_keys

    print(union_keys)

    # 初始化average_dict为列表
    for item in union_keys:
        average_dict.update({item: []})

    # 将parent_dict中的每个数放到average_dict中
    for item in parent_dict:
        for subitem in parent_dict[item]:
            average_dict[subitem].append(parent_dict[item][subitem])

    # 将average_dict中的列表转换为平均值
    for item in average_dict:
        sub_list = average_dict[item]
        sub_list = [float(i) for i in sub_list]
        average_mAP = np.mean(sub_list)
        average_dict[item] = float(average_mAP)

    # 分别将p_dict和a_dict中keys()的类型从str转换为float

    for key in average_dict:
        average_dict[int(key)] = average_dict.pop(key)

    # 存parent_dict的详细信息
    with open(root_path + ".json", "w") as f:
        json.dump(parent_dict, f)

    # 存average_dict中的平均值信息
    average_json_save_path = "{}_average".format(root_path)

    f = open(average_json_save_path + ".txt", "w")
    f.writelines("Average\n")
    for key in sorted(average_dict.keys()):
        lines = "{} : {}\n".format(key, average_dict[key])
        f.writelines(lines)

    f.close()
    return parent_dict

    # sum = 0
    # for subitem in mAP_pool:
    #     sum += float(subitem)
    # average_mAP = sum / len(mAP_pool)
    #
    # store_line = "Average mAP: {}".format(average_mAP)
    # f = open(full_path, 'a')
    # last_line = open(full_path).readlines()[-1]
    # if last_line.startswith("Average") is False:
    #     f.writelines(store_line)
    #     f.close()


# 指定需要测试的model列表
train_lists = ["kitti_train_180_part1",
               "kitti_train_180_part2",
               "kitti_train_180_part3",
               "kitti_train_180_part4",
               "kitti_train_180_part5"]

# 指定测试列表，一个train set对应一个相应的val set
test_lists = ["coco_kitti_val_20_part1",
              "coco_kitti_val_20_part2",
              "coco_kitti_val_20_part3",
              "coco_kitti_val_20_part4",
              "coco_kitti_val_20_part5"]

method_name = "CS_KT_CV"
root_path = "/nfs/project/libo_i/go_kitti/model_test/{}".format(method_name)

for i, item in enumerate(train_lists):
    subproc("bash /nfs/project/libo_i/go_kitti/setup_shell/cross_validation/test.sh {} {} {}".format(item, test_lists[i], method_name))

parent_dict = collect_info(root_path, train_lists)
