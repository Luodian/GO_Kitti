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

        if os.path.exists(full_path) is False:
            continue

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

    # 存average_dict中的平均值信息
    average_json_save_path = "{}_average".format(root_path)

    f = open(average_json_save_path + ".txt", "w")
    f.writelines("Average\n")
    for key in sorted(average_dict.keys()):
        lines = "{} : {}\n".format(key, average_dict[key])
        f.writelines(lines)

    f.close()

    # 存parent_dict的详细信息
    with open(root_path + ".json", "w") as f:
        json.dump(parent_dict, f)

    return parent_dict, average_dict


def merge_cv_results(average_dict, root_path, train_set_lists, exp_name):
    from operator import itemgetter
    formated_item = []
    for key in average_dict:
        formated_item.append({"steps": key, 'mAP': average_dict[key]})

    rows_by_mAP = sorted(formated_item, key=itemgetter('mAP'))
    best_item = rows_by_mAP[-1]
    steps = best_item['steps']

    cat_matrix = []

    for item in train_set_lists:
        item_path = os.path.join(root_path, item, "model_step{}.pth".format(steps))
        if os.path.exists(item_path) is False:
            continue

        results_lines = open(item_path).read()
        categories_value_list = re.findall("INFO json_dataset_evaluator.py: 231: (\S+)\n", results_lines)
        for index, subitem in enumerate(categories_value_list):
            if subitem == 'nan':
                continue
            else:
                categories_value_list[index] = float(subitem)

        cat_matrix.append(categories_value_list)

    average_list = []

    cat_nums = 10
    # 对矩阵逐类别计算均值
    for i in range(cat_nums * 2):
        sum_value = 0
        avai_cnt = 0
        for item in cat_matrix:
            sub_cat_value = item[i]
            if sub_cat_value != 'nan':
                sum_value += float(sub_cat_value)
                avai_cnt += 1

        if avai_cnt == 0:
            average_list.append('nan')
        else:
            avg_value = float(sum_value) / avai_cnt
            average_list.append(round(avg_value, 2))

    merged_results_path = os.path.join(root_path, "{}_{}_category_average.txt".format(exp_name, steps))

    with open(merged_results_path, 'w') as f:
        sum_value = 0
        cnt = 0
        for item in average_list[10:]:
            f.write(str(item) + ',')
            if item != 'nan':
                sum_value += float(item)
                cnt += 1

        avg_value = float(sum_value) / cnt
        f.write('\n' + str(avg_value))

    # # 填充会最后的merged_results文件里
    # filled_lists = []
    # for item in average_list:
    #     filled_lists.append("INFO json_dataset_evaluator.py: 231: {}".format(round(item, 2)))
    #
    # bbox_anchor_index = 0
    # for i in range(len(sample_file_lines)):
    #     if sample_file_lines[i].startswith("INFO json_dataset_evaluator.py: 231: "):
    #         bbox_anchor_index = i
    #         break
    #
    # assert bbox_anchor_index != 0
    #
    # for i in range(10):
    #     sample_file_lines[bbox_anchor_index + i] = filled_lists[i] + '\n'
    #
    # segms_anchor_index = 0
    # for i in range(bbox_anchor_index + 10, len(sample_file_lines)):
    #     if sample_file_lines[i].startswith("INFO json_dataset_evaluator.py: 231: "):
    #         segms_anchor_index = i
    #         break
    #
    # for i in range(10):
    #     sample_file_lines[segms_anchor_index + i] = filled_lists[i + 10] + '\n'


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


def group_run():
    method_name_lists = ["MAP_full_101X_KT",
                         "MAP_aug_101X_KT",
                         "X101_2237"]

    for pa_item in method_name_lists:
        # 这里的root_path+method_name_lists[i]，会形成一个具体的文件夹路径
        # 将生成的对应测试项目信息保存到里面
        root_path = "/nfs/project/libo_i/go_kitti/model_test/{}".format(pa_item)

        for i, item in enumerate(train_lists):
            # 这是一个系列测试脚本，用于运行在一个父目录下的一系列子目录中的ckpt文件
            # M_anchor_test.sh 是一个测试脚本，其中指定的参数有三个
            # 1. 训练集名字：用于找到对应的模型输出路径
            # 2. 测试集名字：用于匹配对应的测试文件路径
            # 3. 父项目名字，用于生成最终的输出文件目录
            subproc(
                "bash /nfs/project/libo_i/go_kitti/setup_shell/test/M_anchor_test.sh {} {} {}".format(item,
                                                                                                      test_lists[i],
                                                                                                      pa_item))

        parent_dict, average_dict = collect_info(root_path, train_lists)
        merge_cv_results(average_dict, root_path, train_lists, pa_item)


# single_run的目的是省去了pa_item这个步骤，主要用于前期测试和后期调试
def single_run():
    method_name = "MAP_aug_101X_KT"
    root_path = "/nfs/project/libo_i/go_kitti/model_test/{}".format(method_name)
    # for i, item in enumerate(train_lists):
    #     subproc(
    #         "bash /nfs/project/libo_i/go_kitti/setup_shell/test/No_anchor_test.sh {} {} {}".format(item,
    #                                                                                                test_lists[i],
    #                                                                                                method_name))

    parent_dict, average_dict = collect_info(root_path, train_lists)
    merge_cv_results(average_dict, root_path, train_lists, method_name)


group_run()
