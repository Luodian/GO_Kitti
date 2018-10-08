import json
import os
import subprocess

# /nfs/project/libo_i/go_kitti/setup_shell/test/ensemble.sh

def output_json(exp_name, ckpt_path, json_save_path, param):
    # 第一个参数：实验名字
    # 第二个参数：ckpt的路径
    # 第三个参数：json_path存下来的路径

    cmd = "bash /nfs/project/libo_i/go_kitti/setup_shell/test/ensemble.sh {} {} {} {}".format(exp_name, ckpt_path,
                                                                                              json_save_path, param)
    print(cmd)
    subprocess.call(cmd, shell=True)


exp_name_list = ["X101_2237",
                 'MAP_aug_101X_KT',
                 'MAP_full_101X_KT']

ckpt_list = ['/nfs/project/libo_i/go_kitti/train_output/X101_2237/kitti_train/ckpt/model_step2999.pth',
             '/nfs/project/libo_i/go_kitti/train_output/MAP_aug_101X_KT/kitti_train/ckpt/model_step999.pth',
             '/nfs/project/libo_i/go_kitti/train_output/MAP_full_101X_KT/kitti_train/ckpt/model_step3999.pth']

json_save_path_list = []
json_root_path = "/nfs/project/libo_i/go_kitti/json_output"

param_lists = ["INFER_OR_TEST True",
               "INFER_OR_TEST True M_ANCHOR True",
               "INFER_OR_TEST True M_ANCHOR True"]

# 组合每一个json_output的路径，用于指定输出位置
for item in exp_name_list:
    save_path = os.path.join(json_root_path, item)
    json_save_path_list.append(save_path)

# 对不同的ckpt进行输出json的操作，这里指定的ckpt_list要详细到具体的pth文件
for i in range(len(ckpt_list)):
    ckpt_path = ckpt_list[i]
    exp_name = exp_name_list[i]
    json_save_path = json_save_path_list[i]
    param = param_lists[i]

    output_json(exp_name, ckpt_path, json_save_path, param)
