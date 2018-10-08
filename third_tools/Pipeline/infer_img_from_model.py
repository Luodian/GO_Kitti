import subprocess
import os
import cv2
import matplotlib
import sys

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def subproc(cmd):
    print(cmd)
    try:
        out_bytes = subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        out_bytes = e.output  # Output generated before error
        code = e.returncode  # Return code


# assigned_model = "/nfs/project/libo_i/go_kitti/train_output/X101_2237/kitti_train/ckpt/model_step2999.pth"
# exp_name = "X101_2237"
# 这里用于指定模型路径和模型名字，会在infer_output/下面生成
assigned_model = sys.argv[1]
exp_name = sys.argv[2]

# 下面的参数不需要动，默认配置好了原图，top1's results, our results的三个路径

infer_output_dir = "/nfs/project/libo_i/go_kitti/infer_output/{}".format(exp_name)
cmp_rush_rob = "/nfs/project/libo_i/go_kitti/data/testing/rush_rob_results"
gt_path = "/nfs/project/libo_i/go_kitti/data/testing/kitti_demo_image"

# 这个命令可以简化为去引导一个文件，但是用命令的方式写更直观，传参数也更加方便，个人倾向于不要单独开新的shell

infer_cmd = "python3 /nfs/project/libo_i/go_kitti/tools/infer_simple.py \
            --dataset kitti \
            --cfg /nfs/project/libo_i/go_kitti/configs/baselines/kitti_final.yaml \
            --load_ckpt {} \
            --image_dir /nfs/project/libo_i/go_kitti/data/testing/kitti_demo_image \
            --output_dir /nfs/project/libo_i/go_kitti/infer_output/{} \
            --set INFER_OR_TEST True".format(assigned_model, exp_name)


def infer_and_combine(infer_cmd, infer_output_dir, cmp_rush_rob, gt_path, exp_name):
    subproc(infer_cmd)
    results_lists = os.listdir(infer_output_dir)

    # 循环内读取output_dir里的每张图片，将其和ground truth和top1的进行组合，生成一张对比图

    for item in results_lists:
        infer_item_path = os.path.join(infer_output_dir, item)
        rush_rob_item_path = os.path.join(cmp_rush_rob, item)
        gt_item_path = os.path.join(gt_path, item)

        print(infer_item_path)
        print(rush_rob_item_path)
        print(gt_item_path)

        infer_item = mpimg.imread(infer_item_path)
        rush_rob_item = mpimg.imread(rush_rob_item_path)
        gt_item = mpimg.imread(gt_item_path)

        fig, axes = plt.subplots(3, 1)
        axes[0].imshow(infer_item)
        axes[0].set_title("Inference")
        axes[0].axis("off")
        axes[1].imshow(rush_rob_item)
        axes[1].set_title("Top1's")
        axes[1].axis("off")
        axes[2].imshow(gt_item)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        plt.axis('off')
        plt.show()

        save_path = "/nfs/project/libo_i/go_kitti/infer_output/{}_cmp".format(exp_name)

        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        plt.savefig(os.path.join(save_path, item), dpi=300)


infer_and_combine(infer_cmd, infer_output_dir, cmp_rush_rob, gt_path, exp_name)
