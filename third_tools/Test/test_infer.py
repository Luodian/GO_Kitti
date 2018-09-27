import subprocess
import os
import cv2
import matplotlib

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


assigned_model = "/nfs/project/libo_i/go_kitti/train_output/full_101x_multi/ckpt/model_step4999.pth"
exp_name = "map_full_demo_4999"

infer_output_dir = "/nfs/project/libo_i/go_kitti/infer_output/{}".format(exp_name)
cmp_rush_rob = "/nfs/project/libo_i/go_kitti/data/testing/rush_rob_results"
gt_path = "/nfs/project/libo_i/go_kitti/data/testing/kitti_demo_image"

test_cmd = "--dataset coco_kitti_val_20_part1 \
            --cfg /nfs/project/libo_i/go_kitti/configs/baselines/kitti_final.yaml \
            --load_ckpt {} \
            --multi-gpu-testing \
            --output_dir /nfs/project/libo_i/go_kitti/test_output/ide_test \
            --set INFER_OR_TEST True M_ANCHOR True".format(assigned_model)

infer_cmd = "python3 /nfs/project/libo_i/go_kitti/tools/infer_simple.py \
            --dataset kitti \
            --cfg /nfs/project/libo_i/go_kitti/configs/baselines/kitti_final.yaml \
            --load_ckpt {} \
            --image_dir /nfs/project/libo_i/go_kitti/data/testing/kitti_demo_image \
            --output_dir /nfs/project/libo_i/go_kitti/infer_output/{} \
            --set INFER_OR_TEST True M_ANCHOR True".format(assigned_model, exp_name)

def infer_and_combine(infer_cmd, infer_output_dir, cmp_rush_rob, gt_path, exp_name):
    subproc(infer_cmd)
    results_lists = os.listdir(infer_output_dir)

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

        plt.savefig(os.path.join(save_path, item), dpi = 300)


infer_and_combine(infer_cmd, infer_output_dir, cmp_rush_rob, gt_path, exp_name)
