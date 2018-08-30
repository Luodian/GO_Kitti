#!/usr/bin/env bash
python3 /nfs/project/libo_i/mask-rcnn.pytorch/tools/test_net.py \
    --dataset kitti_test \
    --cfg /nfs/project/libo_i/mask-rcnn.pytorch/configs/baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x_map.yaml \
    --load_ckpt /nfs/project/libo_i/mask-rcnn.pytorch/LB_Outputs/Best_of_best/X101_2237/ckpt/model_step_53999.pth \
    --multi-gpu-testing \
    --method X101_2325_VAL \
    --set TEST.BBOX_AUG.SCALES [1300,1800,1900,2400,4000] TEST.MASK_AUG.SCALES [1300,1800,1900,2400,4000] TEST.SOFT_NMS.ENABLED True TEST.SOFT_NMS.SIGMA 0.9