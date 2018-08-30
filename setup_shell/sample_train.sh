#!/usr/bin/env bash
log="sample_train.txt"
python3 /nfs/project/libo_i/go_kitti/tools/train_net_step.py \
    --dataset kitti_train \
    --cfg /nfs/project/libo_i/mask-rcnn.pytorch/configs/baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml \
    --use_tfboard --bs 4 --nw 4 --iter_size 4 \
    --load_ckpt /nfs/project/libo_i/mask-rcnn.pytorch/LB_Outputs/Best_of_best/X101_2237/ckpt/model_step_53999.pth \
    --output /nfs/project/libo_i/mask-rcnn.pytorch/LB_Outputs/TRAIN_VAL_PLUS/X101_BEST \
    --set TRAIN.SCALES [800,933,1250,1400,1800,] TRAIN.MAX_SIZE 2400 SOLVER.MAX_ITER 108000 SOLVER.STEPS [0,32000,54000,70000,80000,] > $log