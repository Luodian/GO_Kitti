#!/usr/bin/env bash
log="sample_train.txt"
python3 /nfs/project/libo_i/go_kitti/tools/train_net_step.py \
    --dataset kitti_train_180 \
    --cfg /nfs/project/libo_i/go_kitti/configs/baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x_map.yaml \
    --use_tfboard --bs 8 --nw 4 --iter_size 4 \
    --load_ckpt /nfs/project/libo_i/mask-rcnn.pytorch/LB_Outputs/TRAIN_VAL_PLUS/101X_MULTI_BIG/ckpt/model_step34999.pth \
    --output /nfs/project/libo_i/go_kitti/train_output/baseline \
    --set TRAIN.SCALES [800,933,1250,1400,1800,] TRAIN.MAX_SIZE 2400 SOLVER.MAX_ITER 108000 SOLVER.STEPS [0,32000,54000,70000,80000,] M_ANCHOR True> $log