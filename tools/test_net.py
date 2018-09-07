"""Perform inference on one or more datasets."""

import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, '/nfs/project/libo_i/go_kitti/lib')
import utils.logging
from core.config import assert_and_infer_cfg, cfg, merge_cfg_from_file, merge_cfg_from_list
from core.test_engine import run_inference

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--load_json',
        help='load_json', action='store_true')

    parser.add_argument(
        '--load_json_path',
        help='ensamble_json_path')

    parser.add_argument(
        '--save_json',
        help='load_json', action='store_true')

    parser.add_argument(
        '--save_json_path',
        help='load_json')

    parser.add_argument(
        '--method',
        help='method')

    parser.add_argument(
        '--infer_submission', help='whether if we only infer test', action='store_true')

    parser.add_argument(
        '--method', dest='method_name', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    print(torch.cuda.device_count())
    # assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017_20pct":
        cfg.TEST.DATASETS = ('coco_2017_val_20pct',)
        cfg.MODEL.NUM_CLASSES = 38
    elif args.dataset == "coco_kitti_test_4imgs":
        cfg.TEST.DATASETS = ('coco_kitti_test_4imgs',)
        cfg.MODEL.NUM_CLASSES = 11
    elif args.dataset == "coco_2017_test":
        cfg.TEST.DATASETS = ('coco_2017_test',)
        cfg.MODEL.NUM_CLASSES = 38
    elif args.dataset == "coco_kitti_val":
        cfg.TEST.DATASETS = ('coco_kitti_val',)
        cfg.MODEL.NUM_CLASSES = 11
    elif args.dataset == "coco_kitti_test":
        cfg.TEST.DATASETS = ('coco_kitti_test',)
        cfg.MODEL.NUM_CLASSES = 11
    elif args.dataset == "coco_kitti_val_20":
        cfg.TEST.DATASETS = ('coco_kitti_val_20',)
        cfg.MODEL.NUM_CLASSES = 11
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)
