# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'mapillary_train':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/train',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/training.json'
    },
    'mapillary_train_val_aug':
    {
      IM_DIR:
        _DATA_DIR + '/mapillary/images/aug_new',
      ANN_FN:
          _DATA_DIR + '/mapillary/annotations/mapillary_train_val_aug_new.json'
    },
    'apollo_train': {
        IM_DIR:
            _DATA_DIR + '/apolloscape/images',
        ANN_FN:
            _DATA_DIR + '/apolloscape/annotations/training.json'
    },
    'cityscape_train':
    {
        IM_DIR:
            _DATA_DIR + '/cityscape/images',
        ANN_FN:
            _DATA_DIR + '/cityscape/annotations/training.json'
    },
    'kitti_train_aug': {
        IM_DIR:
            _DATA_DIR + '/aug',
        ANN_FN:
            _DATA_DIR + '/annotations/training_new.json'
    },
    'kitti_train': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training.json'
    },
    'kitti_train_180_part1': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_180_Part1.json'
    },
    'kitti_train_180_part2': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_180_Part2.json'
    },
    'kitti_train_180_part3': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_180_Part3.json'
    },
    'kitti_train_180_part4': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_180_Part4.json'
    },
    'kitti_train_180_part5': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_180_Part5.json'
    },
    'coco_kitti_val': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training.json'
    },
    'coco_kitti_test': {
        IM_DIR:
            _DATA_DIR + '/testing/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/testing.json'
    },
    # 这个train的20imgs
    'coco_kitti_val_20_part1': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_20_Part1.json'
    },
    'coco_kitti_val_20_part2': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_20_Part2.json'
    },
    'coco_kitti_val_20_part3': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_20_Part3.json'
    },
    'coco_kitti_val_20_part4': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_20_Part4.json'
    },
    'coco_kitti_val_20_part5': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_20_Part5.json'
    },
    'coco_kitti_test_4imgs': {
        IM_DIR:
            _DATA_DIR + '/testing/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/testing_4imgs.json'
    },
    'mapillary_train_val':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/train_val/images',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotation/mapillary_train_val_new.json'
    },
    'coco_2017_test':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_new.json'
    },
    'coco_2017_val':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/validations',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/validation.json'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    }
}
