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
    'kitti_train_180': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/training_180.json'
    },
    'kitti_val': {
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
    'coco_kitti_val_20': {
        IM_DIR:
            _DATA_DIR + '/training/image_2',
        ANN_FN:
            _DATA_DIR + '/annotations/testing_20.json'
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
		    _DATA_DIR + '/mapillary/images/train_val',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/mapillary_train_val_new.json'
    },
    'mapillary_train_20p':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/train',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/training_20p.json'
    },
    'mapillary_train_40p':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/train',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/training_40p.json'
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
    'coco_2017_val_4pic':
    {
        IM_DIR:
            _DATA_DIR + '/mapillary/images/validations',
        ANN_FN:
            _DATA_DIR + '/mapillary/annotations/validation_4_image_only.json'
    },
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
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
    },
    'coco_2017_val_20pct': {
        IM_DIR:
            _DATA_DIR + '/mapillary/images/validations',
        ANN_FN:
            _DATA_DIR + '/mapillary/annotations/validation_20p.json',
    },
    'coco_2017_20pct_no_ann': {
        IM_DIR:
            _DATA_DIR + '/mapillary/images/validations',
        ANN_FN:
            _DATA_DIR + '/mapillary/annotations/validation_20p_new.json',
    },
    # coco_2017_10pic_no_ann
    'coco_2017_10pic_no_ann': {
        IM_DIR:
            _DATA_DIR + '/mapillary/images/validations',
        ANN_FN:
            _DATA_DIR + '/mapillary/annotations/val_10_new.json',
    },
    'coco_2017_val_20pct_new': {
        IM_DIR:
            _DATA_DIR + '/mapillary/images/validations',
        ANN_FN:
            _DATA_DIR + '/mapillary/annotations/validation_801.json',
    },
    'coco_2017_test_multi_0':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_0.json'
    },
    'coco_2017_test_multi_1':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_1.json'
    },
    'coco_2017_test_multi_2':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_2.json'
    },
    'coco_2017_test_multi_3':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_3.json'
    },
    'coco_2017_test_multi_4':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_4.json'
    },
    'coco_2017_test_multi_5':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_5.json'
    },
    'coco_2017_test_multi_6':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_6.json'
    },
    'coco_2017_test_multi_7':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_7.json'
    },
    'coco_2017_test_multi_8':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_8.json'
    },
    'coco_2017_test_multi_9':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_9.json'
    },
    'coco_2017_test_multi_10':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_10.json'
    },
    'coco_2017_test_multi_11':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_11.json'
    },
    'coco_2017_test_multi_12':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_12.json'
    },
    'coco_2017_test_multi_13':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_13.json'
    },
    'coco_2017_test_multi_14':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_14.json'
    },
    'coco_2017_test_multi_15':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_15.json'
    },
    'coco_2017_test_multi_16':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_16.json'
    },
    'coco_2017_test_multi_17':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_17.json'
    },
    'coco_2017_test_multi_18':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_18.json'
    },
    'coco_2017_test_multi_19':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_19.json'
    },
    'coco_2017_test_multi_20':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_20.json'
    },
    'coco_2017_test_multi_21':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_21.json'
    },
    'coco_2017_test_multi_22':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_22.json'
    },
    'coco_2017_test_multi_23':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_23.json'
    },
    'coco_2017_test_multi_24':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_24.json'
    },
    'coco_2017_test_multi_25':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_25.json'
    },
    'coco_2017_test_multi_26':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_26.json'
    },
    'coco_2017_test_multi_27':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_27.json'
    },
    'coco_2017_test_multi_28':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_28.json'
    },
    'coco_2017_test_multi_29':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_29.json'
    },
    'coco_2017_test_multi_30':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_30.json'
    },
    'coco_2017_test_multi_31':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_31.json'
    },
    'coco_2017_test_multi_32':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_32.json'
    },
    'coco_2017_test_multi_33':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_33.json'
    },
    'coco_2017_test_multi_34':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_34.json'
    },
    'coco_2017_test_multi_35':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_35.json'
    },
    'coco_2017_test_multi_36':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_36.json'
    },
    'coco_2017_test_multi_37':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_37.json'
    },
    'coco_2017_test_multi_38':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_38.json'
    },
    'coco_2017_test_multi_39':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_39.json'
    },
    'coco_2017_test_multi_40':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_40.json'
    },
    'coco_2017_test_multi_41':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_41.json'
    },
    'coco_2017_test_multi_42':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_42.json'
    },
    'coco_2017_test_multi_43':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_43.json'
    },
    'coco_2017_test_multi_44':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_44.json'
    },
    'coco_2017_test_multi_45':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_45.json'
    },
    'coco_2017_test_multi_46':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_46.json'
    },
    'coco_2017_test_multi_47':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_47.json'
    },
    'coco_2017_test_multi_48':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_48.json'
    },
    'coco_2017_test_multi_49':
    {
	    IM_DIR:
		    _DATA_DIR + '/mapillary/images/test',
	    ANN_FN:
	        _DATA_DIR + '/mapillary/annotations/test_49.json'
    }
}
