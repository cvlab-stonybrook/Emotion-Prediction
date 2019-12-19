# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 15:07
from PyUtils.pickle_utils import loadpickle
from CNNs.dataloaders.sample_loader import SampleLoader
from CNNs.dataloaders.transformations import *


def categorical_train(args):
    image_categorical_dict = loadpickle(args.train_file)
    dataset = SampleLoader(category_dict=image_categorical_dict, root=args.data_dir, transform=get_train_fix_size_transform(), target_transform=None, sample_size=args.sample_size)
    return dataset