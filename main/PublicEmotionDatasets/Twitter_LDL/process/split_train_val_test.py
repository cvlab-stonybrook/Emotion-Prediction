"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to split the list of (image, label) pairs stored in all_data.pkl into:
- test data - 20%
- validation - 8%
- training data - 72%
"""

from PyUtils.pickle_utils import loadpickle, save2pickle
import random
import os
random.seed(0)

data_folder = 'D:/Eugenia/Emotion/V2_PublicEmotion/SentiLDL/Twitter_LDL/z_data/'
all_data = loadpickle(os.path.join(data_folder, 'all_data.pkl'))

split = len(all_data) // 20
random.shuffle(all_data)
test_data = all_data[:split]
train_data = all_data[split:]

split1 = len(train_data) // 10
val_data = train_data[:split1]
train_data = train_data[split1:]

save2pickle(os.path.join(data_folder, 'train_8_90.pkl'), train_data)
save2pickle(os.path.join(data_folder, 'val_8_10.pkl'), val_data)
save2pickle(os.path.join(data_folder, 'test_8_20.pkl'), test_data)
print("DB")