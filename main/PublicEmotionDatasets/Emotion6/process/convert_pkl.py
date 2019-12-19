"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to create a list of (image, label) pairs for Emotion6 dataset.
"""

import os
from PyUtils.pickle_utils import save2pickle
import tqdm
import glob
import numpy as np

folder = 'D:/Eugenia/Emotion/V2_PublicEmotion/Emotion6/'
data_split = 'all_data'

# get labels dictionary
labels = {}
for file in glob.glob(folder + "images/**/*.jpg", recursive=True):
    labels[file.split(folder + "images\\")[1]] = (file.split(folder + "images\\")[1]).split("\\")[0]

categories = sorted(['anger', 'fear', 'love', 'joy', 'sadness', 'surprise'])

data_set = []
data_counts = {}
for s_file in labels.keys():
    s_label = categories.index(labels[s_file])
    data_set.append([s_file, s_label])
    # if s_label == 1:
    if s_label in data_counts:
        data_counts[s_label] += 1
    else:
        data_counts[s_label] = 1

print("total: {}".format(len(data_set)))
print("{}".format( ', '.join('{}\t{}'.format(x, data_counts[x]) for x in data_counts)))

save2pickle(os.path.join(folder, 'z_data', '{}.pkl'.format(data_split)), data_set)