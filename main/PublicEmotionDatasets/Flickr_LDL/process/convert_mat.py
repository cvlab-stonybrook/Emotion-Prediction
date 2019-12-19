"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to create a list of (image, label) pairs for Flickr_LDL dataset,
where label can be: 1) a vector of probabilities, or 2) a single label (highest probability).
"""

import os
from PyUtils.pickle_utils import save2pickle
import tqdm
from random import choice
from scipy.io import loadmat
import numpy as np


#dataset_dir = os.path.join('D:/Eugenia/Emotion/V2_PublicEmotion', 'Flickr_LDL')


all_data = loadmat(os.path.join(dataset_dir, 'data_fli.mat'))
img_inds = all_data['ind'][0]
all_votes = loadmat(os.path.join(dataset_dir, 'nvote_fli.mat'))
votes = all_votes['n_vote_all']
categories = ['Amusement', 'Awe', 'Contentment', 'Excitement', 'Anger', 'Disgust', 'Fear', 'Sadness']
sort_idx = np.argsort(categories) # ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']


### VECTOR DISTRIBUTION LABEL ###
data_set = []
for i in range(len(img_inds)):
    s_file = img_inds[i]
    voted = np.array(votes[i])[sort_idx]
    data_set.append(['{}.jpg'.format(s_file), voted])

print(data_set)
print("total: {}".format(len(data_set)))

data_split = 'all_data'
save2pickle(os.path.join(dataset_dir, 'v_data', '{}.pkl'.format(data_split)), data_set)


"""
### SINGLE LABEL ###
data_set = []
data_counts = {}
for i in range(len(img_inds)):
    s_file = img_inds[i]
    voted = np.array(votes[i])[sort_idx]
    max_inds = [index for index, value in enumerate(voted) if value == max(voted)]
    if len(max_inds) > 1:
        continue
        s_label = int(choice(max_inds))
    else:
        s_label = int(max_inds[0])
    data_set.append(['{}.jpg'.format(s_file), s_label])
    # if s_label == 1:
    if s_label in data_counts:
        data_counts[s_label] += 1
    else:
        data_counts[s_label] = 1

#print(data_set)
print("total: {}".format(len(data_set)))
print("{}".format( ', '.join('{}\t{}'.format(x, data_counts[x]) for x in data_counts)))

data_split = 'all_data'
save2pickle(os.path.join(dataset_dir, 'z_data', '{}.pkl'.format(data_split)), data_set)

"""