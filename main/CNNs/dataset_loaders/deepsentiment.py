# Copyright (c) 2018 Zijun Wei, Yevheniia Soroka
# Licensed under the MIT License.
# Authors: Zijun Wei, Yevheniia Soroka 
# Email: hzwzijun@gmail.com, ysoroka@cs.stonybrook.edu
# Created: 24/Oct/2018 21:47


import os
import torch
import torch.utils.data
from CNNs.dataloaders.basic_loader import ImageRelLists
from CNNs.dataloaders.transformations import *
from PyUtils.pickle_utils import loadpickle
import numpy as np




# Single-class loaders
def deepsentiment_s_train(args):
    image_information = loadpickle(args.train_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_fix_size_transform(), target_transform=None)
    return dataset

def deepsentiment_s_val(args):
    image_information = loadpickle(args.val_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_val_simple_transform(), target_transform=None)
    return dataset

def deepsentiment_s_test(args):
    image_information = loadpickle(args.test_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_val_simple_transform(), target_transform=None)
    return dataset



# Multi-dataset loaders
n_samples = 3200
def deepsentiment_m_train(args):
    image_information = loadpickle(args.train_files[args.ind])
    dataset = ImageRelLists(image_paths=image_information,#[:n_samples],
                            image_root=args.data_dirs[args.ind],
                            transform=get_train_fix_size_transform(),
                            target_transform=None)
    return dataset

def deepsentiment_m_val(args):
    image_information = loadpickle(args.val_files[args.ind])
    dataset = ImageRelLists(image_paths=image_information,#[:n_samples],
                            image_root=args.data_dirs[args.ind],
                            transform=get_val_simple_transform(),
                            target_transform=None)
    return dataset

def deepsentiment_m_test(args):
    image_information = loadpickle(args.test_files[args.ind])
    dataset = ImageRelLists(image_paths=image_information,#[:n_samples],
                            image_root=args.data_dirs[args.ind],
                            transform=get_val_simple_transform(),
                            target_transform=None)
    return dataset





if __name__ == '__main__':
    # x_transform = multilabel2multihot(500)
    # x = x_transform([4, 10])
    # print("DEB")
    from argparse import Namespace
    from CNNs.dataloaders.utils import none_collate

    args = Namespace(num_classes=742)
    args.train_file = 'D:/Eugenia/Emotion/V2_PublicEmotion/Deepsentiment/z_data/train_3_10_list.pkl'
    args.data_dir = '/D:/Eugenia/Emotion/V2_PublicEmotion/Deepsentiment/images-256'
    dataset = deepsentiment_s_train(args)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=10, shuffle=False,
                                             num_workers=4, pin_memory=True, collate_fn=none_collate)
    import tqdm

    for s_images, s_labels in tqdm.tqdm(val_loader):
        pass

    print("Done")