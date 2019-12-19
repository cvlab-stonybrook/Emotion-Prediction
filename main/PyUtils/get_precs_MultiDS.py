"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 07/08/2019

Usage:
Run this script to write accuracies of predictions for different 
datasets to Excel file -- from multi-dataset training methods.
"""

import os
import glob
import numpy as np
from openpyxl import *
from operator import itemgetter


def process_log(datasets, log_path):
	'''
	For a single log file, finds test accuracies 
	where validation accuracies are the highest.
	'''
    vals_precs = []
    test_precs = []

    f = open(log_path, "r")
    content = f.read().split("\n")
    for line in content:
        if "Val precisions: " in line:
            val_split = line.split("Val precisions: ")[-1][1:-1].split(', ')
            vals_precs.append(list(float(el) for el in val_split))
        elif "Test precisions: " in line:
            test_split = line.split("Test precisions: ")[-1][1:-1].split(', ')
            test_precs.append(list(float(el) for el in test_split))

    res = []
    for i in range(len(datasets.keys())):
        ind_max, val_max = max(enumerate(sub[i] for sub in vals_precs), key=itemgetter(1))
        res.append(test_precs[ind_max])
    print(res)

    return res

def write_to_xls(xlsx_file, def_datasets, datasets, precs, row_num=-1):
	'''
	Writes all dataset accuracies to the Excel file:
	each new row i contains accuracies of the best-accuracy model for dataset i
	'''
    wb = load_workbook(xlsx_file)
    ws = wb['Sheet1']

    reversed_dict = dict(map(reversed, datasets.items()))
    for x in range(len(datasets.keys())):
        for c in range(len(datasets.keys())):
            ws.cell(column=(c+1)*2, row=row_num).value = precs[reversed_dict[def_datasets[x]]][reversed_dict[def_datasets[c]]]
            wb.save(xlsx_file)
        row_num += 1


def main():
	#####
    # default (in Excel file) -  do not touch
    def_datasets = {0: 'Deepemotion', 1: 'Flickr_LDL', 2: 'Twitter_LDL', 3: 'Emotion6', 4: 'UnBiasedEmotion', 5: 'Adobe'} 
	#####

    """
    datasets = {0: 'Deepemotion', 1: 'Flickr_LDL', 2: 'Twitter_LDL', 3: 'Emotion6', 4: 'UnBiasedEmotion', 5: 'Adobe'}
    write_row = 26
    for model in ['w2v', 'glove']:
        for method in ['tag-emo', 'all_tags-emo']:
            log_path = "D:/Eugenia/Emotion/MetaEmotion-master/ckpts/Latest/%s/%s/*.txt" % (model, method)
            for file in glob.glob(log_path):
                precs = process_log(datasets, file)
            comparison_file_path = "D:/Eugenia/Emotion/MetaEmotion/checkpts/comparison.xlsx"
            write_to_xls(comparison_file_path, def_datasets, datasets, precs, row_num=write_row)
            write_row += len(datasets)
    """
    write_row = 58
    for model in ['w2v', 'glove']:
        for method in ['all_tags-emo']:
            def_datasets = {0: 'Deepemotion', 1: 'Emotion6', 2: 'UnBiasedEmotion', 3: 'Adobe'}
            datasets = {0: 'Deepemotion', 1: 'Emotion6', 2: 'UnBiasedEmotion', 3: 'Adobe'}
            log_path = "D:/Eugenia/Emotion/MetaEmotion-master/ckpts/Latest/mix4/%s/%s/*.txt" % (model, method)
            for file in glob.glob(log_path):
                precs = process_log(datasets, file)
            comparison_file_path = "D:/Eugenia/Emotion/MetaEmotion/checkpts/comparison.xlsx"
            write_to_xls(comparison_file_path, def_datasets, datasets, precs, row_num=write_row)
            write_row += len(datasets)


if __name__ == '__main__':
    main()