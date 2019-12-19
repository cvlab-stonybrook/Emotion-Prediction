import os, sys
#project_root = os.path.join(os.path.expanduser('~'), 'Projects/MetaEmotion')
project_root = os.path.join("/nfs/bigfovea/add_disk0/eugenia/Emotion/MetaEmotion-master/")
sys.path.append(project_root)

from CNNs.utils.config import parse_config
from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
import numpy as np

def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Image CNN training from Configure Files")
    parser.add_argument('--config_file', required=True, help="This scripts only accepts parameters from Json files")
    input_args = parser.parse_args()

    config_file = input_args.config_file

    args = parse_config(config_file)
    class_lens = args.class_len


    for ind in range(len(class_lens)):
        print("-------------------------------")
        train_dataset = loadpickle(args.train_files[ind])
        add_dataset = loadpickle(args.train_files[ind].replace(".pkl", "_try.pkl"))
        print(len(train_dataset), len(add_dataset))

        file_name = args.train_files[ind]
        save2pickle(file_name.replace(".pkl", "_new.pkl"), train_dataset + add_dataset)



if __name__ == '__main__':
    main()