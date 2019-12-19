"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to label Adobe images using word embeddings:
label with 6 or 8 classes ('num' parameter) according to strategies 1 or 2 ('method').
"""

import glob
import os
from PyUtils.pickle_utils import loadpickle, save2pickle
import pickle
from gensim.models import Word2Vec
import numpy as np
import random


img_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/images-256/"
adobe_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/"
dataset_dir = "/nfs/bigfovea/add_disk0/zijunwei/V2_PublicEmotion/Adobe/z_data/"

# emotion lists
emotion_dict = {
    6: sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear']),
    8: sorted(['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'])
}


# load word2vec model
print("Adobe Word2Vec loading")
model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"
model_file = "w2v_adobe.model"
model = Word2Vec.load(os.path.join(model_folder, model_file))


for num in [6, 8]:
    for method in ['tag-emo', 'all_tags-emo']:
        print("--> Baselines %s %s-class:" % (method, num))
        data_set = []
        data_counts = {}
        k = 0
        for img_path in glob.glob(img_folder + "*.jpg"):
            k += 1
            if k % 1000 == 0:
                print(k // 1000, "K images processed")
                # save the data_set
                save2pickle(os.path.join(dataset_dir, '%s_%s.pkl' % (method, num)), data_set)

            img_name = os.path.basename(img_path)
            txt_path = os.path.join(adobe_folder, 'keyword_lists', str(img_name[:-4]) + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'rb') as fp:
                    tags = pickle.load(fp)

                # compute pairwise similarity
                if method == 'tag-emo':
                    # best match of a single tag with a single emotion
                    match = -float('inf')
                    cand_l = None
                    for it, tag in enumerate(tags):
                        for ie, emo in enumerate(emotion_dict[num]):
                            sim = model.wv.similarity(tag, emo)
                            if sim > match:
                                match = sim
                                cand_l = ie
                            elif sim == match:
                                flip = random.randint(0, 1)
                                if flip:
                                    match = sim
                                    cand_l = ie
                    label = cand_l
                elif method == 'all_tags-emo':
                    # best match of a vector of tags with a single emotion
                    sims = []
                    for it, tag in enumerate(tags):
                        for ie, emo in enumerate(emotion_dict[num]):
                            sim = model.wv.similarity(tag, emo)
                            sims.append(sim)
                    pairwise_sims = np.array(sims).reshape((len(tags), len(emotion_dict[num])))
                    dist_matrix = np.sqrt(np.sum(np.square(pairwise_sims), axis=1)).flatten()
                    label = np.argmax(dist_matrix)
                if label in data_counts:
                    data_counts[label] += 1
                else:
                    data_counts[label] = 1

                data_set.append((img_name, label))
            else:
                print("nope")

        print("total: {}".format(len(data_set)))
        print(data_counts)

