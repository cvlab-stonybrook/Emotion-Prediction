"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to label Adobe images using word embeddings:
label with 8 classes according to strategy 3 -- using only emotional tags,
image label is a vector of probabilities across classes.
"""

import glob
import os
from gensim.models import Word2Vec
import numpy as np
import random
from PyUtils.pickle_utils import loadpickle, save2pickle

def normalize(v):
	"""Normalize vector v, so that it sums to 1."""
    norm = sum(v)
    if norm == 0:
       return v
    return v / norm

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(10*x) / (np.sum(np.exp(10*x)) + 1e-10)

"""
kw_folder = "/nfs/bigfovea/add_disk0/zijunwei/Adobe/"
em_words = set()
for file_path in glob.glob(kw_folder + "*.json"):
    em_word = os.path.basename(file_path)[:-5]
    em_words.add(em_word)
save2pickle('em_words.pkl', em_words)
"""
em_words = loadpickle('/nfs/bigfovea/add_disk0/zijunwei/Adobe/em_words.pkl')

img_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/images-256/"
adobe_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/"

# emotion lists
emotion_dict = {
    6: sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear']),
    8: sorted(['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'])
}

#########   WORD2VEC   #########

dataset_dir = "/nfs/bigfovea/add_disk0/zijunwei/V2_PublicEmotion/Adobe/v_data/w2v/"

# load word2vec model
print("Adobe Word2Vec loading")
model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"
model_file = "w2v_adobe.model"
model = Word2Vec.load(os.path.join(model_folder, model_file))

for num in [8]:
    data_set = []
    k = 0
    for img_path in glob.glob(img_folder + "*.jpg"):
        k += 1
        if k % 1000 == 0:
            print(k // 1000, "K images processed")
            # save the data_set
            save2pickle(os.path.join(dataset_dir, 'all_data.pkl'), data_set)

        img_name = os.path.basename(img_path)
        txt_path = os.path.join(adobe_folder, 'keyword_lists', str(img_name[:-4]) + '.txt')
        label = None
        if os.path.exists(txt_path):
            with open(txt_path, 'rb') as fp:
                tags = pkl.load(fp)

            # compute pairwise similarity
            sims = []
            for tag in tags:
                if tag in em_words:
                    for emo in emotion_dict[num]:
                        try:
                            sim = model.wv.similarity(tag, emo)
                            sims.append(sim)
                        except KeyError:
                            sim = None
            pairwise_sims = np.array(sims).reshape((-1, len(emotion_dict[num])))
            dist_matrix = np.sqrt(np.sum(np.square(pairwise_sims), axis=0)).flatten()
            label = np.array(softmax(dist_matrix))
        if label is None:
            continue
        else:
            data_set.append((img_name, label))

    # save the data_set
    save2pickle(os.path.join(dataset_dir, 'all_data.pkl'), data_set)
    print("total: {}".format(len(data_set)))


"""

#########   Adobe  GLOVE   #########

dataset_dir = "/nfs/bigfovea/add_disk0/zijunwei/V2_PublicEmotion/Adobe/v_data/glove/"

# load glove model
print("Adobe glove loading")
model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"
model_file = "glove_adobe/gensim_glove_vectors.txt"

from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format(os.path.join(model_folder, model_file), binary=False)


for num in [8]:
    data_set = []
    k = 0
    for img_path in glob.glob(img_folder + "*.jpg"):
        k += 1
        if k % 1000 == 0:
            print(k // 1000, "K images processed")
            # save the data_set
            save2pickle(os.path.join(dataset_dir, 'all_data.pkl'), data_set)

        img_name = os.path.basename(img_path)
        txt_path = os.path.join(adobe_folder, 'keyword_lists', str(img_name[:-4]) + '.txt')
        label = None
        if os.path.exists(txt_path):
            with open(txt_path, 'rb') as fp:
                tags = pkl.load(fp)

            # compute pairwise similarity
            sims = []
            for tag in tags:
                if tag in em_words:
                    for emo in emotion_dict[num]:
                        try:
                            sim = model.similarity(tag, emo)
                            sims.append(sim)
                        except KeyError:
                            sim = None
            pairwise_sims = np.array(sims).reshape((-1, len(emotion_dict[num])))
            dist_matrix = np.sqrt(np.sum(np.square(pairwise_sims), axis=0)).flatten()
            label = np.array(softmax(dist_matrix))
        if label is None:
            continue
        else:
            data_set.append((img_name, label))

    # save the data_set
    save2pickle(os.path.join(dataset_dir, 'all_data.pkl'), data_set)
    print("total: {}".format(len(data_set)))

