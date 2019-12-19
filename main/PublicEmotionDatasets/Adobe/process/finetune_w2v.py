"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to train word2vec (pre-trained on Google News) on Adobe tags.
"""

import os
import pickle
import glob
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy as np
from PyUtils.pickle_utils import loadpickle, save2pickle

adobe_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/"


model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"
# load pre-trained word2vec model
model_file = "GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_folder, model_file), binary=True)
print("Model loaded.")


#img_id_kws = loadpickle(os.path.join(adobe_folder, "img_id_kws.pkl"))
#sentences = list(img_id_kws.values())
#save2pickle(os.path.join(adobe_folder, "sentences.pkl"), sentences)
sentences = loadpickle(os.path.join(adobe_folder, "all_sentences.pkl"))

# train word2vec model
print("Preparing models")

model_2 = Word2Vec(min_count=1, size=300, workers=16, window=3, sg=1)
model_2.build_vocab(sentences)
total_examples = model_2.corpus_count
model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format(os.path.join(model_folder, model_file), binary=True, lockf=1.0)
print("Fine tuning Word2Vec model")
model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)

# save trained model
new_model_file = "w2v_finetuned.model"
model_2.save(os.path.join(model_folder, new_model_file))


print("Done.")