"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to train GloVe model on Adobe tags.
"""

import os
import gluonnlp as nlp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from glove import Corpus, Glove
import glob
import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"

#Creating a corpus object
corpus = Corpus()
#Training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(lines, window=10)

# train the model
glove = Glove(no_components=5, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# save the model
glove.save(os.path.join(model_folder, 'glove_adobe.model'))