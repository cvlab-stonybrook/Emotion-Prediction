"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to plot cosine similarity matrix of word embeddings of emotion words.
"""

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob
import os
import numpy as np
import json
import gensim
import matplotlib.pyplot as plt
import seaborn as sb
from bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity


def plot_sim_matrix(sm, classes):
	'''	saves a heatmap plot of a matrix of similarities 'sm' of classes 'classes' '''
    plt.figure(figsize=(10, 8))
    plt.title('Similarity Matrix', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    mask = np.zeros_like(sm)
    mask[np.triu_indices_from(mask)] = True
    with sb.axes_style("white"):
        ax = sb.heatmap(sm, mask=mask, vmin=0., vmax=1., annot=True, cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
    plt.tight_layout()
    plt.savefig('similarities.png', bbox_inches='tight')


def main():
    # emotion lists
    emotion_dict = {
        6: sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear']),
        8: sorted(['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'])
    }

    # load word2vec model
    print("word2vec loading")
	src_dir = "../"
    model_folder = os.path.join(src_dir, "wordembedding_models")
    model_file = "w2v_adobe.model"
    model = Word2Vec.load(os.path.join(model_folder, model_file))

    # get emotion vectors
    emotion_list = sorted([emotion for emotion in set(emotion_dict[6] + emotion_dict[8])])

    # compute similarity matrix
    pairwise_sims = np.array([cosine_similarity(bert([emo0])[0][-1][0].reshape(1, -1), bert([emo1])[0][-1][0].reshape(1, -1))
                              for emo0 in emotion_list for emo1 in emotion_list]).reshape((len(emotion_list), len(emotion_list)))
    plot_sim_matrix(pairwise_sims, emotion_list)



if __name__ == '__main__':
    main()