"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to plot cosine similarity matrix of word embeddings of emotion words.
"""

import os
import gluonnlp as nlp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity
#from glove import Corpus, Glove
import glob
import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


img_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/images/"
adobe_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/"
kw_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/keyword_lists/"
model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"


def plot_sim_matrix(sm, classes, full=False):
	'''	saves a heatmap plot of a matrix of similarities 'sm' of classes 'classes' '''
    if not full: # include annotations and class names in the plot
        plt.figure(figsize=(6, 4))
        fntsz = 8
        cls = range(len(classes))
        save_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/matrices/small"
    else: # exclude annotations and class names from the plot
        plt.figure(figsize=(7, 5))
        fntsz = 9
        cls = classes
        save_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/matrices"
    plt.xticks(fontsize=fntsz)
    plt.yticks(fontsize=fntsz)
    mask = np.zeros_like(sm)
    mask[np.triu_indices_from(mask)] = True
    with sb.axes_style("white"):
        ax = sb.heatmap(sm, mask=mask, vmin=-0.08, vmax=0.83, annot=full, annot_kws={"size": fntsz},
                        cmap="YlGnBu", xticklabels=cls, yticklabels=cls)
    plt.tight_layout()
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.savefig(os.path.join(save_folder, 'ft_w2v.png'), bbox_inches='tight')



#########  WORD2VEC MODEL  #########

# load word2vec model
model = Word2Vec.load(os.path.join(model_folder, "w2v_finetuned.model"))
#model = KeyedVectors.load_word2vec_format(os.path.join(model_folder, "finetuned_Adobe_emotions.model.wv.vectors.npy"), binary=False)
print("Model loaded")

# emotion lists
emotion_dict = {
    6: sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear']),
    8: sorted(['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'])
}

# get emotion vectors
emotion_list = sorted([emotion for emotion in set(emotion_dict[6] + emotion_dict[8])])

# compute similarity matrix
pairwise_sims = np.array([model.wv.similarity(emo0, emo1) for emo0 in emotion_list for emo1 in emotion_list]).reshape((len(emotion_list), len(emotion_list)))
plot_sim_matrix(pairwise_sims, emotion_list)
plot_sim_matrix(pairwise_sims, emotion_list, full=True)



"""

#########  DEFAULT GLOVE MODEL  #########

# load word2vec model
glove = nlp.embedding.create('glove', source='glove.6B.100d')
vocab = nlp.Vocab(nlp.data.Counter(glove.idx_to_token))
vocab.set_embedding(glove)

# emotion lists
emotion_dict = {
    6: sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear']),
    8: sorted(['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'])
}

# get emotion vectors
emotion_list = sorted([emotion for emotion in set(emotion_dict[6] + emotion_dict[8])])

# compute similarity matrix
pairwise_sims = np.array([cosine_similarity(vocab.embedding[emo0].asnumpy().reshape((1, -1)), vocab.embedding[emo1].asnumpy().reshape((1, -1)))
                          for emo0 in emotion_list for emo1 in emotion_list]).reshape((len(emotion_list), len(emotion_list)))

plot_sim_matrix(pairwise_sims, emotion_list)
plot_sim_matrix(pairwise_sims, emotion_list, full=True)
"""

"""

#########  ADOBE GLOVE MODEL  #########

#from gensim.scripts.glove2word2vec import glove2word2vec
out_file = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/glove_adobe/gensim_glove_vectors.txt"
#glove2word2vec(glove_input_file="/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/glove_adobe/vectors.txt",
#               word2vec_output_file=out_file)

# load word2vec model
from gensim.models.keyedvectors import KeyedVectors
glove = KeyedVectors.load_word2vec_format(out_file, binary=False)

# emotion lists
emotion_dict = {
    6: sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear']),
    8: sorted(['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'])
}

# get emotion vectors
emotion_list = sorted([emotion for emotion in set(emotion_dict[6] + emotion_dict[8])])

# compute similarity matrix
pairwise_sims = np.array([glove.similarity(emo0, emo1) for emo0 in emotion_list for emo1 in emotion_list]).reshape((len(emotion_list), len(emotion_list)))
plot_sim_matrix(pairwise_sims, emotion_list)
plot_sim_matrix(pairwise_sims, emotion_list, full=True)
"""