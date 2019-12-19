"""
Copyright (c) 2019 Yevheniia Soroka
Licensed under the MIT License
Author: Yevheniia Soroka
Email: ysoroka@cs.stonybrook.edu
Last modified: 18/12/2019

Usage:
Run this script to train word2vec model on Adobe tags.
"""

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob
import os
import numpy as np
import json
from gensim.models import Word2Vec


adobe_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/Adobe/"
kw_folder = "/nfs/bigfovea/add_disk0/zijunwei/Adobe/keyword_retrievals/keywords/"


def main():
    """
    img_id_kws = {}
    print("Reading keyword files")
    for file in glob.glob(kw_folder + "*.json"):
        print(file)
        with open(file, 'r') as of_:
            # keyword file
            lines = of_.readlines()
            for l in lines:
                # image
                d = json.loads(l)
                if d['cid'] not in img_id_kws:
                    tags = []
                    for t in d['tags']:
                        words = t.split('^')[0].split()
                        for w in words:
                            tags.append(w)
                    img_id_kws[d['cid']] = tags
    save2pickle(os.path.join(adobe_folder, "img_id_kws.pkl"), img_id_kws)
    """

    img_id_kws = loadpickle(os.path.join(adobe_folder, "img_id_kws.pkl"))
    sentences = list(img_id_kws.values())
    #save2pickle(os.path.join(adobe_folder, "sentences.pkl"), sentences)
    # train word2vec model
    model_folder = "/nfs/bigfovea/add_disk0/eugenia/Emotion/wordembedding_models/"
    model_file = "w2v_adobe.model"
	
    print("Training Word2Vec model")
        model = Word2Vec(sentences, min_count=1, size= 50, workers=16, window=3, sg=1)
    model.save(os.path.join(model_folder, model_file))
	


if __name__ == '__main__':
    main()