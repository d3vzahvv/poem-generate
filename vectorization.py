import os
import pickle
from gensim.models.word2vec import Word2Vec

from preprocess import split


def trainVec(originalFile="poetry_7.txt", splitFile="split.txt"):
    if os.path.exists(splitFile) == False:
        split()
    splitData = open(splitFile, "r", encoding="utf-8").read().split("\n")[:300]
    originalData = open(originalFile, "r", encoding="utf-8").read().split("\n")[:300]

    if os.path.exists("vecParams.pk1"):
        return originalData, pickle.load(open("vecParams.pk1", "rb"))

    model = Word2Vec(splitData, vector_size=107, min_count=1, workers=6)

    pickle.dump((model.syn1neg, model.wv.key_to_index, model.wv.index_to_key), \
                open("vecParams.pk1", "wb"))

    return originalData, (model.syn1neg, model.wv.key_to_index, model.wv.index_to_key)