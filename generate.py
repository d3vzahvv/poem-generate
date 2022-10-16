import numpy as np
import torch
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import pickle
import os


def generateAuto(w1, wordSize, index_2_word, model, hiddenNum):
    result = ""
    wordIndex = np.random.randint(0, wordSize, 1)[0]
    result += index_2_word[wordIndex]

    h_0 = torch.tensor(np.zeros((2, 1, hiddenNum), np.float32))
    c_0 = torch.tensor(np.zeros((2, 1, hiddenNum), np.float32))

    for i in range(31):
        # wordEmbedding = torch.tensor(w1[wordIndex].reshape(1,1,-1))
        wordEmbedding = torch.tensor(w1[wordIndex][None][None])
        pre, (h_0, c_0) = model(wordEmbedding, h_0, c_0)
        wordIndex = int(torch.argmax(pre))
        result += index_2_word[wordIndex]
    return result