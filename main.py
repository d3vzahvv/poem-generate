"""
Date: 8/17/2021
Author: Shuo Wu
"""

from os import read
from gensim.models import word2vec
from gensim.models.phrases import original_scorer
import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader, dataloader
import pickle
import os




def split(file = "poetry_7.txt"):
    
    data = open(file, encoding = "utf-8").read()
    dataSplit = " ".join(data)
    
    with open("split.txt", "w", encoding = "utf-8") as f:
        f.write(dataSplit)


def trainVec(originalFile = "poetry_7.txt" ,splitFile = "split.txt"):
    
    if os.path.exists(splitFile) == False:
        split()
    splitData = open(splitFile, "r", encoding = "utf-8").read().split("\n")
    originalData = open(originalFile, "r", encoding = "utf-8").read().split("\n")

    if os.path.exists("vecParams.pk1"):
        return originalData, pickle.load(open("vecParams.pk1", "rb"))
    
    model = Word2Vec(splitData, vector_size = 107, min_count = 1, workers = 6)
    
    pickle.dump((model.syn1neg, model.wv.key_to_index, model.wv.index_to_key),\
        open("vecParams.pk1", "wb"))
    
    return originalData, (model.syn1neg, model.wv.key_to_index, model.wv.index_to_key)

class MyDataset(Dataset):
    
    # Load all Data
    # Storge and init variables
    def __init__(self, data, w1, word_2_index):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.data = data

    # Process Data
    def __geitem__(self, index):
        aPoetryWords = self.data[index]
        aPoetryWordsIndex = [self.word_2_index[word] for word in aPoetryWords] 

        xsIndex = aPoetryWordsIndex[:-1]
        ysIndex = aPoetryWordsIndex[1:]
        print(" ")

    def __len__(self):
        return len(originalData)
        

if __name__ == '__main__':
    # split()
    batchSize = 32
    originalData, (w1, word_2_index, index_2_word) = trainVec()
    dataset = MyDataset(originalData, w1, word_2_index)
    dataloader = DataLoader(dataset, batch_size = batchSize, shuffle = False)
    pass
