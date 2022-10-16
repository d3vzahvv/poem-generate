from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):

    # Load all Data
    # Storge and init variables
    def __init__(self, data, w1, word_2_index, originalData):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.data = data
        self.originalData = originalData

    # Process Data
    def __getitem__(self, index):
        aPoetryWords = self.data[index]
        aPoetryWordsIndex = [self.word_2_index[word] for word in aPoetryWords]

        xsIndex = aPoetryWordsIndex[:-1]
        ysIndex = aPoetryWordsIndex[1:]

        xsEmbedding = self.w1[xsIndex]

        return xsEmbedding, np.array(ysIndex).astype(np.int64)

    def __len__(self):
        return len(self.originalData)