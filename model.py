import torch.nn as nn
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyModel(nn.Module):

    def __init__(self, embeddingNum, hiddenNum, wordSize):
        super().__init__()

        self.embeddingNum = embeddingNum
        self.hiddenNum = hiddenNum
        self.wordSize = wordSize

        self.lstm = nn.LSTM(input_size=embeddingNum, hidden_size=hiddenNum, batch_first=True, \
                            num_layers=2, bidirectional=False)
        self.dropout = nn.Dropout(0.3)  # dropout, to generate random poetry
        self.flatten = nn.Flatten(0, 1)
        self.linear = nn.Linear(hiddenNum, wordSize)
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, xsEmbedding, h_0=None, c_0=None):
        xsEmbedding = xsEmbedding.to(device)

        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2, xsEmbedding.shape[0], self.hiddenNum), np.float32))
            c_0 = torch.tensor(np.zeros((2, xsEmbedding.shape[0], self.hiddenNum), np.float32))

        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        hidden, (h_0, c_0) = self.lstm(xsEmbedding, (h_0, c_0))  # h means h_0, c means c_0
        hiddenDrop = self.dropout(hidden)
        flattenHidden = self.flatten(hiddenDrop)

        pre = self.linear(flattenHidden)
        return pre, (h_0, c_0)