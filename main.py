from flask import Flask, render_template, request
from gensim.models import word2vec
from gensim.models.phrases import original_scorer
import numpy as np
from gensim.models.word2vec import Word2Vec
import torch
from torch.nn.modules.sparse import Embedding
from torch.utils.data import Dataset, DataLoader, dataloader
import pickle
import os
import torch.nn as nn

import fivespeech


# 输入http://127.0.0.1:5000/PoetryMake.html进入首页
app = Flask(__name__)

""" 首页 """
@app.route('/PoetryMake.html', methods=['GET', 'POST'])
def getdata():

    """ 发送数据给前端 """
    if request.method == 'GET':
        text = '测试'  # 前端的初始化数据
        return render_template('PoetryMake.html', text=text)


    """ 获取前端数据,并进行回传更改 """
    if request.method == 'POST':
        inputText = request.form.get("text1")
        if len(inputText) <= 3:
            result = generateAuto()
        else:
            result = ""
            punctuation_list = ["，", "。", "，", "。"]
            for i in range(4):

                h_0 = torch.tensor(np.zeros((2, 1, hiddenNum), dtype=np.float32))
                c_0 = torch.tensor(np.zeros((2, 1, hiddenNum), dtype=np.float32))
                word = inputText[i]
                try:
                    wordIndex = word_2_index[word]
                except:
                    wordIndex = np.random.randint(0, wordSize, 1)[0]
                    word = index_2_word[wordIndex]
                result += word

                for j in range(6):
                    wordIndex = word_2_index[word]
                    word_embedding = torch.tensor(w1[wordIndex][None][None])
                    pre, (h_0, c_0) = model(word_embedding, h_0, c_0)
                    word = index_2_word[int(torch.argmax(pre))]
                    result += word
                result += punctuation_list[i]
        return render_template('PoetryMake.html', text=result)


""" 关于我们页面 """
@app.route('/aboutus.html')
def goto_aboutus():
    return render_template('help.html')


""" 帮助页面 """
@app.route('/help.html')
def goto_help():
    return render_template('help.html')


def split(file="poetry_7.txt"):
    data = open(file, encoding="utf-8").read()
    dataSplit = " ".join(data)

    with open("split.txt", "w", encoding="utf-8") as f:
        f.write(dataSplit)


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


class MyDataset(Dataset):

    # Load all Data
    # Storge and init variables
    def __init__(self, data, w1, word_2_index):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.data = data

    # Process Data
    def __getitem__(self, index):
        aPoetryWords = self.data[index]
        aPoetryWordsIndex = [self.word_2_index[word] for word in aPoetryWords]

        xsIndex = aPoetryWordsIndex[:-1]
        ysIndex = aPoetryWordsIndex[1:]

        xsEmbedding = self.w1[xsIndex]

        return xsEmbedding, np.array(ysIndex).astype(np.int64)

    def __len__(self):
        return len(originalData)


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


def generateAuto():
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


if __name__ == '__main__':
    # split()
    print("\n请耐心等待古诗生成器启动哦\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelResultFile = "PoetryModelLSTMmodel.pkl"

    batchSize = 16
    originalData, (w1, word_2_index, index_2_word) = trainVec()
    dataset = MyDataset(originalData, w1, word_2_index)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
    wordSize, embeddingNum = w1.shape
    hiddenNum = 128
    lr = 0.005
    epochs = 1000

    model = MyModel(embeddingNum, hiddenNum, wordSize)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if os.path.exists(modelResultFile):
        model = pickle.load(open(modelResultFile, 'rb'))
    else:
        for e in range(epochs):
            for batchIndex, (xsEmbedding, ysIndex) in enumerate(dataloader):
                xsEmbedding = xsEmbedding.to(device)
                ysIndex = ysIndex.to(device)

                pre, _ = model(xsEmbedding)
                loss = model.crossEntropy(pre, ysIndex.reshape(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batchIndex % 30 == 0:
                    print(f"loss:{loss:.5f}")
                    generateAuto()
        pickle.dump(model, open(modelResultFile, "wb"))
    app.run(debug=True)