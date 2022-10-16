"""
Author: Shuo / Huaizhe / Jiachun / Xinyi
at Beijing Forestry University, Haidian, Beijing, China's mainland
2021, Summer Project
GPL-3.0 https://www.gnu.org/licenses/gpl-3.0.html
"""

import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, dataloader
from flask import Flask, render_template, request

from model import MyModel
from dataset import MyDataset
from vectorization import trainVec
from generate import generateAuto

app = Flask(__name__)

""" Homepage """
@app.route('/', methods=['GET', 'POST'])
def getdata():

    if request.method == 'GET':
        text = 'Where Poems Build'  # Initial
        return render_template('index.html', text=text)


    """ Post Key Words to Model """
    if request.method == 'POST':
        inputText = request.form.get("text1")
        if len(inputText) <= 3:
            result = generateAuto(w1, wordSize, index_2_word, model, hiddenNum)
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
        return render_template('index.html', text=result)


""" About us page """
@app.route('/about')
def goto_aboutus():
    return render_template('aboutus.html')


""" Help page """
@app.route('/help')
def goto_help():
    return render_template('help.html')



if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelResultFile = "model.pkl"

    batchSize = 16
    originalData, (w1, word_2_index, index_2_word) = trainVec()
    dataset = MyDataset(originalData, w1, word_2_index, originalData)
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
                    generateAuto(w1, wordSize, index_2_word, model, hiddenNum)
        pickle.dump(model, open(modelResultFile, "wb"))

    app.run(host = '0.0.0.0')