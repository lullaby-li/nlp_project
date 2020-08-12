import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import random
plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置画中文
plt.rcParams['axes.unicode_minus'] = False

from sklearn.manifold import TSNE
from gensim.models import Word2Vec


file_path = r"./zhou/Project01/model/word2vec_v1.0.model"

model = Word2Vec.load(file_path)

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []    # list,e.g. ['sex', 'relationship', 'start', 'best', 'way']
    tokens = []    # list of array, 表示labels中每个词对应的词向量

    for word in random.sample(list(model.wv.vocab),500):    # model.wv.vocab:dict, This object essentially contains the mapping between words and embeddings. After training, it can be used directly to query those embeddings in various ways. See the module level docstring for examples.
        tokens.append(model[word])
        labels.append(word)
    
    # n_components:嵌入空间的维数，默认是2
    # init: 选择嵌入初始化的方法
    # n_iter：最大迭代次数
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)     # new_values, np.array
#     print(type(new_values))   

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
#     return labels, tokens
tsne_plot(model)