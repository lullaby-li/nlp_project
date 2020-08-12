# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:01:27 2020

@author: yihua
"""

import pandas as pd
import numpy as np
from gensim.models import word2vec
from SIF import SIF, wash_and_split
from sklearn.cluster import KMeans
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''
第1步：对问题库中的问题做聚类，找出并保存每个kernel的中心点

聚类效果评价方法：待定
'''

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_path = './models/word2vec_v1.1.model'

model = word2vec.Word2Vec.load(model_path)

df = pd.read_csv('./data/qa_corpus.csv').drop([3198, 12749, 13649, 19748, 32510])

sents = list(df.iloc[:,1])

def preprocessing(sentence, model) -> np.array:
    '''
    句子预处理函数
    输入：
        一句句子: string
        word2vec模型: gensim.models.word2vec.Word2Vec
    输出：
        句向量: np.array - with shape 300
    '''
    splitted = wash_and_split(sentence)
    splitted = [j for i in splitted for j in i]
    return SIF([splitted], model).reshape(200)

# 预处理
# 把sents中的每一句话变成向量

#i = 0
#def map_do(x):
#    global i
#    print(f'{i}')
#    try:
#        i+=1
#        return preprocessing(x, model)
#    except Exception as e:
#        print(str(e))
#        i+=1
#    
#out = list(map(map_do, sents))
#out = np.array(out)
#pickle.dump(out, open('out.numpy', 'wb'))
questions_SIFVec = pickle.load(open('./models/questions_SIFVec.numpy', 'rb'))

def clustering(sentence_vectors, n, method='kmeans') -> KMeans:
    '''
    聚类函数，
    输入：
        句向量列表: list
        聚类类别数n: int
        聚类方法：string
    输出：
        聚类的中心点，以及它们所包含的句子，dict(center: [(center, sentences)])
    '''
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n, random_state=0).fit(sentence_vectors)
        return kmeans
    else:
        raise Exception("Method Error")

kmeans = clustering(questions_SIFVec, n=3)

kmeans.labels_

#kmeans.predict([[0, 0], [12, 3]])

kmeans.cluster_centers_

pickle.dump(kmeans, open('./models/kmeans.pickle', 'wb'))

for i in range(1,11):
    pickle.dump(clustering(questions_SIFVec, n=i), open('./models/kmeans_'+str(i)+'.pickle', 'wb'))
    print(i)

for i in [20, 50, 100, 200, 300, 500, 1000]:
    pickle.dump(clustering(questions_SIFVec, n=i), open('./models/kmeans_'+str(i)+'.pickle', 'wb'))
    print(i)

# 输入一句话, 计算它与所有kernel之间的距离
def calculate_distances_to_each_kernel(input_vector, cluster_model=kmeans):
    '''
    输入一个向量，计算这个向量与所有kernel的距离
    '''
    return cluster_model.cluster_centers_.dot(input_vector.transpose())


calculate_distances_to_each_kernel(preprocessing("我要办业务。", model), cluster_model=kmeans)


if __name__ == '__main__':
    pass
