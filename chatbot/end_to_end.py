# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:00:48 2020

@author: yihua
"""
from questionSimilarity import *
from bleu import BLEU, getAnswers_BLEU
import os
'''
主函数文件

需要预读取的文件有：

- 初始阶段
    1. qaDataset

- 句向量相似度阶段
    1. KMeans聚类结果
    2. gensim.models.word2vec
    3. 句向量矩阵

- 布尔搜索阶段
    1. CountVectorizer对象
    2. 词频矩阵
    
- 爬虫阶段
    Null
'''

'''
1. 句向量相似度 - 预加载文件
'''
qaDataset = pd.read_csv("./data/qa_corpus.csv").drop([3198, 12749, 13649, 19748, 32510])
model = loadW2vModel('./models/word2vec_v1.1.model')
#kmeansModel = loadKmeansModel('./models/kmeans_1.pickle')
for file in os.listdir('./models'):
    if file.startswith('kmeans_'):
        i = file.split('_')[1][:-7]
        locals()['kmeans_'+str(i)] = loadKmeansModel('./models/'+file)
sentsSIFVec = np.load('./models/questions_SIFVec.numpy', allow_pickle=True)

'''
2. BLEU相似度阶段 - 预加载文件
'''
questions_countVec = pickle.load(open('./models/questions_countVec.pickle', 'rb'))
ngram = pickle.load(open('./models/ngram_CountVectorizer.pickle', 'rb'))

'''
3. 爬虫阶段 - 导入写好的包
'''

pass


if __name__ == '__main__':
    query = '我要存钱'
    top10q_sif, top10a_sif, top10v_sif = getAnswers(query, model, kmeans_50, sentsSIFVec, qaDataset, distanceFunc='l2')
    top10q_bleu, top10a_bleu, top10v_bleu = getAnswers_BLEU(query, questions_countVec, ngram, qaDataset, denom='candidate')
    
    print(f'======输入问题：{query}=========')
    [print(f'⚪相似问题: {i},  ⭐SIF距离值: {k}, \n>回答: {j}') for i,j,k in zip(top10q_sif, top10a_sif, top10v_sif)];
    print('=================================')
    [print(f'⚪相似问题: {i},  🎈BLEU值: {k}, \n>回答: {j}') for i,j,k in zip(top10q_bleu, top10a_bleu, top10v_bleu)];

#    from collections import Counter
#    b = Counter(top10q_SIF+top10q_bleu)
#    b
