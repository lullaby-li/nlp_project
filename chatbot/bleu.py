# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:01:27 2020

@author: yihua
"""

import pandas as pd
import numpy as np
import jieba
from SIF import SIF, wash_and_split
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack, csr_matrix


'''
0. 读取并预处理语料，
        保存CountVectorizor对象 和 词频矩阵
'''

#df = pd.read_csv('./data/qa_corpus.csv').drop([3198, 12749, 13649, 19748, 32510])
#
#questions = list(df.iloc[:,1])
#answers = list(df.iloc[:,2])

def build_ngram_and_fit():
    '''
    用CountVectorizer处理所有qa库中的问题，
    并且保存CountVectorizer对象和transform过后的计数矩阵
    '''
    #uni_gram = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(1,1))
    #bi_gram = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(2,2))
    #tri_gram = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(3,3))
    #quad_gram = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(4,4))
    ngram = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(1,4))
    questions_countVec = ngram.fit_transform([' '.join(jieba.cut(q)) for q in questions])
    
    pickle.dump(questions_countVec, open('./models/questions_countVec.pickle', 'wb'))
    pickle.dump(ngram, open('./models/ngram_CountVectorizer.pickle', 'wb'))

#build_ngram_and_fit()

#tfidf = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(1,4))
'''
1. 最终目的：写出一个BLEU(candidate, reference)函数，计算两者的相似度
'''

def BLEU(candidate, references, vectorizer, denom='candidate') -> [float]:
    '''
    计算candidate与references的BLEU值
    输入：
        candidate: array of shape (1, vocab_size) 
        references: array of shape (n, vocab_size)
        
    输出：
        Top 10, LIST of (index, BLEU score), for those candidates w.r.t reference
        
    '''
    candidate = vectorizer.transform([' '.join(jieba.cut(candidate))])
    vocab_size = references.shape[1]
    n = references.shape[0]
    references_length = references.astype(bool).sum(axis=1)
    candidate_length = candidate.count_nonzero()
    
    numerator = references.minimum(vstack([candidate]*n)).sum(axis=1) # Pn 的分子
    if denom == 'candidate':
        denominator = candidate.sum() # Pn 的分母
    elif denom == 'reference':
        denominator = references.sum(axis=1)+0.0001 # Pn 的分母，另一种计算方式
    else:
        raise Exception('BLEU denom ERROR')
    pn = np.divide(numerator, denominator)
#    BP = np.minimum(np.exp(1-references_length/candidate_length), 1)
#    
#    bleu = np.multiply(BP, np.exp(np.log(pn)))

    return sorted(enumerate(pn), key = lambda x: x[1], reverse=True)[:10]

def getAnswers_BLEU(question:str, qa_questions_countVec, countVectorizer, qaDataset, denom='candidate'):
    '''
    输入：
        问题
        QA库中所有问题的CountVectorizer矩阵
        CountVectorizer对象（已经fit过后的）
        qaDataset
    输出：
        通过BLEU计算后相似度排名前10的 问题 与 回答 
    '''
    
    result_top10 = BLEU(question, qa_questions_countVec, vectorizer=countVectorizer, denom=denom)
    top10q = list(qaDataset.iloc[[i[0] for i in result_top10]]['question'])
    top10a = list(qaDataset.iloc[[i[0] for i in result_top10]]['answer'])
    top10bleu_value = [i[1] for i in result_top10]
    return top10q, top10a, top10bleu_value
    
    
    
if __name__ == '__main__':
    qaDataset = pd.read_csv("./data/qa_corpus.csv").drop([3198, 12749, 13649, 19748, 32510])
    
    questions_countVec = pickle.load(open('./models/questions_countVec.pickle', 'rb'))
    ngram = pickle.load(open('./models/ngram_CountVectorizer.pickle', 'rb'))
#    BLEU("我要存款", questions_vec, vectorizer=ngram)
    top10q, top10a, top10v = getAnswers_BLEU('我要存款', questions_countVec, ngram, qaDataset)
    [print(i) for i in top10q];
