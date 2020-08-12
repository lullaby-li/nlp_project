import numpy as np
import pandas as pd

from gensim.models import word2vec
from SIF import SIF, wash_and_split
from sklearn.cluster import KMeans
import pickle

def loadKmeansModel(modelPath):
    with open(modelPath,'rb') as fr:
        kmeans = pickle.load(fr)
    return kmeans

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

def loadW2vModel(modelPath):
    return word2vec.Word2Vec.load(modelPath)


def getSimilarity(x, y, funcName="l2"):
    """
    计算两个array的相似度
        注意：cosine similarity是越大越相似（值接近1，夹角接近0）
            其余都是越小越相似（值接近0，距离接近0）
    """
    if funcName=="cosine":
        return (np.dot(x.reshape(-1), y.reshape(-1))/(np.linalg.norm(x.reshape(-1)) * np.linalg.norm(y.reshape(-1))) + 1)/2
    elif funcName=="l1":
        return np.linalg.norm(x-y, ord=1)
    elif funcName=="l2":
        return np.linalg.norm(x-y, ord=2)
    else:
        return np.linalg.norm(x-y, ord=np.inf)

def sortSimilarty(sentsMat, qVec, funcName='l2'):
    """
    Docstring:
        计算sent对qVec的每个行向量的相似度
    Param:
        sentsMat:DataFame，某一类问题的向量矩阵
        qVec:要查询问题的句向量
    Return:
        返回一个list，元素1是句子的索引，元素2是相似度值
    Example:
        sortSimilarty(queryMat, sentVec)
    """
    sortDict = {} 
    for i in range(len(sentsMat)):
        tmpVec = sentsMat[i]
        similarity = getSimilarity(qVec, tmpVec, funcName=funcName)
        sortDict[i] = similarity
    return sorted(sortDict.items(), key=lambda item:item[1])

def sortSimilarty_vectorized(emu_sentsMat, qVec, funcName='l2'):
    """
    Docstring:
        计算sent对qVec的每个行向量的相似度，向量化版本
    Param:
        sentsMat:DataFame，某一类问题的向量矩阵
        qVec:要查询问题的句向量
    Return:
        返回一个list，元素1是句子的索引，元素2是相似度值
    Example:
        sortSimilarty(queryMat, sentVec)
    """
    indices = np.array([i[0] for i in emu_sentsMat])
    sentsMat = np.array([i[1] for i in emu_sentsMat])

    if funcName=='l1':
        distances = np.linalg.norm(sentsMat-qVec, ord=1, axis=1)
        return sorted(zip(indices,distances), key=lambda x: x[1])    
    elif funcName=='l2':
        distances = np.linalg.norm(sentsMat-qVec, ord=2, axis=1)
        return sorted(zip(indices,distances), key=lambda x: x[1])
    elif funcName=='linf':
        distances = np.linalg.norm(sentsMat-qVec, ord=np.inf, axis=1)
        return sorted(zip(indices,distances), key=lambda x: x[1])
    elif funcName=='cosine':
        cosine_distances = sentsMat.dot(qVec.T)
        return sorted(zip(indices, cosine_distances), key=lambda item:item[1], reverse=True)
    else:
        raise Exception("Distance Function ERROR")

def getAnswers(question:str, w2vModel:word2vec, kmeansModel, sentsVec, qaDataset, distanceFunc='l2'):
    """
    Param:
        question:str,用户输入查询的问题
        w2vModelPath: word2vec词向量模型
        sentsMatPath： 训练的句向量文件路径
        qaDatasetPath：qa数据集路径
        kmeansModelPath:聚类模型所在路径
    Return:
        questionTextList: list,相似度最高的前10个问题
        answerTextList: list,相似度最高的前10个问题的回答
    """
    # 1. 载入训练的句向量文件,和数据集文文件
#    queryText = pd.read_csv("./data/qa_corpus.csv").drop([3198, 12749, 13649, 19748, 32510])
    # 更新：第一步的所有文件都改成预加载
    
    # 2. 将question问题转化为句向量sentVec，并预测所属类别y_pred
    model = w2vModel
    queryVec = preprocessing(question, model).reshape(1,200)
    y_pred = kmeansModel.predict(queryVec).item()
    
    # 3. 计算sent与类别i中所有问题句向量的相似度，排序
    sentsMat = np.array(list(enumerate(sentsVec)))[kmeansModel.labels_==y_pred]    # DataFrame类型
    sortResult = sortSimilarty_vectorized(sentsMat, queryVec, funcName=distanceFunc)[:10]
    questionTextList  = qaDataset.loc[[i[0] for i in sortResult],"question"]
    answerTextList = qaDataset.loc[[i[0] for i in sortResult],"answer"]
    similarityValues = [i[1] for i in sortResult]
    
    return questionTextList, answerTextList, similarityValues

if __name__ == '__main__':
    qaDataset = pd.read_csv("./data/qa_corpus.csv").drop([3198, 12749, 13649, 19748, 32510])
    
    model = loadW2vModel('./models/word2vec_v1.1.model')  
    kmeansModel = loadKmeansModel('./models/kmeans_50.pickle')
    sentsSIFVec = np.load('./models/questions_SIFVec', allow_pickle=True)
    
    questionTextList, answerTextList, similarityValues = getAnswers("你叫啥", model, kmeansModel, sentsSIFVec, qaDataset)
    [print(i) for i in questionTextList];














