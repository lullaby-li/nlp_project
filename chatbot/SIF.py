# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:19:09 2019

"""
import numpy as np
from sklearn.decomposition import PCA
from gensim.models import word2vec
from collections import Counter
import jieba, re
from hanziconv import HanziConv
import logging

def get_stopwords(path='./Project01/content/stopwords.txt', punctuations=True, bookmarks=False, text_stopwords=True):
    '''
    加载停用词表，去掉一些噪声
    punctions是否作为停用词，是停用词为False（不保留），要保留在语料里为True
    bookmarks是否作为停用词，是停用词为False（不保留），要保留在语料里为True
    text_stopwords是否作为停用词，是停用词为False（不保留）, 要保留在语料里为True
    :return: 停用词集合
    '''
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # 加载停用词表
    stopword_set = set()
    with open(path, 'r', encoding="utf-8") as stopwords:  # stopwords.txt停用词表，我放在同一目录下
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
    stopword_set = set([i.strip() for i in stopword_set])
    if punctuations == True: [stopword_set.remove(i) for i in ["！","!","，",",","。",".","：","；",";",":","？","?"]];
    if bookmarks == True: [stopword_set.remove(i) for i in ["《","》"]];
    # if text_stopwords == True: [stopword_set.remove(i) for i in set(["第二","一番","一直","一个","一些","许多","种","有的是","也就是说","末##末","啊","阿","哎","哎呀","哎哟","唉","俺","俺们","按","按照","吧","吧哒","把","罢了","被","本","本着","比","比方","比如","鄙人","彼","彼此","边","别","别的","别说","并","并且","不比","不成","不单","不但","不独","不管","不光","不过","不仅","不拘","不论","不怕","不然","不如","不特","不惟","不问","不只","朝","朝着","趁","趁着","乘","冲","除","除此之外","除非","除了","此","此间","此外","从","从而","打","待","但","但是","当","当着","到","得","的","的话","等","等等","地","第","叮咚","对","对于","多","多少","而","而况","而且","而是","而外","而言","而已","尔后","反过来","反过来说","反之","非但","非徒","否则","嘎","嘎登","该","赶","个","各","各个","各位","各种","各自","给","根据","跟","故","故此","固然","关于","管","归","果然","果真","过","哈","哈哈","呵","和","何","何处","何况","何时","嘿","哼","哼唷","呼哧","乎","哗","还是","还有","换句话说","换言之","或","或是","或者","极了","及","及其","及至","即","即便","即或","即令","即若","即使","几","几时","己","既","既然","既是","继而","加之","假如","假若","假使","鉴于","将","较","较之","叫","接着","结果","借","紧接着","进而","尽","尽管","经","经过","就","就是","就是说","据","具体地说","具体说来","开始","开外","靠","咳","可","可见","可是","可以","况且","啦","来","来着","离","例如","哩","连","连同","两者","了","临","另","另外","另一方面","论","嘛","吗","慢说","漫说","冒","么","每","每当","们","莫若","某","某个","某些","拿","哪","哪边","哪儿","哪个","哪里","哪年","哪怕","哪天","哪些","哪样","那","那边","那儿","那个","那会儿","那里","那么","那么些","那么样","那时","那些","那样","乃","乃至","呢","能","你","你们","您","宁","宁可","宁肯","宁愿","哦","呕","啪达","旁人","呸","凭","凭借","其","其次","其二","其他","其它","其一","其余","其中","起","起见","起见","岂但","恰恰相反","前后","前者","且","然而","然后","然则","让","人家","任","任何","任凭","如","如此","如果","如何","如其","如若","如上所述","若","若非","若是","啥","上下","尚且","设若","设使","甚而","甚么","甚至","省得","时候","什么","什么样","使得","是","是的","首先","谁","谁知","顺","顺着","似的","虽","虽然","虽说","虽则","随","随着","所","所以","他","他们","他人","它","它们","她","她们","倘","倘或","倘然","倘若","倘使","腾","替","通过","同","同时","哇","万一","往","望","为","为何","为了","为什么","为着","喂","嗡嗡","我","我们","呜","呜呼","乌乎","无论","无宁","毋宁","嘻","吓","相对而言","像","向","向着","嘘","呀","焉","沿","沿着","要","要不","要不然","要不是","要么","要是","也","也罢","也好","一","一般","一旦","一方面","一来","一切","一样","一则","依","依照","矣","以","以便","以及","以免","以至","以至于","以致","抑或","因","因此","因而","因为","哟","用","由","由此可见","由于","有","有的","有关","有些","又","于","于是","于是乎","与","与此同时","与否","与其","越是","云云","哉","再说","再者","在","在下","咱","咱们","则","怎","怎么","怎么办","怎么样","怎样","咋","照","照着","者","这","这边","这儿","这个","这会儿","这就是说","这里","这么","这么点儿","这么些","这么样","这时","这些","这样","正如","吱","之","之类","之所以","之一","只是","只限","只要","只有","至","至于","诸位","着","着呢","自","自从","自个儿","自各儿","自己","自家","自身","综上所述","总的来看","总的来说","总的说来","总而言之","总之","纵","纵令","纵然","纵使","遵照","作为","兮","呃","呗","咚","咦","喏","啐","喔唷","嗬","嗯","嗳"])];
    return stopword_set

def split_sentence(document):
    '''
    this is to split a doc into sentences
    分句之前先把\n，\r，\r\n，\u3000直接洗掉即可
    分句的规则有：1.碰到句号，问号，感叹号，分号还有空格就分，并把标点符号加到前句
    '''
    document = document.strip().replace("\r", "").replace("\n", "").replace("\\n", "").replace("\u3000", "").replace("\\", "")
    # 以结尾标点切分
    splitted = re.split(r"([？?。!！；…])", document)
    # 标点符号加回句子
    splitted.append("")
    splitted = ["".join(i) for i in zip(splitted[0::2],splitted[1::2])]
    return [i for i in splitted if i != '']



def wash_and_split(input_text, punctuations=True, bookmarks=False, text_stopwords=True, my_stopwords={}):
    '''
    主要的数据预处理函数
    输入一篇文章string
    输出分词分句并清洗后的句子列表，以备训练word2vec使用
    
    关于停用词见 get_stopwords 函数的注释
    '''
    if input_text.strip() == "": return [[""]]
    
    sentences = split_sentence(HanziConv.toSimplified(input_text))
    cutted = list(map(jieba.lcut, sentences))
    
    stopwds = get_stopwords(path='../Project01/content/stopwords.txt', punctuations=punctuations, bookmarks=bookmarks, text_stopwords=text_stopwords)
    
    def remove_stopwords_for_each_sentence(list_of_words):
        return [word for word in list_of_words if word not in stopwds.union(" ").union(my_stopwords)]
    
    cutted = map(remove_stopwords_for_each_sentence, cutted)
    return [i for i in list(cutted) if i != []]


# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
def SIF(sentence_list, model, a: float=1e-3, unlisted_word_freq=0.0001, skip_mode=False):
    '''
    Input:
        sentence_list: a list of tokenized sentences, text format
        a: param
        model: word2vec model object, pretrained by gensim
        embedding_size: word2vec model embedding size
    Output:
        SIF sentence embeddings
    '''
    
    vlookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    embedding_size = model.vector_size  # Embedding size

    vocab_count = 0
    for k in vlookup:
        vocab_count += vlookup[k].count # Compute the normalization constant Z (ALL Words Count)

    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        for word in sentence:
            if word in vlookup:
                a_value = a / (a + vlookup[word].count / vocab_count)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, vectors[word]))  # vs += sif * word_vector
            if word not in vlookup:
                a_value = a / (a + unlisted_word_freq)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, np.zeros(embedding_size)))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average

        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    # only if we have more than 2 sentences!
    if len(sentence_set) >= 2:
        pca = PCA()
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT
    else: 
        return vs

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return np.array(sentence_vecs)


if __name__ == "__main__":
    model = word2vec.Word2Vec.load('C:/Users/yihua/学习/开课吧NLP_course/my/Project01/model/word2vec_v1.1.model')
    
    sample_text = ["今天小明吃了一个包子，他觉得非常好吃。", "于是他又去买了一个包子，然后喂了狗。", "但是还不够爽，于是他又买了一笼包子。", "然后又喂了猫。"]
    sampt = [jieba.lcut(s) for s in sample_text]
    a = SIF(sampt, model)
