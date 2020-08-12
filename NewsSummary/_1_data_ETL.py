#!/usr/bin/env python
# coding: utf-8

import logging, jieba, os, re, json, time
import pandas as pd
from hanziconv import HanziConv

def preprocess_wiki_data(filepath = "content/raw_zhwiki/AA", output_path = "content/wiki_data"):
    '''
    预处理中文维基源数据并保存
    包括：繁体字汉化、只提取词条的词条名和正文
    保存为一行行的Json
    '''
    data = list()
    for file in os.listdir(filepath):
        path = os.path.join(filepath, file)
        with open(path, encoding='utf-8') as f:
            for record in f.read().strip().split("\n"):
                line = json.loads(record)
                data.append({
                        "title": HanziConv.toSimplified(line["title"]),
                        "content": HanziConv.toSimplified(line["text"])
                        })
    with open(output_path, "w", encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record)+'\n')
    print("Wiki Data Saved into: {}".format(output_path))

#preprocess_wiki_data()

def preprocess_news_data(filepath = "content/raw_news/sqlResult_1558435.csv", output_path = "content/news_data"):
    '''
    预处理新闻源数据并保存
    包括：只提取新闻的新闻名和正文
    保存为一行行的Json
    '''
    df = pd.read_csv(filepath, encoding = "ANSI")
    string = df[["title", "content"]].fillna("").to_json(force_ascii=False, orient="records")
    with open(output_path, "w", encoding = 'utf-8') as f:
        for record in eval(string):
            f.write(json.dumps(record)+'\n')
    print("News Data Saved into: {}".format(output_path))
        
#preprocess_news_data()









def get_stopwords(path='content/stopwords.txt', punctuations=True, bookmarks=False, text_stopwords=True):
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
    if text_stopwords == True: [stopword_set.remove(i) for i in set(["第二","一番","一直","一个","一些","许多","种","有的是","也就是说","末##末","啊","阿","哎","哎呀","哎哟","唉","俺","俺们","按","按照","吧","吧哒","把","罢了","被","本","本着","比","比方","比如","鄙人","彼","彼此","边","别","别的","别说","并","并且","不比","不成","不单","不但","不独","不管","不光","不过","不仅","不拘","不论","不怕","不然","不如","不特","不惟","不问","不只","朝","朝着","趁","趁着","乘","冲","除","除此之外","除非","除了","此","此间","此外","从","从而","打","待","但","但是","当","当着","到","得","的","的话","等","等等","地","第","叮咚","对","对于","多","多少","而","而况","而且","而是","而外","而言","而已","尔后","反过来","反过来说","反之","非但","非徒","否则","嘎","嘎登","该","赶","个","各","各个","各位","各种","各自","给","根据","跟","故","故此","固然","关于","管","归","果然","果真","过","哈","哈哈","呵","和","何","何处","何况","何时","嘿","哼","哼唷","呼哧","乎","哗","还是","还有","换句话说","换言之","或","或是","或者","极了","及","及其","及至","即","即便","即或","即令","即若","即使","几","几时","己","既","既然","既是","继而","加之","假如","假若","假使","鉴于","将","较","较之","叫","接着","结果","借","紧接着","进而","尽","尽管","经","经过","就","就是","就是说","据","具体地说","具体说来","开始","开外","靠","咳","可","可见","可是","可以","况且","啦","来","来着","离","例如","哩","连","连同","两者","了","临","另","另外","另一方面","论","嘛","吗","慢说","漫说","冒","么","每","每当","们","莫若","某","某个","某些","拿","哪","哪边","哪儿","哪个","哪里","哪年","哪怕","哪天","哪些","哪样","那","那边","那儿","那个","那会儿","那里","那么","那么些","那么样","那时","那些","那样","乃","乃至","呢","能","你","你们","您","宁","宁可","宁肯","宁愿","哦","呕","啪达","旁人","呸","凭","凭借","其","其次","其二","其他","其它","其一","其余","其中","起","起见","起见","岂但","恰恰相反","前后","前者","且","然而","然后","然则","让","人家","任","任何","任凭","如","如此","如果","如何","如其","如若","如上所述","若","若非","若是","啥","上下","尚且","设若","设使","甚而","甚么","甚至","省得","时候","什么","什么样","使得","是","是的","首先","谁","谁知","顺","顺着","似的","虽","虽然","虽说","虽则","随","随着","所","所以","他","他们","他人","它","它们","她","她们","倘","倘或","倘然","倘若","倘使","腾","替","通过","同","同时","哇","万一","往","望","为","为何","为了","为什么","为着","喂","嗡嗡","我","我们","呜","呜呼","乌乎","无论","无宁","毋宁","嘻","吓","相对而言","像","向","向着","嘘","呀","焉","沿","沿着","要","要不","要不然","要不是","要么","要是","也","也罢","也好","一","一般","一旦","一方面","一来","一切","一样","一则","依","依照","矣","以","以便","以及","以免","以至","以至于","以致","抑或","因","因此","因而","因为","哟","用","由","由此可见","由于","有","有的","有关","有些","又","于","于是","于是乎","与","与此同时","与否","与其","越是","云云","哉","再说","再者","在","在下","咱","咱们","则","怎","怎么","怎么办","怎么样","怎样","咋","照","照着","者","这","这边","这儿","这个","这会儿","这就是说","这里","这么","这么点儿","这么些","这么样","这时","这些","这样","正如","吱","之","之类","之所以","之一","只是","只限","只要","只有","至","至于","诸位","着","着呢","自","自从","自个儿","自各儿","自己","自家","自身","综上所述","总的来看","总的来说","总的说来","总而言之","总之","纵","纵令","纵然","纵使","遵照","作为","兮","呃","呗","咚","咦","喏","啐","喔唷","嗬","嗯","嗳"])];
    return stopword_set

def read_(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
        return data

#ad = read_("content/wiki_data")
#bd = read_("content/news_data")

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



def wash_and_split(input_text, punctuations=True, bookmarks=False, text_stopwords=True):
    '''
    主要的数据预处理函数
    输入一篇文章string
    输出分词分句并清洗后的句子列表，以备训练word2vec使用
    
    关于停用词见 get_stopwords 函数的注释
    '''
    if input_text == "": return [[""]]
    
    sentences = split_sentence(HanziConv.toSimplified(input_text))
    cutted = list(map(jieba.lcut, sentences))
    
    stopwds = get_stopwords(path='content/stopwords.txt', punctuations=punctuations, bookmarks=bookmarks, text_stopwords=text_stopwords)
    
    def remove_stopwords_for_each_sentence(list_of_words):
        for i, word in enumerate(list_of_words):
            if word in stopwds or word == " ":
                list_of_words.remove(list_of_words[i])
        return list_of_words
    
    cutted = map(remove_stopwords_for_each_sentence, cutted)
#   cutted = [i for i in list(cutted) if i != []]
    return [i for i in list(cutted) if i not in [[], ["…"]]]
        
#print(wash_and_split(bd[12]['content'], punctuations=False, text_stopwords=False))




if __name__ == "__main__":
#    wiki_data = read_("content/wiki_data")
#    news_data = read_("content/news_data")
#    everything = []
    
#    a = list(map(wash_and_split, [i["content"] for i in wiki_data]))
#    b = list(map(wash_and_split, [i["content"] for i in news_data]))
#    
#    for i, k in enumerate(wiki_data):
#        everything.extend(wash_and_split(k['content'], punctuations=False, bookmarks=False, text_stopwords=False))
#        print("\rWashing Wiki Data: {}/{}".format(i+1, len(wiki_data)), end="")
#    print("\nOK!")
#    for i, k in enumerate(news_data):
#        everything.extend(wash_and_split(k['content'], punctuations=False, bookmarks=False, text_stopwords=False))
#        print("\rWashing News Data: {}/{}".format(i+1, len(news_data)), end="")
#    print("\nOK!")


# 多线程
    from threading import Thread, Lock
    from queue import Queue

    def worker(q, fw, wash_params=(True, False, True)):
        '''
        worker是一个搬运工，_只要队列里还有任务_，他从队列q里取一行/多行数据，进行清洗分词 wash_and_split，然后整理一下，最后写进一个文件fw里面
        然后拿这个文件就可以用来训练词向量啦
        '''
        global writeLock
        while not q.empty():
            washed = wash_and_split(q.get(), wash_params[0], wash_params[1], wash_params[2])
            for sentence in washed:
                writeLock.acquire()
                fw.write(" ".join(sentence)+"\n")
                writeLock.release()
            q.task_done()
            print("\rUnfinished Tasks: {}".format(q.unfinished_tasks), end = "")
        

    num_workers = 50
#    wiki_data = read_("content/wiki_data")
#    print("wiki data loaded")
#    news_data = read_("content/news_data")
#    print('news data loaded')
    new_data1 = pd.read_csv('content/raw_added/task3-train.csv')
    new_data1 = new_data1.fillna("")
    new_data2 = pd.read_csv('content/raw_added/task3-test.csv')
    new_data2 = new_data1.fillna("")
    q = Queue()
    writeLock = Lock()
#    for i in news_data:
#        q.put(i['title'])
#        q.put(i['content'])
#    for i in wiki_data:
#        q.put(i['content'])
    for i in new_data1.itertuples():
        q.put(i.abstract)
        q.put(i.context)
        q.put(i.title)
    for i in new_data2.itertuples():
        q.put(i.abstract)
        q.put(i.context)
        q.put(i.title)

    f = open('content/new_data_to_train_word2vec', 'a+', encoding='utf-8')

    st = int(time.time())
    sn = q.unfinished_tasks

    threads = []
    for i in range(num_workers):
        threads.append(Thread(target=worker, args=(q, f, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    f.close()
        