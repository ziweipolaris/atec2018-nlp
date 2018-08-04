#/usr/bin/env python
#coding=utf-8

import os
import re
import multiprocessing

import gensim
from gensim.models.word2vec import LineSentence
import jieba_fast as jieba
import numpy as np
import pandas as pd
import fasttext


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
model_dir = "pai_model/"

new_words = "支付宝 付款码 二维码 收钱码 转账 退款 退钱 余额宝 运费险 还钱 还款 花呗 借呗 蚂蚁花呗 蚂蚁借呗 蚂蚁森林 小黄车 飞猪 微客 宝卡 芝麻信用 亲密付 淘票票 饿了么 摩拜 滴滴 滴滴出行".split(" ")
for word in new_words:
    jieba.add_word(word)

class MyChars(object):
    def __init__(self):
        pass

    def __iter__(self):
        with open(model_dir + "atec_nlp_sim_train.csv","r", encoding="utf8") as atec:
            for line in atec:
                lineno, s1, s2, label=line.strip().split("\t")
                yield list(s1) + list(s2)

        with open("resources/wiki_corpus/wiki.csv",'r',encoding="utf8") as wiki:
            for line in wiki:
                title, doc = line.strip().split("|")
                for sentense in doc.split("#"):
                    if len(sentense)>0:
                        yield [char for char in sentense if char and 0x4E00<= ord(char[0]) <= 0x9FA5]
        

class MyWords(object):
    def __init__(self):
        pass
 
    def __iter__(self):
        with open(model_dir + "atec_nlp_sim_train.csv","r", encoding="utf8") as atec:
            for line in atec:
                lineno, s1, s2, label=line.strip().split("\t")    
                yield list(jieba.cut(s1)) + list(jieba.cut(s2))

        with open("resources/wiki_corpus/wiki.csv",'r',encoding="utf8") as wiki:
            for line in wiki:
                title, doc = line.strip().split("|")
                for sentense in doc.split("#"):
                    if len(sentense)>0:
                        yield [word for word in list(jieba.cut(sentense)) if word and 0x4E00<= ord(word[0]) <= 0x9FA5]


def gen_data():
    with open(model_dir + "train_char.txt","w",encoding="utf8") as file:
        mychars = MyChars()
        for cs in mychars:
            file.write(" ".join(cs)+"\n")

    with open(model_dir + "train_word.txt","w",encoding="utf8") as file:
        mywords = MyWords()
        for ws in mywords:
            file.write(" ".join(ws)+"\n")

def train_embedding_gensim():
    dim=256
    embedding_size = dim
    model = gensim.models.Word2Vec(LineSentence(model_dir + 'train_char.txt'),
                                   size=embedding_size,
                                   window=5,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())

    model.save(model_dir + "char2vec_gensim"+str(embedding_size))
    # model.wv.save_word2vec_format("model/char2vec_org"+str(embedding_size),"model/chars"+str(embedding_size),binary=False)
    
    dim=256
    embedding_size = dim
    model = gensim.models.Word2Vec(LineSentence(model_dir + 'train_word.txt'),
                                   size=embedding_size,
                                   window=5,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())

    model.save(model_dir + "word2vec_gensim"+str(embedding_size))
    # model.wv.save_word2vec_format("model/word2vec_org"+str(embedding_size),"model/vocabulary"+str(embedding_size),binary=False)


def train_embedding_fasttext():
    
    # Skipgram model
    model = fasttext.skipgram(model_dir + 'train_char.txt', model_dir + 'char2vec_fastskip256', word_ngrams=2, ws=5, min_count=10, dim=256)
    del(model)

    # CBOW model
    model = fasttext.cbow(model_dir + 'train_char.txt', model_dir + 'char2vec_fastcbow256', word_ngrams=2, ws=5, min_count=10, dim=256)
    del(model)

    # Skipgram model
    model = fasttext.skipgram(model_dir + 'train_word.txt', model_dir + 'word2vec_fastskip256', word_ngrams=2, ws=5, min_count=10, dim=256)
    del(model)

    # CBOW model
    model = fasttext.cbow(model_dir + 'train_word.txt', model_dir + 'word2vec_fastcbow256', word_ngrams=2, ws=5, min_count=10, dim=256)
    del(model)

# gen_data()
train_embedding_gensim()
# train_embedding_word()