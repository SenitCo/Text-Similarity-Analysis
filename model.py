#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import os
import csv
import jieba
import jieba.analyse
import jieba.posseg

import heapq
import pprint
import numpy as np

import re
import codecs
from gensim import corpora
#from gensim.summarization import bm25
import bm25
from text import *


def cut_word_jieba_posseg(data_dict, stop_flag, stopwords):
    seg_dict = {}
    text_list = []
    i = 0
    for key in data_dict:
        if i % 100 == 0:
            print(key)
        i = i + 1
        words = jieba.posseg.cut(data_dict[key])
        seg_dict[key] = []
        for word, tag in words:
            if tag not in stop_flag and word not in stopwords:
                seg_dict[key].append(word)
        text_list.append(seg_dict[key])
    return seg_dict, text_list

def cut_word(data_dict, stopwords):
    seg_dict = {}
    text_list = []
    i = 0
    for key in data_dict:
        seg_dict[key] = []
        words = jieba.cut(data_dict[key])
        for word in words:
            if word not in stopwords:
                seg_dict[key].append(word)
        text_list.append(seg_dict[key])
        if i % 100 == 0:
            print(key)
        i = i + 1
    return seg_dict, text_list


def get_bm25_model_file():
    dict_path = "corpora/posseg_dict.dict"
    model_path = "model/bm25_model.bm25"
    return dict_path, model_path


def train_BM25_model(text_list):
    dict_path, model_path = get_bm25_model_file()
    if os.path.exists(dict_path):
        dictionary = corpora.Dictionary.load(dict_path)
    else:
        dictionary = corpora.Dictionary(text_list)
        dictionary.save(dict_path)
    bm25_model = bm25.BM25(text_list)
    # corpus = [dictionary.doc2bow(text) for text in text_list]
    # tfidf_path = "model/bm25_tfidf_model.tfidf"
    # tfidf = TfidfModel(corpus)
    # tfidf.save(tfidf_path)
    # corpus = tfidf[corpus]
    # bm25_model = bm25.BM25(corpus)
    return bm25_model

def test_BM25_model(seg_dict, model=None):
    if model == None:
        dict_path, model_path = get_bm25_model_file()
        model = BM25.load(model_path)
    average_idf = sum(map(lambda k: float(model.idf[k]), model.idf.keys())) / len(model.idf.keys())
    result_file = "BM25" + "-" + str(num_topics) + "-" + "results.txt"
    with codecs.open(result_file, "w", encoding="utf-8") as f:
        f.write("source_id\ttarget_id\r\n")
        for num, key in enumerate(seg_dict):
            if num == 0:
                continue
            print(key)
            scores = model.get_scores(seg_dict[key], average_idf)
            max_sim = max_sim_n(scores, 21)
            i = 0
            for item in max_sim:    
                if i == 20:
                    break
                if str(item[0]) == key:
                    continue
                i = i + 1
                f.write(key + "\t" + str(item[0]) + "\r\n")
        f.close()

def train():
    train_data_path = 'train_data.csv'
    
    stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
    # stopwords = codecs.open("stopwords.txt", 'r', encoding='utf8').readlines()
    stopwords = codecs.open("punctuation.txt", 'r', encoding='utf8').readlines()  
    stopwords = [ w.strip() for w in stopwords ]
    keywords_file = "model/cut_jieba_punc_least.txt"
    if os.path.exists(keywords_file):
        keywords_dict = load_keywords(keywords_file)
        text_list = list(keywords_dict.values())
    else:
        data_dict = load_data(train_data_path)
        seg_dict, text_list = cut_word(data_dict, stopwords)
        save_keywords(seg_dict, keywords_file)
    bm25_model = train_BM25_model(text_list)
    return bm25_model

def test(model=None):
    test_data_path = 'test_data.csv'
    data_dict = load_data(test_data_path)
    stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
    # stopwords = codecs.open("stopwords.txt", 'r', encoding='utf8').readlines()
    stopwords = codecs.open("punctuation.txt", 'r', encoding='utf8').readlines()   
    stopwords = [ w.strip() for w in stopwords ]
    seg_dict, text_list = cut_word(data_dict, stopwords)
    test_BM25_model(seg_dict, model)

model = train()
test(model)
