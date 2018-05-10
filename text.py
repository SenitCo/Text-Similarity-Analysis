#!/usr/bin/env python 
# -*- coding:utf-8 -*- 

import os
import csv
import jieba
import jieba.analyse
import thulac
import time
import heapq
import numpy as np

import codecs
from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.models import doc2vec
from gensim.models import LsiModel
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities
from gensim import corpora
from pprint import pprint
from args import parse_args

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

args = parse_args()
num_topics = args.num_topics
cut_method = args.cut_method
vec_method = args.vec_method
use_stopwords = args.use_stopwords
force_cut = args.force_cut
use_tfidf = True

doc2vec_param = {}
doc2vec_param['min_count'] = 1
doc2vec_param['window'] = 1
doc2vec_param['iter'] = args.epochs

jieba.enable_parallel(8)

def load_data(file_path):
    data_dict = {}
    csvFile = open(file_path, 'r', encoding="utf-8")
    reader = csv.reader(csvFile)
    for item in reader:
        data_dict[item[0]] = item[1]
    return data_dict

# 使用jieba分词
def cut_words_jieba(data_dict, stopwords=None):
    seg_dict = {}
    text_list = []
    i = 0
    for key in data_dict:
        if i % 100 == 0:
            print(key)
        i = i + 1
        seg_dict[key] = jieba.cut(data_dict[key])
        if stopwords != None:       # 如果停用词stopwords不为None，则执行过滤操作
            seg_dict[key] = filter_stopwords(seg_dict[key], stopwords)
        text_list.append(seg_dict[key])
    return seg_dict, text_list

# 使用thulac分词
def cut_words_thulac(data_dict, stopwords=None):
    cut_dict = {}
    text_list = []
    thu = thulac.thulac(seg_only=True)
    i = 0
    for key in data_dict:
        if i % 100 == 0:
            print(key)
        i = i + 1
        cut_dict[key] = thu.fast_cut(data_dict[key], text=True)
        if stopwords != None:       # 如果停用词stopwords不为None，则执行过滤操作
            cut_dict[key] = filter_stopwords(cut_dict[key], stopwords)
        text_list.append(cut_dict[key])
    return cut_dict, text_list

# 根据停用词表滤除文本中的相关词
def filter_stopwords(word_list, stopwords):
    filterd_words = []
    text_list = []
    for word in word_list:
        if word not in stopwords:
            filterd_words.append(word)
    return filterd_words

# 使用jiebe的TextRank算法进行分词并提取关键词
def extract_keywords_textrank(data_dict):
    keywords_dict = {}
    text_list = []
    i = 0
    for key in data_dict:
        if i % 100 == 0:
            print(key)
        i = i + 1
        # 直接调用jieba的关键词提取方法（e.g. textrank, extract_tags）
        keywords_dict[key] = jieba.analyse.textrank(data_dict[key], withWeight=False, 
            allowPOS=('n', 'ns', 'nr', 'nt', 'nz', 'a', 'ad', 'an', 'vn', 'v', 'vg', 
                'vn', 'z', 'q', 's', 't', 'Ng', 'd', 'dg', 'b', 'f', 'g', 'i', 'I', 'm'))
        text_list.append(keywords_dict[key])
    return keywords_dict, text_list

# 使用jieba的tfidf算法进行分词并提取关键词
def extract_keywords_tfidf(data_dict):
    keywords_dict = {}
    text_list = []
    i = 0
    for key in data_dict:
        if i % 100 == 0:
            print(key)
        i = i + 1
        keywords_dict[key] = jieba.analyse.extract_tags(data_dict[key], topK=50, withWeight=False)
        text_list.append(keywords_dict[key])
    return keywords_dict, text_list

# 存储关键词
def save_keywords(seg_dict, keywords_file):
    pardir = os.path.abspath(os.path.dirname(keywords_file))
    if not os.path.exists(pardir):
        os.mkdir(pardir)
    with codecs.open(keywords_file, "w", encoding="utf-8") as f:
        for key in seg_dict:
            f.write(key + "\t" + " ".join(seg_dict[key]) + "\r\n")   
        f.close()

# 加载关键词
def load_keywords(keywords_file):
    with codecs.open(keywords_file, "r") as f:
        keywords_dict = {}
        for line in f:
            line = line.split()
            # print(line[0], len(line))
            if(len(line) > 1):
                keywords_dict[line[0]] = line[1:]
            else:
                keywords_dict[line[0]] = []
        f.close()
    return keywords_dict

def get_train_words(cut_word):
    pass

# 训练LDA模型
def train_LDA_model(text_list):
    dictionary, corpus = get_dict_corpora(cut_method, text_list)
    lda_path, index_path = get_lda_model_file(cut_method, num_topics)
    # 如果LDA模型存在则加载，否则训练模型并保存
    if os.path.exists(lda_path):
        lda = LdaModel.load(lda_path)
    else:
        # lda = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=8)
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        lda.save(lda_path)

    if os.path.exists(index_path):
        index = similarities.MatrixSimilarity.load(index_path)
    else:        
        index = similarities.MatrixSimilarity(lda[corpus])
        index.save(index_path)
    # lda.print_topics(20)
    # now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    return lda, index

# 训练LSI模型
def train_LSI_model(text_list):
    dictionary, corpus = get_dict_corpora(cut_method, text_list)
    tfidf_path = "model/tfidf_model.tfidf"
    if use_tfidf:
        if os.path.exists(tfidf_path):
            tfidf = TfidfModel.load(tfidf_path)
        else:
            tfidf = TfidfModel(corpus)
            tfidf.save(tfidf_path)
        corpus = tfidf[corpus]
    lsi_path, index_path = get_lsi_model_file(cut_method, num_topics, use_tfidf)

    if os.path.exists(lsi_path) and os.path.exists(index_path):
        lsi = LsiModel.load(lsi_path)
        index = similarities.MatrixSimilarity.load(index_path)
    else:
        lsi = LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        index = similarities.MatrixSimilarity(lsi[corpus])
        lsi.save(lsi_path)
        index.save(index_path)

    return lsi, index

# 训练或者加载词典、语料库
def get_dict_corpora(cut_method, text_list):
    dict_path, corpus_path = get_corpora_file(cut_method)
    # 如果词典存在则加载，否则创建词典并保存
    if os.path.exists(dict_path):
        dictionary = corpora.Dictionary.load(dict_path)
    else:
        dictionary = corpora.Dictionary(text_list)
        dictionary.save(dict_path)
    # 如果语料库存在则加载，否则生成语料库并保存
    if os.path.exists(corpus_path):
        corpus = corpora.MmCorpus(corpus_path)
    else:
        corpus = [dictionary.doc2bow(text) for text in text_list]
        corpora.MmCorpus.serialize(corpus_path, corpus)
    return dictionary, corpus

# 获取词典和语料库的存储路径
def get_corpora_file(cut_method):
    file_dir = "corpora/"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    dict_path = file_dir + "dict.dict"
    corpus_path = file_dir + "corpus.mm"
    return dict_path, corpus_path

# 获取LDA模型文件的存储路径
def get_lda_model_file(cut_method, num_topics):
    file_dir = "model/" + cut_method + "-" + str(num_topics)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    lda_path = file_dir + "/model-" + str(num_topics) + ".lda"
    index_path = file_dir + "/sim-" + str(num_topics) + ".index"
    # dict_path = file_dir + "/dict-" + str(num_topics) + ".dict"
    # corpus_path = file_dir + "/corpus-" + str(num_topics) + ".mm"
    # return lda_path, index_path, dict_path, corpus_path
    return lda_path, index_path

# 获取LSI模型文件的存储路径
def get_lsi_model_file(cut_method, num_topics, use_tfidf):
    file_dir = "model/" + cut_method + "-" + str(num_topics)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if use_tfidf:
        tag = "-tfidf"
    else:
        tag = ''
    lsi_path = file_dir + "/lsi-model-" + str(num_topics) + tag + ".lsi"
    index_path = file_dir + "/lsi-sim-" + str(num_topics) + tag +  ".lsi_index"
    return lsi_path, index_path

# 训练doc2vec模型
def train_doc2vec_model(text_list):
    train_corpus = []
    for i, text in enumerate(text_list):
        tagged_text = doc2vec.TaggedDocument(text, [i])
        train_corpus.append(tagged_text)
    model = doc2vec.Doc2Vec(size=num_topics, min_count=doc2vec_param['min_count'], window=doc2vec_param['window'], 
                iter=doc2vec_param['iter'], workers=8)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model_path = get_vec_model_file(cut_method, vec_method, num_topics, doc2vec_param)
    model.save(model_path)
    return model

def get_vec_model_file(cut_method, vec_method, num_topics, doc2vec_param):
    file_dir = "model/" + cut_method + "-" + str(num_topics) 
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name = vec_method + str(num_topics) + "-min_count" + str(doc2vec_param['min_count']) + "-window" + \
                str(doc2vec_param['window'])+ "-iter" + str(doc2vec_param['iter']) + ".model"
    model_path = os.path.join(file_dir, file_name)
    return model_path    

# 获取关键词文件的路径
def get_keywords_file(cut_method):
    file_path = "model/cut-" + cut_method +  ".txt"
    return file_path

# 筛选最大的n个值所对应的元素
def max_sim_n(data, n):
    max_sim = heapq.nlargest(n, enumerate(data), key=lambda item: item[1])
    return max_sim

# 加载主题模型（LDA、LSI）
def load_topic_model(vec_method, model_path, index_path, dict_path, corpus_path):
    if vec_method == 'LDA':
        model = LdaModel.load(model_path)
    elif vec_method == 'LSI':
        model = LsiModel.load(model_path)
    index = similarities.MatrixSimilarity.load(index_path)
    dictionary = corpora.Dictionary.load(dict_path)
    corpus = corpora.MmCorpus(corpus_path)
    # vec_lda = lda[corpus]
    return model, index, dictionary, corpus

def load_doc2vec_model():
    model_path = get_vec_model_file(cut_method, vec_method, num_topics, doc2vec_param)
    model = doc2vec.Doc2Vec.load(model_path)
    return model


def load_txt(file_path):
    with codecs.open(file_path, 'r', encoding='utf8') as f:
        text = {}  
        for i, line in enumerate(f):   
            text[str(i)] = line
        f.close()
    return text

def train(cut_method='jieba', vec_method='LDA', use_stopwords=True):
    file_path = "train_data.csv"
    stopwords = None
    if use_stopwords:   # 是否加载停用词
        stopwords = codecs.open("stopwords.txt", 'r', encoding='utf8').readlines()
        stopwords = [ w.strip() for w in stopwords ]

    data_dict = load_data(file_path)
    keywords_file = get_keywords_file(cut_method)
    if os.path.exists(keywords_file):   # 如果关键词文件存在，则直接加载；否则进行分词提取关键词
        keywords_dict = load_keywords(keywords_file)
        text_list = list(keywords_dict.values())
    else:
        if cut_method == 'jieba_textrank':
            seg_dict, text_list = extract_keywords_textrank(data_dict)
        elif cut_method == 'jieba_tfidf':
            seg_dict, text_list = extract_keywords_tfidf(data_dict)
        elif cut_method == 'jieba':
            seg_dict, text_list = cut_words_jieba(data_dict, stopwords)
        elif cut_method == 'thulac':
            seg_dict, text_list = cut_words_thulac(data_dict, stopwords)
        save_keywords(seg_dict, keywords_file)
    print(len(text_list))
    if vec_method == 'LDA':
        lda, index = train_LDA_model(text_list)
    elif vec_method == 'LSI':
        lsi, index = train_LSI_model(text_list)
    elif vec_method == 'doc2vec':
        model = train_doc2vec_model(text_list)

def test(cut_method='jieba', vec_method='LDA', use_stopwords=True):
    file_path = "test_data.csv"
    stopwords = None
    if use_stopwords:   # 是否加载停用词
        stopwords = codecs.open("stopwords.txt", 'r', encoding='utf8').readlines()
        stopwords = [ w.strip() for w in stopwords ]

    data_dict = load_data(file_path)
    if cut_method == 'jieba_textrank':
        seg_dict, text_list = extract_keywords_textrank(data_dict)
    elif cut_method == 'jieba_tfidf':
            seg_dict, text_list = extract_keywords_tfidf(data_dict)
    elif cut_method == 'jieba':
        seg_dict, text_list = cut_words_jieba(data_dict, stopwords)
    elif cut_method == 'thulac':
        seg_dict, text_list = cut_words_thulac(data_dict, stopwords)       
    pprint(seg_dict)

    if vec_method == 'LDA' or vec_method == 'LSI':
        test_topic_model(seg_dict, vec_method)
    elif vec_method == 'doc2vec':
        test_doc2vec_model(seg_dict)


def test_topic_model(seg_dict, vec_method):
    text_list = list(seg_dict.values())
    dict_path, corpus_path = get_corpora_file(cut_method)
    if vec_method == 'LDA':
        model_path, index_path = get_lda_model_file(cut_method, num_topics)
    elif vec_method == 'LSI':
        model_path, index_path = get_lsi_model_file(cut_method, num_topics, use_tfidf)
    model, index, dictionary, corpus = load_topic_model(vec_method, model_path, index_path, dict_path, corpus_path)
    # 保存输出结果
    result_file = vec_method + "-" + str(num_topics) + "-" + "results.txt"
    with codecs.open(result_file, "w", encoding="utf-8") as f:
        f.write("source_id\ttarget_id\r\n")
        for num, key in enumerate(seg_dict):
            if num == 0:
                continue
            print(key)
            vec_bow = dictionary.doc2bow(seg_dict[key])
            vec_lda = model[vec_bow]
            sims = index[vec_lda]
            max_sim = max_sim_n(sims, 21)
            i = 0
            for item in max_sim:    
                if i == 20:
                    break
                if str(item[0]) == key:
                    continue
                i = i + 1
                f.write(key + "\t" + str(item[0]) + "\r\n")
        f.close()

def test_doc2vec_model(seg_dict):
    model = load_doc2vec_model()
    keywords_file = get_keywords_file(cut_method)
    train_text = load_keywords(keywords_file)
    print("len =", len(train_text))
    with codecs.open("results_doc2vec.txt", "w", encoding="utf-8") as f:
        f.write("source_id\ttarget_id\r\n")
        for num, key in enumerate(seg_dict):
            if num == 0:
                continue
            print(key)
            test_vec = model.infer_vector(seg_dict[key])
            max_sim = model.docvecs.most_similar([test_vec], topn=21)
            i = 0
            for item in max_sim:    
                if i == 20:
                    break
                if str(item[0]) == key:
                    continue
                i = i + 1
                f.write(key + "\t" + str(item[0]) + "\r\n")
        f.close()

    with codecs.open("results_doc2vec_output.txt", "w", encoding="utf-8") as f:
        f.write("source_id\ttarget_id\r\n")
        for num, key in enumerate(seg_dict):
            if num == 0:
                continue
            print(key)
            test_vec = model.infer_vector(seg_dict[key])
            max_sim = model.docvecs.most_similar([test_vec], topn=21)
            i = 0
            f.write(key + "\t" + ", ".join(seg_dict[key]) + "\r\n")
            for item in max_sim:    
                f.write(str(item[0]) + "\t" + ", ".join(train_text[str(item[0])]) + "\r\n")
                i = i + 1
            f.write("\r\n============================================\r\n")
        f.close()


def main():
    print("num_topics -- ", num_topics)
    print("cut_method -- ", cut_method)
    print("use_stopwords -- ", use_stopwords) 
    print("vec_method -- ", vec_method)
    print("Begin ", args.is_train)

    if args.is_train == 'training':
        train(cut_method, vec_method, use_stopwords)
    else:
        test(cut_method, vec_method, use_stopwords) 

    print("num_topics -- ", num_topics)
    print("cut_method -- ", cut_method)
    print("use_stopwords -- ", use_stopwords) 
    print("vec_method -- ", vec_method)
    print("End ", args.is_train) 

if __name__ == "__main__":
    main()