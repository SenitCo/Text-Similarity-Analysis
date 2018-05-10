#!/usr/bin/env/python
#-*- encoding:utf-8 -*-

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training a text similarity analysis model!")
    # parser.add_argument('--device_id', dest='device_id', help='device id to use', default=0, type=int)
    parser.add_argument('--is_train', dest='is_train', default='training', choices=['training', 'test'], 
                        help='training or test', type=str)
    parser.add_argument('--num_topics', dest='num_topics', default=500, help='number of topics', type=int)
    parser.add_argument('--cut_method', dest='cut_method', default='jieba', 
                        choices=['jieba', 'thulac', 'jieba_tfidf', 'jieba_textrank'],
                        help='method to cut word', type=str)
    parser.add_argument('--vec_method', dest='vec_method', default='LDA', choices=['LDA', 'LSI', 'word2vec', 'doc2vec'], 
                        help='method to vectorize text', type=str)
    parser.add_argument('--use_stopwords', dest='use_stopwords', default=True, help='Whether to use stopwords', type=bool)
    parser.add_argument('--epochs', dest='epochs', default=100, help='epochs of training doc2vec model', type=int)
    parser.add_argument('--force_cut', dest='force_cut', default=False, help='Whether to cut word forcibly', type=bool)

    args = parser.parse_args()
    return args
    