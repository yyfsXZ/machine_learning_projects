#!/usr/env/bin python
#coding=utf-8

import os
import sys

from gensim.models import Word2Vec
from CommonLibs.FileIoUtils import MySentences

if __name__ == "__main__":
    # 读取文件到list
    sentences = MySentences(sys.argv[1])
    print len(sentences)

    # 训练
    model = Word2Vec(sentences,
                     size=200,
                     window=3,
                     min_count=3,
                     workers=2)
    model.save('word2vec.model')
    model.wv.save_word2vec_format('vector.txt')