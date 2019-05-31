#!/usr/env/bin python
#coding=utf-8

import jieba

class Nlp(object):
    def __init__(self):
        pass

    def wordseg(self, query):
        return [word.encode("utf-8") for word in jieba.cut(query)]