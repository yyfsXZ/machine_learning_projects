#!/usr/env/bin python
#coding=utf-8

import string
import re

class PreProcess(object):
    def __init__(self):
        self.html_sym = re.compile(r'<[^>]+>',re.S)

    def uni2utf(self, query):
        try:
            return query.encode("utf-8")
        except:
            return query

    def utf2uni(self, query):
        try:
            return query.decode("utf-8")
        except:
            return query

    def removePuncuation(self, query):
        query = self.html_sym.sub(" ", query)
        uni_query = self.utf2uni(query)
        for idx in range(0, len(uni_query)):
            if (uni_query[idx] >= u'\u4e00' and uni_query[idx] <= u'\u9fff') or \
                (uni_query[idx] >= u'\u0041' and uni_query[idx] <= u'\u005a') or \
                (uni_query[idx] >= u'\u0061' and uni_query[idx] <= u'\u007a') or \
                (uni_query[idx] >= u'\u0030' and uni_query[idx] <= u'\u0039'):
                continue
            uni_query = uni_query.replace(uni_query[idx], " ", 1)
        return self.uni2utf(uni_query)


if __name__ == "__main__":
    handler = PreProcess()
    query = "hello,123，我们测试一下是否替换标点符号成功？如果没了就成功了!"
    print handler.removePuncuation(query)