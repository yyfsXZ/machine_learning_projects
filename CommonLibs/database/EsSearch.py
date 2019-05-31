#!/usr/env/bin python
#coding=utf-8

import sys
import json
from elasticsearch import Elasticsearch

class EsSearcher:
    def __init__(self):
        self.es = None

    def initialize(self, ip, port):
        try:
            self.es = Elasticsearch([ip], port=port)
            print self.es.info()
            return True
        except Exception, err:
            print "failed to connect to es, err=%s" % err
            return False

    def search_by_query(self, index_name, query, max_return_num=10):
        query_json = {"match": {"question": {"query": query, "type": "boolean"}}}
        res = self.es.search(index=index_name, body={"query": query_json}, size=max_return_num)
        return res["hits"]["hits"]


if __name__ == "__main__":
    esSearch = EsSearcher()
    ip = "127.0.0.1"
    port = "9250"
    if not esSearch.initialize(ip, port):
        print "failed to init es"
        sys.exit(1)

    index_name = "my_index"
    while 1:
        query = raw_input("Input your query\n")
        hits = esSearch.search_by_query(index_name, query, max_return_num=2)
        for hit in hits:
			print hit

