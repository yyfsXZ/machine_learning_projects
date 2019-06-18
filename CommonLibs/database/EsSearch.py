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

    def get_index_count(self, index_name):
        return self.es.count(index=index_name)["count"]

    def get_index_querys(self, index_name):
        query_json = {"match_all": {}}
        # res = self.es.search(index=index_name, body={"query": query_json})
        # return res["hits"]["hits"]
        query = es.search(index=index_name, body={"query": query_json}, scroll='5m', size=100)

        results = query['hits']['hits'] # es查询出的结果第一页
        total = query['hits']['total']  # es查询出的结果总量
        scroll_id = query['_scroll_id'] # 游标用于输出es查询出的所有结果

        for i in range(0, int(total/100)+1):
            # scroll参数必须指定否则会报错
            query_scroll = es.scroll(scroll_id=scroll_id, scroll='5m')['hits']['hits']
            results += query_scroll
        return results

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

