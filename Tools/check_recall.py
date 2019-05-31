#!/usr/env/bin python
#coding=utf-8

import os
import sys
import json

def uni2utf(query):
    try:
        return query.encode("utf-8")
    except:
        return query

def load_expect(filename):
    with open(filename, 'r') as fp:
        liLines = fp.readlines()
        fp.close()
    expect_results = {}
    for line in liLines:
        splits = line.strip('\r\n').split('\t')
        query = uni2utf(splits[0])
        if float(splits[1]) < -1:
            continue
        kid = splits[2]
        stdQ = splits[3]
        expect_results[query] = {"kid": kid, "stdQ": stdQ}
    return expect_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: cat $es_result | python check_recall.py $testcases"
        sys.exit(0)

    querySet = set()
    expect_results = load_expect(sys.argv[1])
    miss = 0.0
    sum_ = 0.0
    not_matched_kids = {}
    for line in sys.stdin:
        infos = line.strip('\r\n').split("query=")[-1]
        infos = infos.split(", esResult=")
        query = uni2utf(infos[0])
        if query in querySet:
            continue
        querySet.add(query)

        if not expect_results.has_key(query):
            # print "query=%s not found" % query
            continue
        results = json.loads(infos[1])

        sum_ += 1
        r = False
        for result in results:
            kid = result["knowledgeId"]
            if expect_results[query]["kid"] == kid:
                r = True
                break
        if not r:
            if not not_matched_kids.has_key(expect_results[query]["kid"]):
                not_matched_kids[expect_results[query]["kid"]] = {"querys": [],
                                                                  "stdQ": expect_results[query]["stdQ"]}
            not_matched_kids[expect_results[query]["kid"]]["querys"].append(query)
            miss += 1
            # print "*" * 50
            # print "miss query=%s, stdQ=%s, kid=%s" % (query, expect_results[query]["stdQ"], expect_results[query]["kid"])
            # print "recall -"
            # print "\n".join([result["origQuestion"]+"\t"+result["knowledgeId"] for result in results])
    recall = (sum_ - miss) / sum_
    results = sorted(not_matched_kids.items(), key=lambda x:len(x[1]["querys"]), reverse=True)
    for elem in results:
        print "*" * 50
        print "kid=%s, stdQ=%s" % (elem[0], elem[1]["stdQ"])
        print "\n".join(elem[1]["querys"])
    print "recall=%.4f, sum=%d, miss=%d" % (recall, int(sum_), int(miss))