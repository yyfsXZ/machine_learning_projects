#!/usr/env/bin python
#coding=utf-8

import os
import sys
import math
import time

import libs.PreProcess
import libs.Nlp

class Wordcut(object):
    def __init__(self):
        self.nlp = libs.Nlp.Nlp()
        self.preprocess = libs.PreProcess.PreProcess()

    def queryTerms(self, query):
        query = self.preprocess.removePuncuation(query)
        return self.nlp.wordseg(query)

    def wordbag(self, input_filename, output_filename):
        fp = open(input_filename, 'r')
        liLines = fp.readlines()
        fp.close()
        wp = open(output_filename, 'w')
        count = 0
        start_t = time.time()
        for line in liLines:
            count += 1
            if count % 100000 == 0:
                end_t = time.time()
                dur = end_t - start_t
                print "%d\t%d" % (count, dur)
                start_t = end_t
            #line = fp.readline().strip('\r\n')
            words = self.queryTerms(line)
            print >> wp, "\t".join(words)
        #fp.close()
        wp.close()

    def run(self, input_filename, output_filename):
        # 首先统计词袋信息，取出一元词频和二元词频，以及二元词频的左右词频
        self.wordbag(input_filename, output_filename)


if __name__ == "__main__":
    wordcut = Wordcut()
    wordcut.run(sys.argv[1], sys.argv[2])

