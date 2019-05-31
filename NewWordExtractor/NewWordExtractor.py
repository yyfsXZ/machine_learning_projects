#!/usr/env/bin python
#coding=utf-8

import sys
import math
import time

import libs.PreProcess
import libs.Nlp

class NewWordExtractor(object):
    def __init__(self):
        # N元词组的N
        self.ngram = 2
        # N元词组组成的短语最小出现次数
        self.minPhraseNum = 50
        # N元词组的凝固度参数
        self.linkage = 2000
        # 左右邻的熵占最大熵占比
        self.entrory_threshold = 0.55
        self.nlp = libs.Nlp.Nlp()
        self.preprocess = libs.PreProcess.PreProcess()
        # 一元词袋
        self.singleWordbag = {}
        # N元词袋
        self.biWordbag = {}
        # 总词语数
        self.sumWords = 0.0
        # 总短语数
        self.sumPhrases = 0.0

    def queryTerms(self, query):
        query = self.preprocess.removePuncuation(query)
        return self.nlp.wordseg(query)

    def wordbag(self, filename):
        with open(filename, 'r') as fp:
            liLines = fp.readlines()
            fp.close()
        count = 0
        start_t = time.time()
        for line in liLines:
            count += 1
            if count % 100000 == 0:
                end_t = time.time()
                dur = end_t - start_t
                print "%d\t%d" % (count, dur)
                start_t = end_t
            #words = self.queryTerms(line.strip('\r\n'))
            words = line.strip('\r\n').split('\t')
            word_num = len(words)
            for idx in range(word_num):
                word = words[idx]
                if not self.singleWordbag.has_key(word):
                    self.singleWordbag[word] = 0
                self.singleWordbag[word] += 1
                self.sumWords += 1
                if idx < (word_num-self.ngram+1):
                    phrase = ""
                    tmp = []
                    for i in range(self.ngram):
                        phrase += words[idx+i]
                        tmp.append(words[idx+i])
                    phrase_key = "\t".join(tmp)
                    # word_next = words[idx+1]
                    # phrase = word + word_next
                    # phrase_key = "%s\t%s" % (word, word_next)
                    if not self.biWordbag.has_key(phrase):
                        self.biWordbag[phrase] = {"sum":0, "data":set(), "left":{}, "right":{}}
                    self.biWordbag[phrase]["sum"] += 1  # 短语词频加1
                    self.sumPhrases += 1
                    # if phrase_key not in self.biWordbag[phrase]["data"]:
                    self.biWordbag[phrase]["data"].add(phrase_key)   # 统计出现的排列组合
                    left_word = ""
                    right_word = ""
                    if idx > 0:
                        left_word = words[idx-1]
                        if not self.biWordbag[phrase]["left"].has_key(left_word):
                            self.biWordbag[phrase]["left"][left_word] = 0
                        self.biWordbag[phrase]["left"][left_word] += 1  # 统计左边词汇
                    if idx < (word_num-self.ngram):
                        right_word = words[idx+self.ngram]
                        if not self.biWordbag[phrase]["right"].has_key(right_word):
                            self.biWordbag[phrase]["right"][right_word] = 0
                        self.biWordbag[phrase]["right"][right_word] += 1    # 统计右边词汇
        return self.singleWordbag

    def findByLinkage(self):
        backupPhrases = {}
        for phrase, elem in self.biWordbag.items():
            if elem["sum"] < self.minPhraseNum:
                continue
            phrase_freq = elem["sum"] / self.sumPhrases
            liWordFreq = []
            for phrase_key in elem["data"]:
                words = phrase_key.split("\t")
                freq = 1.0
                for word in words:
                    freq = freq * (self.singleWordbag[word] / self.sumWords)
                # freq = (self.singleWordbag[words[0]] / self.sumWords) * (self.singleWordbag[words[1]] / self.sumWords)
                liWordFreq.append(freq)
            if (phrase_freq/min(liWordFreq)) >= self.linkage:
                #print "%s\t%d" % (phrase, elem["sum"])
                backupPhrases[phrase] = elem["sum"]
        return backupPhrases

    def getNeighborEntrory(self, dictPhrases, output_filename):
        wp = open(output_filename, 'w')
        liSorted = sorted(dictPhrases.items(), key=lambda x:x[1], reverse=True)
        for elem in liSorted:
            phrase = elem[0]
            freq = elem[1]
            dictLeft = self.biWordbag[phrase]["left"]
            dictRight = self.biWordbag[phrase]["right"]
            labels = ["left", "right"]
            infos = [dictLeft, dictRight]
            msg = ["%s\t%d" % (phrase, int(freq))]
            remove = False
            for info, label in zip(infos, labels):
                sum_num = sum(info.values()) * 1.0
                entrory = 0.0
                max_entrory = 0.0
                if len(info.values()) > 0:
                    max_entrory = math.log(len(info.values()))
                #msg.append("%.2f" % max_entrory)
                for key, value in info.items():
                    #print "%s\t%d" % (key, value)
                    entrory += (-1.0*value/sum_num) * (math.log(value/sum_num))
                #msg.append("%.2f" % entrory)
                entrory_percent = 0.0
                if max_entrory > 0:
                    entrory_percent = entrory / max_entrory
                msg.append("%.2f" % entrory_percent)
                if entrory_percent < self.entrory_threshold:
                    remove = True
            if not remove:
                print >> wp, "\t".join(msg)
        wp.close()

    def run(self, input_filename, output_filename):
        # 首先统计词袋信息，取出一元词频和二元词频，以及二元词频的左右词频
        self.wordbag(input_filename)
        print "sum word: %d" % int(self.sumWords)
        print "sum phrase: %d" % int(self.sumPhrases)
        # 取出凝固度较高的二元词组合
        dictPhrases = self.findByLinkage()
        # 获取左右邻接词的信息熵
        self.getNeighborEntrory(dictPhrases, output_filename)


if __name__ == "__main__":
    newWordExtractor = NewWordExtractor()
    newWordExtractor.run(sys.argv[1], sys.argv[2])

