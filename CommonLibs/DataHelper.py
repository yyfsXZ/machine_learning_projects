#!/usr/env/bin python
#coding=utf-8

"""
    @file   :           DataHelper.py
    @Author :           Xiang Zhang
    @Date   :           6:52 PM 2019/5/30
    @Description:       data process apis
"""

import sys
import re
import logging

from third_party.lang_conv.LangConv import LangConv
from CommonLibs.FileIoUtils import MySentences
from third_party.vocabs import Vocab

# 段落切分成多句的标点符号
para_split_puncs = ["!", "?", ";", "。", "？", "！", "；"]

def utf2uni(query):
    try:
        return query.decode("utf-8")
    except:
        return query

def uni2utf(query):
    try:
        return query.encode("utf-8")
    except:
        return query

class QueryPreprocess:
    def __init__(self):
        pass

    def tradition_to_simple(self, query):
        """
            @Author:            Xiang Zhang
            @Date:              12:02 PM 2019/4/24
            @Description:       繁转简

        """
        return LangConv.tradition_to_simple(query)

    def strQ2B(self, query):
        """
            @Author:            Xiang Zhang
            @Date:              12:04 PM 2019/4/24
            @Description:       全角转半角

        """
        elems = []
        uni_query = utf2uni(query)
        for char in uni_query:
            char_num = ord(char)
            if char_num == 0x3000 or char_num == 8291:
                char_num = 32
            elif char_num >= 0xFF01 and char_num <= 0xff5E:
                char_num -= 0xfee0
            char_num = unichr(char_num)
            elems.append(char_num)
        return uni2utf("".join(elems))

    def remove_baidu_baike_symbo(self, query):
        return query.replace(r"baike.baidu.com/item/", "")

    def parse(self, query):
        query = self.remove_baidu_baike_symbo(query)
        query = self.strQ2B(query)
        return self.tradition_to_simple(query)

class SplitParagraphToSent:
    def __init__(self):
        self.__black_tags = [u"(已完成)"]
        self.__l2_tag_max_len = 8
        self.__l2_tag_min_len = 1
        self.__split_puncs = re.compile("[%s]" % "".join(para_split_puncs).decode("utf-8"))

    def split_para(self, paragraph):
        """
            @Author:            Xiang Zhang
            @Date:              6:54 PM 2019/5/30
            @Description:       段落切分

        """
        sents = re.split(self.__split_puncs, paragraph.decode("utf-8"))
        results = []
        l2_tag = "NULL"
        for sent in sents:
            if sent.replace(" ", "") == "":
                continue
            while ord(sent[0]) == 32:
                sent = sent[1:]
            elems = sent.split(" ")
            if len(elems) > 1 and len(elems[0]) <= self.__l2_tag_max_len and len(elems[0]) > self.__l2_tag_min_len and elems[0] not in self.__black_tags:
                l2_tag = elems[0]
                elems = elems[1:]
            sent = "%s\t%s" % (l2_tag, "".join(elems))
            results.append(sent.encode("utf-8"))
        return results

class GenerateWordemb:
    def __init__(self, embedding_file):
        self.__sentences = MySentences(embedding_file)
        self.__word_num = 0
        self.__embedding_size = 0
        self.__embeddings = {}
        self.__words = []

    def initialize(self):
        for sent in self.__sentences:
            splits = sent.split()
            if len(splits) == 2:
                self.__word_num = int(splits[0])
                self.__embedding_size = int(splits[1])
                logging.info("embedding_dim=%d, word_number=%d" % (self.__embedding_size, self.__word_num))
            elif len(splits) >= self.__embedding_size+1:
                word = splits[0]
                vec = [float(v) for v in splits[1:self.__embedding_size+1]]
                self.__embeddings[utf2uni(word)] = vec

    def get_word_emb(self, word, code="unicode"):
        if code != "unicode":
            word = word.decode(code)
        return self.__embeddings.get(word, [0.0 for i in xrange(self.__embedding_size)])


class CharHelper:
    def __init__(self):
        self._unknown_char = "[UNK]"
        self._padd_char = "[PAD]"
        self._char_list = []
        self._char2id = {}
        self._id2char = {}
        self._vocab_size = 0

    def initialize(self):
        self._char_list.append(self._unknown_char)
        self._char2id[self._unknown_char] = 0
        self._id2char['0'] = self._unknown_char

        self._char_list.append(self._padd_char)
        self._char2id[self._padd_char] = 1
        self._id2char['1'] = self._padd_char

        for char_id, char in enumerate(Vocab.vocab):
            char_id += 2
            uni_char = utf2uni(char)
            self._char_list.append(uni_char)
            self._char2id[uni_char] = char_id
            self._id2char[str(char_id)] = uni_char
        self._vocab_size = len(self._char_list)
        logging.info("Init CharHelper done, vocab size=%d" % self._vocab_size)

    def get_id_by_char(self, char):
        return self._char2id.get(char, 0)

    def get_char_by_id(self, id):
        return self._id2char.get(str(id), self._unknown_char)

    def trans_query_to_ids(self, query):
        return [self.get_id_by_char(char) for char in list(utf2uni(query))]

    def get_padd_char_id(self):
        return self._char2id[self._padd_char]

    def get_vocab_size(self):
        return self._vocab_size


if __name__ == "__main__":
    query_preprocess_handler = QueryPreprocess()
    para_handler = SplitParagraphToSent()
    content_id = 0
    print "content_id\ttag_l1\ttag_l2\tsentence"
    for line in sys.stdin:
        content_id += 1
        try:
            splits = line.strip('\r\n').split('\t')
            if len(splits) < 2:
                continue
            tag_l1 = splits[1]
            line = splits[0]
            line = query_preprocess_handler.parse(line)
            sents = para_handler.split_para(line)
            for sent in sents:
                print "%d\t%s\t%s" % (content_id, tag_l1, sent)
        except:
            continue
