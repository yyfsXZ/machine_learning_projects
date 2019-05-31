#!/usr/env/bin python
#coding=utf-8

import os
import sys
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../.."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../../.."))

from CommonLibs.DataHelper import CharHelper
from CommonLibs.FileIoUtils import MySentences

class TrainDataHelper:
    def __init__(self, max_seq_length):
        # char文件工具
        self.char_helper = CharHelper()
        # sequence长度
        self.max_seq_length = max_seq_length
        # query到input的映射
        self.query2input = {}
        # query到id的映射
        self.query2id = {}
        # id到query的映射
        self.id2query = {}
        # 用户存储下一个query的id
        self.next_query_id = 0

    def initialize(self):
        self.char_helper.initialize()

    def get_vocab_size(self):
        return self.char_helper.get_vocab_size()

    def trans_query_to_input_id(self, query):
        if self.query2id.has_key(query):
            query_id = self.query2id[query]
        else:
            # query不存在列表中，存储query相关信息到内存中
            query_id = self.next_query_id   # 新增query递增query_id
            self.query2id[query] = query_id
            self.id2query[str(query_id)] = query
            self.next_query_id += 1
            ids = self.char_helper.trans_query_to_ids(query)
            # paddle
            if len(ids) < self.max_seq_length:
                ids.extend([self.char_helper.get_padd_char_id()] * (self.max_seq_length-len(ids)))
            else:
                ids = ids[:self.max_seq_length]
            self.query2input[query] = ids
        return query_id

    def read_input_file(self, filename, type="train"):
        sentences = MySentences(filename)
        datas = []
        for sentence in sentences:
            try:
                splits = sentence.split('\t')
                if len(splits) < 3:
                    continue

                query_a = splits[0]
                query_b = splits[1]
                one_data = [self.trans_query_to_input_id(query_a), self.trans_query_to_input_id(query_b)]
                label = int(float(splits[2]))
                label = [0, 1] if label==1 else [1, 0]
                one_data.extend(label)
                datas.append(one_data)
            except:
                continue
        logging.info("Load %s data done, size=%d", type, len(datas))
        return datas

    def batch_iter(self, data, batch_size, shuffle=True):
        data_list = np.array(data)
        data_size = len(data_list)
        batch_per_epoch = data_size / batch_size
        if data_size % batch_size != 0:
            batch_per_epoch += 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data_list[shuffle_indices]
        else:
            shuffled_data = data_list
        for batch_num in range(batch_per_epoch):
            end_idx = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[batch_num*batch_size:end_idx]

    def get_input_ids(self, query_id):
        return self.query2input[self.id2query[str(query_id)]]

    def trans_id_to_query(self, query_id):
        return self.id2query[str(query_id)]

    def save_vocab_file(self, filename):
        with open(filename, 'w') as wp:
            for id in range(self.char_helper.get_vocab_size()):
                print >> wp, "%s\t%d" % (self.char_helper.get_char_by_id(str(id)).encode("utf-8"), id)

if __name__ == "__main__":
    helper = TrainDataHelper(80)
    helper.initialize()
    for id in range(helper.char_helper.get_vocab_size()):
        print "%s\t%d" % (helper.char_helper.get_char_by_id(str(id)).encode("utf-8"), id)



