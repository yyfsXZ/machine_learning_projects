#!/usr/env/bin python
#coding=utf-8

import os
import sys
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../.."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../../.."))

from CommonLibs.FileIoUtils import MySentences

class TrainDataHelper:
    def __init__(self, d_embedding):
        self.query2id = {}
        self.id2query = {}
        self.id2input = {}
        self.d_embedding = d_embedding

    def initialize(self, query_embedding_file):
        sentences = MySentences(query_embedding_file)
        query_id = 0
        for sentence in sentences:
            splits = sentence.strip('\r\n').split('\t')
            if len(splits) < 2:
                continue
            query = splits[0]
            if self.query2id.has_key(query):
                continue
            embedding = [float(v) for v in splits[1].split(" ")]
            if len(embedding) < self.d_embedding:
                continue
            self.query2id[query] = query_id
            self.id2query[str(query_id)] = query
            self.id2input[str(query_id)] = embedding
            query_id += 1
        logging.info("Load %d querys" % query_id)

    def get_input_by_id(self, query_id):
        return self.id2input.get(str(query_id), [])

    def get_query_by_id(self, query_id):
        return self.id2query.get(str(query_id), "")

    def get_id_by_query(self, query):
        return self.query2id.get(query, -1)

    def read_input_file(self, filename, type="train"):
        sentences = MySentences(filename)
        datas = []
        for sentence in sentences:
            splits = sentence.split('\t')
            if len(splits) < 3:
                continue
            query_id_a = self.get_id_by_query(splits[0])
            query_id_b = self.get_id_by_query(splits[1])
            if query_id_a < 0 or query_id_b < 0:
                continue
            one_data = [query_id_a, query_id_b]
            label = int(float(splits[2]))
            label = [0, 1] if label==1 else [1, 0]
            one_data.extend(label)
            datas.append(one_data)
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

    def trans_batch_to_inputs(self, batch_data):
        inputs_a = [self.get_input_by_id(elem[0]) for elem in batch_data]
        inputs_b = [self.get_input_by_id(elem[1]) for elem in batch_data]
        labels = [0 if elem[2]==1 else 1 for elem in batch_data]
        return inputs_a, inputs_b, labels
