#!/usr/env/bin python
#coding=utf-8
"""
    @file   :           data_helper.py
    @Author :           Xiang Zhang
    @Date   :           2:52 PM 2018/12/17
    @Description:       Data helper for Machine Learning
"""

import numpy as np
from hyperparams import Hyperparams
import copy
import logging

def read_vocab_emb(emb_file, model_path):
    """
        @Author:            Xiang Zhang
        @Date:              5:51 PM 2019/3/22
        @Description:       get pretrained embeddings (not trainable)
    """
    word2id = {}
    word2id["<PAD>"] = 0
    word2id["<UNK>"] = 1
    word2id["<S>"] = 2
    word2id["</S>"] = 3
    with open(emb_file, 'r') as fp:
        line = fp.readline()
        params = line.strip('\r\n').split()
        sum_words = int(params[0])
        embedding_dim = int(params[1])
        embedding_matrix = np.random.randn(sum_words+4, embedding_dim)  # init embedding
        word_id = 4
        while line:
            splits = line.strip("\r\n").split()
            if len(splits) < (embedding_dim + 1):
                line = fp.readline()
                continue
            word2id[splits[0]] = word_id
            embedding_matrix[word_id] = [float(v) for v in splits[1:]]
            word_id += 1
            line = fp.readline()
        fp.close()

    id2word = {}
    vocab_file = "%s/word2id.txt" % model_path
    with open(vocab_file, 'w') as wp:
        for word, idx in word2id.items():
            print >> wp, "%s\t%d" % (word, idx)
            id2word[str(idx)] = word
        wp.close()
    logging.info("Finish reading Embedding file! %d words loaded" % (sum_words+4))
    return embedding_matrix, word2id, id2word


def create_vocab(train_file, model_path, output_freq=True):
    """
        @Author:            Xiang Zhang
        @Date:              2:31 PM 2018/12/25
        @Description:       create vocab list for train data
    """
    word2id = {}
    word2id["<PAD>"] = 0
    word2id["<UNK>"] = 1
    word2id["<S>"] = 2
    word2id["</S>"] = 3
    words_summary = set()
    words_count = {}
    with open(train_file, 'r') as fp:
        line = fp.readline()
        while line:
            splits = line.lower().strip('\r\n').split("\t")
            if len(splits) < 3:
                continue
            else:
                for sent in splits[:2]:
                    words = sent.split(" ")
                    for word in words:
                        words_summary.add(word)
                        if not words_count.has_key(word):
                            words_count[word] = 0
                        words_count[word] += 1
            line = fp.readline()
        fp.close()
    idx = 4

    for word in words_summary:
        if words_count.get(word, 0) >= Hyperparams.min_word_freq:
            word2id[word] = idx
            idx += 1
    id2word = {}
    vocab_file = "%s/word2id.txt" % model_path
    with open(vocab_file, 'w') as wp:
        sorted_words = sorted(word2id.items(), key=lambda x:x[1])
        for elem in sorted_words:
            word = elem[0]
            idx = elem[1]
            print >> wp, "%s\t%d" % (word, idx)
            id2word[str(idx)] = word
        wp.close()
    if output_freq:
        sorted_words = sorted(words_count.items(), key=lambda x:x[1], reverse=True)
        with open("%s/word_freq.txt" % model_path, 'w') as wp:
            print >> wp, "<PAD>\t10000000\n<UNK>\t10000000\n<S>\t10000000\n<\S>\t10000000"
            for elem in sorted_words:
                print >> wp, "%s\t%d" % (elem[0], elem[1])
            wp.close()
    return word2id, id2word

def create_vocab_emb(word2id, embedding_file, embedding_dim=Hyperparams.emb_size):
    """
        @Author:            Xiang Zhang
        @Date:              3:30 PM 2019/3/13
        @Description:       get pretrained words
    """
    embedding_matrix = np.random.randn(len(word2id.keys()), embedding_dim)
    with open(embedding_file, 'r') as fp:
        line = fp.readline()
        valid_emb = 0
        while line:
            splits = line.strip("\r\n").split()
            if len(splits) == embedding_dim+1 and word2id.has_key(splits[0]):
                word_id = word2id[splits[0]]
                embedding_matrix[word_id] = [float(v) for v in splits[1:]]
                valid_emb += 1
            line = fp.readline()
    logging.info("Null word embeddings: %d" % (len(word2id.keys()) - valid_emb))
    logging.info("Finish reading Embedding file !")
    return embedding_matrix

def create_data(input_file, word2id, keep_query=False):
    """
        @Author:            Xiang Zhang
        @Date:              2:28 PM 2018/12/25
        @Description:       convert sentence words to word ids
    """
    encode_querys, encode_ids, decode_querys, decode_ids, labels = [], [], [], [], []
    fp = open(input_file, 'r')
    line = fp.readline()
    idx = 0
    while line:
        splits = line.strip('\r\n').split("\t")
        if len(splits) < 3:
            line = fp.readline()
            continue
        encode_query_words = splits[0].split(" ")
        decode_query_words = splits[1].split(" ")
        if max(len(encode_query_words), len(decode_query_words)) >= Hyperparams.max_len:
            line = fp.readline()
            continue
        else:
            encode_query_words.extend(["</S>"])
            decode_query_words.extend(["</S>"])
            if keep_query or idx < 100:
                encode_querys.append(encode_query_words)
                decode_querys.append(decode_query_words)
            encode_ids.append([word2id.get(word, 1) for word in encode_query_words])
            decode_ids.append([word2id.get(word, 1) for word in decode_query_words])
            if float(splits[2]) == 1:
                label = [0, 1]
            else:
                label = [1, 0]
            label.extend([-1]*(Hyperparams.max_len-2))
            labels.append(label)
            line = fp.readline()
            idx += 1
    # Paddle
    for idx, (encode_input, decode_input) in enumerate(zip(encode_ids, decode_ids)):
        encode_ids[idx].extend([0]*(Hyperparams.max_len - len(encode_input)))
        decode_ids[idx].extend([0]*(Hyperparams.max_len - len(decode_input)))
    return encode_ids, decode_ids, encode_querys, decode_querys, labels

def batch_iter_old(query_ids_a, query_ids_b, label_list, batch_size, shuffle=True):
    """
        @Author:            Xiang Zhang
        @Date:              3:01 PM 2018/12/17
        @Description:       data iterator
    """
    query_ids_a_list = np.array(query_ids_a)
    query_ids_b_list = np.array(query_ids_b)
    label_list = np.array(label_list)
    data_size = len(query_ids_a_list)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_query_ids_a = query_ids_a_list[shuffle_indices]
        shuffled_query_ids_b = query_ids_b_list[shuffle_indices]
        shuffled_label = label_list[shuffle_indices]
    else:
        shuffled_query_ids_a = query_ids_a_list
        shuffled_query_ids_b = query_ids_b_list
        shuffled_label = label_list

    batch_per_epoch = data_size / batch_size
    # if data_size % batch_size != 0:
    #     batch_per_epoch += 1
    for batch_num in range(batch_per_epoch):
        end_idx = (batch_num+1) * batch_size
        # end_idx = min((batch_num+1)*batch_size, data_size)
        yield shuffled_query_ids_a[batch_num*batch_size:end_idx], shuffled_query_ids_b[batch_num*batch_size:end_idx], shuffled_label[batch_num*batch_size:end_idx]

def batch_iter(data, batch_size, shuffle=True):
    data_list = np.array(data)
    del data[:]
    data_size = len(data_list)
    batch_per_epoch = data_size / batch_size
    if data_size % batch_size != 0:
        batch_per_epoch += 1
    shuffle_indices = []
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data_list[shuffle_indices]
    else:
        shuffled_data = data_list
    for batch_num in range(batch_per_epoch):
        end_idx = min((batch_num+1)*batch_size, data_size)
        yield shuffled_data[batch_num*batch_size:end_idx]

def load_word2id(word2id_file):
    """
        @Author:            Xiang Zhang
        @Date:              4:13 PM 2018/12/28
        @Description:       load word2id.txt

    """
    with open(word2id_file, 'r') as fp:
        liLines = fp.readlines()
        fp.close()
    word2id = {}
    words = []
    for line in liLines:
        splits = line.strip("\r\n").split("\t")
        word2id[splits[0]] = int(splits[1])
        words.append(splits[0])
    return word2id, words

def read_kdb_querys(input_file, word2id):
    """
        @Author:            Xiang Zhang
        @Date:              2:28 PM 2018/12/25
        @Description:       convert sentence words to word ids(for convert kdb querys to tensors)
    """
    encode_querys, encode_ids, labels = [], [], []
    fp = open(input_file, 'r')
    line = fp.readline()
    while line:
        words = line.strip('\r\n').split(' ')
        if len(words) >= Hyperparams.max_len:
            line = fp.readline()
            continue
        else:
            words.extend(["</S>"])
        encode_querys.append(line.strip('\r\n'))
        encode_ids.append([word2id.get(word, 1) for word in words])
        label = [0, 1]
        label.extend([-1]*(Hyperparams.max_len-2))
        labels.append(label)
        line = fp.readline()
    # Paddle
    for idx, encode_id in enumerate(encode_ids):
        encode_ids[idx].extend([0]*(Hyperparams.max_len - len(encode_id)))
    return encode_querys, encode_ids, labels