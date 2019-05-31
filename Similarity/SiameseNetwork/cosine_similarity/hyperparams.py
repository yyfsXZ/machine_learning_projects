#!/usr/env/bin python
#coding=utf-8

"""
    @file   :           hyperparams.py
    @Author :           Xiang Zhang
    @Date   :           3:12 PM 2018/12/25
    @Description:       params for model
"""

class Hyperparams:
    train_file = "../data/train.txt"
    valid_file = "../data/valid.txt"
    test_file = "../data/test.txt"
    kdb_querys = "../data/kdb.txt"  # kdb querys

    # pretrained_embedding
    embedding_file = "../wdic/Tencent_AILab_ChineseEmbedding.txt"
    embedding_type = 1 # 1: trainable=true, 0: trainable=false
    emb_size = 200  # embedding dim

    # model path
    model_path = "../model"

    # log path
    log_dir = "../log"

    # training
    batch_size = 32
    learning_rate = 0.0001
    accu_threshold = 0.1

    # test
    test_batch_size = 32

    # model
    max_len = 20    # Maximum number of words in a sentence.
    min_word_freq = 1  # words whose occurred less than min_cnt are encoded as <UNK>.
    num_blocks = 5  # number of encoder/decoder blocks
    num_epochs = 30
    num_heads = 5   # number for multi-head self-attention
    dropout_keep_rate = 0.5
    hidden_size = emb_size  # hidden size (must equal to emb_size)
    ff_size = 1024   # feed_forword node size
    fc_size = 1024   # fc layer node size
