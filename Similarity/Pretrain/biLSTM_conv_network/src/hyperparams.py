#!/usr/env/bin python
#coding=utf-8

class Hyperparams:
    def __init__(self, prj_name):
        self.prj_name = prj_name
        self.train_params = {
            "train_file": "../%s/data/train.txt" % self.prj_name,
            "test_file": "../%s/data/test.txt" % self.prj_name,
            "valid_file": "../%s/data/valid.txt" % self.prj_name,
            "kdb_file": "../%s/data/kdb.txt" % self.prj_name,
            "model_path": "../%s/model/" % self.prj_name,
            "log_path": "../%s/log/" % self.prj_name,
            "learning_rate": 0.00001,
            "epochs": 20,
            "batch_size": 96,
            "dropout_rate": 0.5,
            "accu_threshold": 0.1,
            "num_checkpoints": 5,
        }

        self.model_params = {
            "embedding_size": 200,
            "max_seq_len": 80,
            "d_hidden_lstm": [256], # lstm层的节点数，list长度为层数
            "d_hidden_conv": [[2, 256, 1], [3, 256, 1], [4, 256, 1]],   # convolution层参数，list长度决定卷积核个数。list每个元素的含义为：[kernel_height, channels, stride]
            "d_fc": [1024],     # full-connect层节点数，list长度为层数
        }
