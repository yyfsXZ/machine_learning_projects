#!/usr/env/bin python
#coding=utf-8

class Hyperparams:
    def __init__(self):
        self.train_params = {
            "train_file": "../prj/data/train.txt",
            "test_file": "../prj/data/test.txt",
            "valid_file": "../prj/data/valid.txt",
            "kdb_file": "../prj/data/kdb.txt",
            "model_path": "../prj/model/",
            "log_path": "../prj/log/",
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
