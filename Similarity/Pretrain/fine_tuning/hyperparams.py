#!/usr/env/bin python
#coding=utf-8

class Hyperparams:
    def __init__(self):
        self.train_params = {
            "train_file": "../data/train.txt",
            "test_file": "../data/test.txt",
            "valid_file": "../data/valid.txt",
            "pretrained_embedding_file": "../data/pretrained_embedding.txt",
            "kdb_file": "../data/kdb.txt",
            "model_path": "../model",
            "log_path": "../log/",
            "learning_rate": 0.00001,
            "epochs": 50,
            "batch_size": 96,
            "dropout_keep_rate": 0.5,
            "accu_threshold": 0.1,
            "num_checkpoints": 5,
        }

        self.model_params = {
            "embedding_size": 768,
            "d_hidden": [256], # 隐藏层的节点数，list长度为层数
            "d_fc": 256,     # full-connect层节点数，list长度为层数
        }