#!/usr/env/bin python
#coding=utf-8

class Hyperparams:
    def __init__(self, prj_name):
        self.prj_name = prj_name
        self.train_params = {
            "train_file": "../%s/data/train.txt" % self.prj_name,
            "test_file": "../%s/data/test.txt" % self.prj_name,
            "valid_file": "../%s/data/valid.txt" % self.prj_name,
            "pretrained_embedding_file": "../%s/data/pretrained_embedding.txt" % self.prj_name,
            "kdb_file": "../%s/data/kdb.txt" % self.prj_name,
            "model_path": "../%s/model" % self.prj_name,
            "log_path": "../%s/log/" % self.prj_name,
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