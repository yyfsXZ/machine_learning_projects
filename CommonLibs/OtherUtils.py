#!/usr/env/bin python
#coding=utf-8

import os
import sys
import logging

def initPaths(model_path, log_path):
    if not os.path.exists(model_path):
        os.system("mkdir %s" % model_path)
    # 训练日志
    train_log = log_path + "/train.log"
    # tensorboard日志
    tensorboard_log_path = log_path + "/tensorboard"
    if not os.path.exists(tensorboard_log_path):
        os.mkdir(tensorboard_log_path)
    os.system("rm %s/*" % tensorboard_log_path)

    fmt = '%(asctime)s %(name)s %(filename)s(%(funcName)s[line:%(lineno)d]) %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=train_log,
                        filemode='a'
                        )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)