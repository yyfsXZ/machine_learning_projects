#!/usr/env/bin python
#coding=utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import logging
import time

from hyperparams import Hyperparams
import data_helper as dh

prj_name = sys.argv[1]
model_id = int(sys.argv[2])

hp = Hyperparams(prj_name)
tf.flags.DEFINE_string("kdb_file", hp.train_params["kdb_file"], "Data for the training data.")
tf.flags.DEFINE_string("model_path", hp.train_params["model_path"], "Path to save model")
tf.flags.DEFINE_integer("max_seq_len", hp.model_params["max_seq_len"], "Max length for sequence")
tf.flags.DEFINE_integer("batch_size", hp.train_params["batch_size"], "Batch size for validation")

FLAGS = tf.flags.FLAGS

train_data_helper = dh.TrainDataHelper(FLAGS.max_seq_len)
train_data_helper.initialize()
vocab_size = train_data_helper.get_vocab_size() # 词汇量大小

with open(FLAGS.kdb_file, 'r') as fp:
    querys = [line.strip('\r\n').replace(" ", "") for line in fp.readlines()]
    fp.close()

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("%s/model-%d.meta" % (FLAGS.model_path, model_id))
        saver.restore(sess, "%s/model-%d" % (FLAGS.model_path, model_id))

        input_x_1 = graph.get_operation_by_name("input_ids_a").outputs[0]
        dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]

        encoder_outputs_a = graph.get_operation_by_name("outputs_a").outputs[0]

        def predict_step(batch_ids_a):
            feed_dict = {
                input_x_1: batch_ids_a,
                dropout_prob: 0.0,
            }
            encs_a = sess.run([encoder_outputs_a], feed_dict=feed_dict)
            return encs_a

        wp = open("query_embedding.txt", 'w')

        batches = len(querys) / FLAGS.batch_size
        if len(querys) % FLAGS.batch_size != 0:
            batches += 1

        begin = int(time.time() * 1000)
        for batch_id in range(batches):
            start = batch_id * FLAGS.batch_size
            end = min((batch_id+1)*FLAGS.batch_size, len(querys))
            batch_query = querys[start:end]
            query_ids = [train_data_helper.trans_query_to_input_id(query) for query in batch_query]
            input_ids = [train_data_helper.get_input_ids(query_id) for query_id in query_ids]
            encoder_outputs = predict_step(input_ids)[0]
            for i, query in enumerate(batch_query):
                query_emb = [str(v) for v in encoder_outputs[i]]
                print >> wp, "%s\t%s" % (query, " ".join(query_emb))
        wp.close()

        avr = (int(time.time()*1000) - begin) * 1.0 / batches
        print "batch number=%d, avr time=%.3fms" % (batches, avr)



