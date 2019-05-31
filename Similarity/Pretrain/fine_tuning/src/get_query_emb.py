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
tf.flags.DEFINE_integer("embedding_size", hp.model_params["embedding_size"], "Size of embedding for token/position")
tf.flags.DEFINE_string("pretrained_embedding_file", hp.train_params["pretrained_embedding_file"], "Pretrained embeddings for querys")
tf.flags.DEFINE_integer("batch_size", hp.train_params["batch_size"], "Batch size for validation")

FLAGS = tf.flags.FLAGS

train_data_helper = dh.TrainDataHelper(FLAGS.embedding_size)
train_data_helper.initialize(FLAGS.pretrained_embedding_file)

with open(FLAGS.kdb_file, 'r') as fp:
    querys = [line.strip('\r\n') for line in fp.readlines()]
    fp.close()

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("%s/model-%d.meta" % (FLAGS.model_path, model_id))
        saver.restore(sess, "%s/model-%d" % (FLAGS.model_path, model_id))

        inputs_a = graph.get_operation_by_name("inputs_a").outputs[0]
        dropout_keep_rate = graph.get_operation_by_name("dropout_keep_rate").outputs[0]

        fc_outputs_a = graph.get_operation_by_name("fc_layer/outputs_a").outputs[0]

        def predict_step(batch_ids_a):
            feed_dict = {
                inputs_a: batch_ids_a,
                dropout_keep_rate: 1.0,
            }
            outputs_a = sess.run([fc_outputs_a], feed_dict=feed_dict)
            return outputs_a

        wp = open("query_embedding.txt", 'w')

        batches = len(querys) / FLAGS.batch_size
        if len(querys) % FLAGS.batch_size != 0:
            batches += 1

        begin = int(time.time() * 1000)
        for batch_id in range(batches):
            start = batch_id * FLAGS.batch_size
            end = min((batch_id+1)*FLAGS.batch_size, len(querys))
            batch_query = querys[start:end]
            query_ids = [train_data_helper.get_id_by_query(query) for query in batch_query]
            input_ids = [train_data_helper.get_input_by_id(query_id) for query_id in query_ids]
            encoder_outputs = predict_step(input_ids)[0]
            for i, query in enumerate(batch_query):
                query_emb = [str(v) for v in encoder_outputs[i]]
                print >> wp, "%s\t%s" % (query, " ".join(query_emb))
        wp.close()

        avr = (int(time.time()*1000) - begin) * 1.0 / batches
        print "batch number=%d, avr time=%.3fms" % (batches, avr)



