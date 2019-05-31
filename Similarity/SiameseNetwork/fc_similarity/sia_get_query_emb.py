#!/usr/env/bin python
#coding=utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import logging

from hyperparams import Hyperparams as hp
import data_helper

tf.flags.DEFINE_string("kdb_querys", hp.kdb_querys, "Data in kdb")
tf.flags.DEFINE_string("model_path", hp.model_path, "Path to save model")
tf.flags.DEFINE_integer("max_seq_len", hp.max_len, "Max length for sequence")
tf.flags.DEFINE_integer("test_batch_size", hp.test_batch_size, "Batch size for validation")

FLAGS = tf.flags.FLAGS

def examples(querys, ids, labels):
    print "*" * 100
    print "Load datas like this:"
    for query, id, label in zip(querys, ids, labels):
        print "query:\t%s" % query
        print "ids:\t%s" % " ".join([str(id_) for id_ in ids])
        print "*" * 100

# Load word2id
word2id_file = "%s/word2id.txt" % FLAGS.model_path
word2id, words = data_helper.load_word2id(word2id_file)

# Load test file
querys, ids, labels = data_helper.read_kdb_querys(FLAGS.kdb_querys, word2id)
data_size = len(ids)
print "Load kdb file done, size=%d" % data_size

batch_sum = data_size / FLAGS.test_batch_size
if data_size % FLAGS.test_batch_size != 0:
    batch_sum += 1

model_id = 57
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("%s/model-%d.meta" % (FLAGS.model_path, model_id))
        saver.restore(sess, "%s/model-%d" % (FLAGS.model_path, model_id))

        input_x_1 = graph.get_operation_by_name("input_query_a").outputs[0]
        dropout_keep_rate = graph.get_operation_by_name("dropout_keep_rate").outputs[0]

        encoder_output_1 = graph.get_operation_by_name("reduce/encoder_output_a").outputs[0]

        def predict_step(batch_ids_a):
            feed_dict = {
                input_x_1: batch_ids_a,
                dropout_keep_rate: 1.0,
            }
            query_output = sess.run([encoder_output_1], feed_dict=feed_dict)
            return query_output

        wp = open("query_embedding.txt", 'w')
        batch_sum = data_size / FLAGS.test_batch_size
        if data_size % FLAGS.test_batch_size != 0:
            batch_sum += 1
        for idx in range(batch_sum):
            end_idx = min((idx+1)*FLAGS.test_batch_size, data_size)
            batch_ids_a, batch_querys_a = ids[idx*FLAGS.test_batch_size:end_idx], querys[idx*FLAGS.test_batch_size:end_idx]
            encoder_outputs = predict_step(np.array(batch_ids_a))[0]
            for i in range(len(batch_querys_a)):
                query = batch_querys_a[i]
                query_emb = [str(v) for v in encoder_outputs[i]]
                print >> wp, "%s\t%s" % (query, " ".join(query_emb))




