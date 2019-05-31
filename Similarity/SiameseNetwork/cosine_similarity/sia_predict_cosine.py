#!/usr/env/bin python
#coding=utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import logging

from hyperparams import Hyperparams as hp
import data_helper

tf.flags.DEFINE_string("test_file", hp.test_file, "Data for the training data.")
tf.flags.DEFINE_string("model_path", hp.model_path, "Path to save model")
tf.flags.DEFINE_integer("max_seq_len", hp.max_len, "Max length for sequence")
tf.flags.DEFINE_integer("test_batch_size", hp.test_batch_size, "Batch size for validation")

FLAGS = tf.flags.FLAGS

def examples(querys_a, ids_a, querys_b, ids_b, labels):
    logging.info("Load datas like this:\n")
    for query_a, id_a, query_b, id_b, label in zip(querys_a, ids_a, querys_b, ids_b, labels):
        print "query_a:\t%s" % " ".join(query_a)
        print "id_a:\t%s" % " ".join([str(id_) for id_ in id_a])
        print "query_b:\t%s" % " ".join(query_b)
        print "id_b:\t%s" % " ".join([str(id_) for id_ in id_b])
        if label[0] == 1:
            label_ = 1
        else:
            label_ = 0
        print "label:\t%d" % label_
        print "*" * 100

# Load word2id
word2id_file = "%s/word2id.txt" % FLAGS.model_path
word2id, words = data_helper.load_word2id(word2id_file)

# Load test file
test_ids_a, test_ids_b, test_querys_a, test_querys_b, test_labels = data_helper.create_data(FLAGS.test_file, word2id, keep_query=True)
test_data_size = len(test_ids_a)
print "Load test file done, size=%d" % test_data_size
examples(test_querys_a[:3], test_ids_a[:3], test_querys_b[:3], test_ids_b[:3], test_labels[:3])

test_batch_sum = test_data_size / FLAGS.test_batch_size
if test_data_size % FLAGS.test_batch_size != 0:
    test_batch_sum += 1

model_id = sys.argv[1]
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("%s/model-%d.meta" % (FLAGS.model_path, model_id))
        saver.restore(sess, "%s/model-%d" % (FLAGS.model_path, model_id))

        input_x_1 = graph.get_operation_by_name("input_query_a").outputs[0]
        input_x_2 = graph.get_operation_by_name("input_query_b").outputs[0]
        input_y = graph.get_operation_by_name("input_Y").outputs[0]
        dropout_keep_rate = graph.get_operation_by_name("dropout_keep_rate").outputs[0]

        scores = graph.get_operation_by_name("output/scores").outputs[0]

        def predict_step(batch_ids_a, batch_ids_b, batch_labels):
            feed_dict = {
                input_x_1: batch_ids_a,
                input_x_2: batch_ids_b,
                input_y: batch_labels,
                dropout_keep_rate: 1.0,
            }
            probs = sess.run([scores], feed_dict=feed_dict)
            return probs

        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        wp = open("predict_result.txt", 'w')
        test_batch_sum = test_data_size / FLAGS.test_batch_size
        if test_data_size % FLAGS.test_batch_size != 0:
            test_batch_sum += 1
        for idx in range(test_batch_sum):
            end_idx = min((idx+1)*FLAGS.test_batch_size, test_data_size)
            batch_ids_a, batch_ids_b, batch_labels, batch_querys_a, batch_querys_b = \
                test_ids_a[idx*FLAGS.test_batch_size:end_idx], \
                test_ids_b[idx*FLAGS.test_batch_size:end_idx], \
                test_labels[idx*FLAGS.test_batch_size:end_idx], test_querys_a[idx*FLAGS.test_batch_size:end_idx], test_querys_b[idx*FLAGS.test_batch_size:end_idx]
            batch_labels = np.array(batch_labels)[:, 0]
            probs = predict_step(np.array(batch_ids_a), np.array(batch_ids_b), np.array(batch_labels))
            expect = tf.argmax(batch_labels, 1)
            for i in range(len(batch_querys_a)):
                query_a = batch_querys_a[i][:-1]
                query_b = batch_querys_b[i][:-1]
                exp = batch_labels[i]
                prob = probs[i]
                print >> wp, "%s\t%s\t%d\t%f" % ("".join(query_a), "".join(query_b), exp, prob)




