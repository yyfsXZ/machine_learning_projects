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
tf.flags.DEFINE_string("test_file", hp.train_params["test_file"], "Data for the training data.")
tf.flags.DEFINE_string("model_path", hp.train_params["model_path"], "Path to save model")
tf.flags.DEFINE_string("pretrained_embedding_file", hp.train_params["pretrained_embedding_file"], "Pretrained embeddings for querys")
tf.flags.DEFINE_integer("embedding_size", hp.model_params["embedding_size"], "Size of embedding for token/position")
tf.flags.DEFINE_integer("batch_size", hp.train_params["batch_size"], "Batch size for validation")

FLAGS = tf.flags.FLAGS

train_data_helper = dh.TrainDataHelper(FLAGS.embedding_size)
train_data_helper.initialize(FLAGS.pretrained_embedding_file)

test_datas = train_data_helper.read_input_file(FLAGS.test_file, type="test")
test_data_size = len(test_datas)

test_batch_sum = test_data_size / FLAGS.batch_size
if test_data_size % FLAGS.batch_size != 0:
    test_batch_sum += 1

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("%s/model-%d.meta" % (FLAGS.model_path, model_id))
        saver.restore(sess, "%s/model-%d" % (FLAGS.model_path, model_id))

        input_x_1 = graph.get_operation_by_name("inputs_a").outputs[0]
        input_x_2 = graph.get_operation_by_name("inputs_b").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_rate = graph.get_operation_by_name("dropout_keep_rate").outputs[0]

        scores = graph.get_operation_by_name("output/scores").outputs[0]
        fc_outputs_a = graph.get_operation_by_name("fc_layer/outputs_a").outputs[0]
        fc_outputs_b = graph.get_operation_by_name("fc_layer/outputs_b").outputs[0]

        def predict_step(batch_ids_a, batch_ids_b, batch_labels):
            feed_dict = {
                input_x_1: batch_ids_a,
                input_x_2: batch_ids_b,
                # input_y: batch_labels,
                dropout_keep_rate: 1.0,
            }
            scores_, embs_a, embs_b = sess.run([scores, fc_outputs_a, fc_outputs_b], feed_dict=feed_dict)
            return scores_, embs_a, embs_b

        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        wp = open("predict_result.txt", 'w')
        test_batch_sum = test_data_size / FLAGS.batch_size
        if test_data_size % FLAGS.batch_size != 0:
            test_batch_sum += 1
        batches = train_data_helper.batch_iter(test_datas, FLAGS.batch_size, shuffle=False)
        begin = int(time.time() * 1000)
        for idx, batch in enumerate(batches):
            batch_ids_a, batch_ids_b, batch_labels = train_data_helper.trans_batch_to_inputs(batch)
            scores_, embs_a, embs_b = predict_step(np.array(batch_ids_a), np.array(batch_ids_b), np.array(batch_labels))
            batch_querys_a = [train_data_helper.get_query_by_id(data[0]) for data in batch]
            batch_querys_b = [train_data_helper.get_query_by_id(data[1]) for data in batch]
            for i in range(len(batch_querys_a)):
                query_a = batch_querys_a[i]
                query_b = batch_querys_b[i]
                exp = batch_labels[i]
                prob = scores_[i][1]
                print >> wp, "%s\t%s\t%d\t%f" % ("".join(query_a), "".join(query_b), exp, prob)
        sum_time = int(time.time()*1000) - begin
        avr_time = sum_time * 1.0 / test_data_size
        print "sum_data: %d, sum_time: %dms, avr_time: %.3fms" % (test_data_size, sum_time, avr_time)



