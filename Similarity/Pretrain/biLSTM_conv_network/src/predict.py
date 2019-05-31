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
tf.flags.DEFINE_integer("max_seq_len", hp.model_params["max_seq_len"], "Max length for sequence")
tf.flags.DEFINE_integer("batch_size", hp.train_params["batch_size"], "Batch size for validation")

FLAGS = tf.flags.FLAGS

def examples(querys_a, ids_a, querys_b, ids_b, labels):
    logging.info("Load datas like this:\n")
    for query_a, id_a, query_b, id_b, label in zip(querys_a, ids_a, querys_b, ids_b, labels):
        print "query_a:\t%s" % " ".join(query_a)
        print "id_a:\t%s" % " ".join([str(id_) for id_ in id_a])
        print "query_b:\t%s" % " ".join(query_b)
        print "id_b:\t%s" % " ".join([str(id_) for id_ in id_b])
        if label[0] == 1:
            label_ = 0
        else:
            label_ = 1
        print "label:\t%d" % label_
        print "*" * 100

train_data_helper = dh.TrainDataHelper(FLAGS.max_seq_len)
train_data_helper.initialize()
vocab_size = train_data_helper.get_vocab_size() # 词汇量大小

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

        input_x_1 = graph.get_operation_by_name("input_ids_a").outputs[0]
        input_x_2 = graph.get_operation_by_name("input_ids_b").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]

        predictions = graph.get_operation_by_name("loss/predictions").outputs[0]
        softmax_scores = graph.get_operation_by_name("probs").outputs[0]
        encoder_outputs_a = graph.get_operation_by_name("outputs_a").outputs[0]
        encoder_outputs_b = graph.get_operation_by_name("outputs_b").outputs[0]

        def predict_step(batch_ids_a, batch_ids_b, batch_labels):
            feed_dict = {
                input_x_1: batch_ids_a,
                input_x_2: batch_ids_b,
                # input_y: batch_labels,
                dropout_prob: 0.0,
            }
            preds, probs, encs_a, encs_b = sess.run([predictions, softmax_scores, encoder_outputs_a, encoder_outputs_b], feed_dict=feed_dict)
            return preds, probs, encs_a, encs_b

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
            batch_ids_a = [train_data_helper.get_input_ids(data[0]) for data in batch]
            batch_ids_b = [train_data_helper.get_input_ids(data[1]) for data in batch]
            batch_labels = np.array([data[2:] for data in batch])
            preds, probs, encs_a, encs_b = predict_step(np.array(batch_ids_a), np.array(batch_ids_b), np.array(batch_labels))
            expect = tf.argmax(batch_labels, 1)
            # print "*" * 100
            # print expect
            # print preds
            # for exp, res in zip(expect, preds):
            #     if exp==1 and res==1:
            #         tp += 1
            #     elif exp==1 and res==0:
            #         fn += 1
            #     elif exp==0 and res==1:
            #         fp += 1
            #     elif exp==0 and res==0:
            #         tn += 1
            batch_querys_a = [train_data_helper.trans_id_to_query(data[0]) for data in batch]
            batch_querys_b = [train_data_helper.trans_id_to_query(data[1]) for data in batch]
            for i in range(len(batch_querys_a)):
                query_a = batch_querys_a[i]
                query_b = batch_querys_b[i]
                exp = 0
                if batch_labels[i][1] == 1:
                    exp = 1
                prob = probs[i][1]
                print >> wp, "%s\t%s\t%d\t%f" % ("".join(query_a), "".join(query_b), exp, prob)
        sum_time = int(time.time()*1000) - begin
        avr_time = sum_time * 1.0 / test_data_size
        print "sum_data: %d, sum_time: %dms, avr_time: %.3fms" % (test_data_size, sum_time, avr_time)



