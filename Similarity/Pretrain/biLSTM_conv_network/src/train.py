#!/ur/env/bin python
#coding=utf-8

import os
import sys
import logging
import tensorflow as tf
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../.."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../../.."))

from bi_lstm_conv_network import BILSTM_CONV_NETWORK
import data_helper as dh
from hyperparams import Hyperparams

from CommonLibs import ModelUtils
from CommonLibs import Metrix
from CommonLibs import OtherUtils

prj_name = sys.argv[1]
hp = Hyperparams(prj_name)
tf.flags.DEFINE_string("train_file", hp.train_params["train_file"], "Data for the training data.")
tf.flags.DEFINE_string("valid_file", hp.train_params["valid_file"], "Data for validation")

tf.flags.DEFINE_string("model_path", hp.train_params["model_path"], "Path to save model")
tf.flags.DEFINE_string("log_path", hp.train_params["log_path"], "Path to save log")
tf.flags.DEFINE_integer("epochs", hp.train_params["epochs"], "Numbers of train epoch")
tf.flags.DEFINE_float("dropout_rate", hp.train_params["dropout_rate"], "Dropout rate")
tf.flags.DEFINE_float("accu_threshold", hp.train_params["accu_threshold"], "Accurate score threshold")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "allow_soft_placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "log_device_placement")
tf.flags.DEFINE_integer("num_checkpoints", hp.train_params["num_checkpoints"], "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("batch_size", hp.train_params["batch_size"], "Batch size for training")
tf.flags.DEFINE_float("learning_rate", hp.train_params["learning_rate"], "Learning rate")

tf.flags.DEFINE_integer("embedding_size", hp.model_params["embedding_size"], "Size of embedding for token/position")
tf.flags.DEFINE_integer("max_seq_len", hp.model_params["max_seq_len"], "Max length for sequence")
dims_hidden_lstm = hp.model_params["d_hidden_lstm"]
dims_hidden_conv = hp.model_params["d_hidden_conv"]
dims_fc = hp.model_params["d_fc"]

FLAGS = tf.flags.FLAGS


def train():
    # 初始化日志和模型路径
    OtherUtils.initPaths(FLAGS.model_path, FLAGS.log_path)

    # 初始化输入文件
    train_data_helper = dh.TrainDataHelper(FLAGS.max_seq_len)
    train_data_helper.initialize()
    vocab_size = train_data_helper.get_vocab_size() # 词汇量大小

    train_datas = train_data_helper.read_input_file(FLAGS.train_file, type="train")
    train_data_size = len(train_datas)
    valid_datas = train_data_helper.read_input_file(FLAGS.valid_file, type="valid")
    valid_data_size = len(valid_datas)

    logging.info("Train start")

    # Build a graph and rnn object
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            gpu_options=gpu_options,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = BILSTM_CONV_NETWORK(vocab_size=vocab_size,
                                        seq_length=FLAGS.max_seq_len,
                                        d_emb=FLAGS.embedding_size,
                                        d_hidden_lstm=dims_hidden_lstm,
                                        d_hidden_conv=dims_hidden_conv,
                                        d_fc=dims_fc)
            model.build()
            # 获取train_operator
            train_op = ModelUtils.train_step(model.loss, FLAGS.learning_rate, model.global_step, decay=False)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            def train_step(batch_ids_a, batch_ids_b, batch_labels):
                batch_labels = np.array(batch_labels)
                feed_dict = {
                    model.input_a: batch_ids_a,
                    model.input_b: batch_ids_b,
                    model.input_y: batch_labels,
                    model.dropout_prob: FLAGS.dropout_rate,
                }
                _, loss, scores, preds, labels = sess.run(
                    [train_op, model.loss, model.probs, model.predictions, model.labels], feed_dict=feed_dict
                )
                print "actucal-{} predict-{}".format(labels[:32], preds[:32])
                tp, fp, tn, fn = Metrix.get_accu(scores[:, 1], batch_labels[:, 1], FLAGS.accu_threshold)
                return loss, tp, fp, tn, fn

            def validation_step(batch_ids_a, batch_ids_b, batch_labels):
                batch_labels = np.array(batch_labels)
                feed_dict = {
                    model.input_a: batch_ids_a,
                    model.input_b: batch_ids_b,
                    model.input_y: batch_labels,
                    model.dropout_prob: 0.0,
                }
                loss, scores = sess.run(
                    [model.softmax_score_losses, model.probs],
                    feed_dict=feed_dict
                )
                tp, fp, tn, fn = Metrix.get_accu(scores[:, 1], batch_labels[:, 1], FLAGS.accu_threshold)
                return loss, tp, fp, tn, fn

            with tf.device("/gpu:0"):
                batch_per_epoch = train_data_size / FLAGS.batch_size
                if train_data_size % FLAGS.batch_size != 0:
                    batch_per_epoch += 1
                valid_batch_sum = valid_data_size / FLAGS.batch_size
                if valid_data_size % FLAGS.batch_size != 0:
                    valid_batch_sum += 1

                best_val_loss = 1000
                best_val_accu = 0.0
                best_val_recall = 0.0
                best_val_prec = 0.0
                best_val_f1 = -1
                best_epoch = -1
                for epoch in range(FLAGS.epochs):
                    total_loss = 0.0
                    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
                    batches = train_data_helper.batch_iter(train_datas, FLAGS.batch_size, shuffle=True)
                    for idx, batch in enumerate(batches):
                        batch_ids_a = [train_data_helper.get_input_ids(data[0]) for data in batch]
                        batch_ids_b = [train_data_helper.get_input_ids(data[1]) for data in batch]
                        batch_labels = [data[2:] for data in batch]
                        _loss, _tp, _fp, _tn, _fn = train_step(batch_ids_a, batch_ids_b, batch_labels)
                        total_loss += _loss
                        tp += _tp
                        tn += _tn
                        fp += _fp
                        fn += _fn
                        if idx !=0 and idx % (batch_per_epoch/10) == 0:
                            tmp_loss = total_loss / idx
                            tmp_accu = (tp+tn) / (tp+tn+fp+fn)
                            per = idx/(batch_per_epoch/10)
                            mess = "Epoch: %d, percent: %d0%%, loss: %f, accu: %f" % (epoch, per, tmp_loss, tmp_accu)
                            logging.info(mess)
                            logging.info("Epoch: %d, percent: %d0%%, tp=%d, tn=%d, fp=%d, fn=%d" % (epoch, per, int(tp), int(tn), int(fp), int(fn)))


                    total_loss = total_loss / batch_per_epoch
                    accu = (tp+tn) / (tp+tn+fp+fn)
                    mess = "Epoch %d: train result - loss %f, accu %f"%(epoch, total_loss, accu)
                    logging.info(mess)


                    total_loss = 0.0
                    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
                    batches = train_data_helper.batch_iter(valid_datas, FLAGS.batch_size, shuffle=False)
                    # for batch_ids_a, batch_ids_b, batch_labels in zip(batches_ids_a, batches_ids_b, batches_labels):
                    for batch in batches:
                        batch_ids_a = [train_data_helper.get_input_ids(data[0]) for data in batch]
                        batch_ids_b = [train_data_helper.get_input_ids(data[1]) for data in batch]
                        batch_labels = [data[2:] for data in batch]
                        loss_, _tp, _fp, _tn, _fn = validation_step(batch_ids_a, batch_ids_b, batch_labels)
                        total_loss += loss_
                        tp += _tp
                        tn += _tn
                        fp += _fp
                        fn += _fn
                    total_loss = total_loss / valid_batch_sum
                    accu, recall, f1, prec = Metrix.eva(tp, tn, fp, fn)
                    mess = "Evaluation: loss %f, acc %f, recall %f, precision %f, f1 %f" % \
                           (total_loss, accu, recall, prec, f1)
                    logging.info(mess)
                    logging.info("Evaluation: tp=%d, tn=%d, fp=%d, fn=%d" % (int(tp), int(tn), int(fp), int(fn)))

                    # checkpoint_prefix = "%s/model" % FLAGS.model_path
                    # path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                    # print("Saved model checkpoint to {0}".format(path))
                    if best_val_loss > total_loss:
                        best_val_loss = total_loss
                        best_val_accu = accu
                        best_val_recall = recall
                        best_val_prec = prec
                        best_val_f1 = f1
                        best_epoch = epoch
                        checkpoint_prefix = "%s/model" % FLAGS.model_path
                        path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                        print("Saved model checkpoint to {0}".format(path))
                        model_conf = {"epoch": 0,
                                      "maxSeqLength": FLAGS.max_seq_len,
                                      "hiddenSize": sum([size[1] for size in dims_hidden_conv]),
                                      "vocabDic": "vocab.txt",
                                      "encoderModelPath": "encoder",
                                      "similarityModelPath": "similairity"}
                        train_data_helper.save_vocab_file("%s/%s" % (FLAGS.model_path, model_conf["vocabDic"]))
                        model_conf_file = "%s/model.conf" % FLAGS.model_path
                        with open(model_conf_file, 'w') as wp:
                            print >> wp, json.dumps(model_conf, ensure_ascii=False)
                            wp.close()
                logging.info("Best epoch=%d, loss=%f, accu=%.4f, recall=%.4f, prec=%.4f, f1=%.4f",
                             best_epoch, best_val_loss, best_val_accu, best_val_recall, best_val_prec, best_val_f1)
        logging.info("Training done")

if __name__ == "__main__":
    train()




