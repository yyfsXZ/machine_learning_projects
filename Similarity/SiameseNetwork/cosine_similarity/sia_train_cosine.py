#!/usr/env/bin python
#coding=utf-8

import sys
import os
import numpy as np
import tensorflow as tf
import logging

import siamese_transformer_cosine as st
from hyperparams import Hyperparams as hp
import data_helper

# Data parameters
tf.flags.DEFINE_string("train_file", hp.train_file, "Data for the training data.")
tf.flags.DEFINE_string("valid_file", hp.valid_file, "Data for validation")

# Train parameters
tf.flags.DEFINE_string("model_path", hp.model_path, "Path to save model")
tf.flags.DEFINE_string("log_path", hp.log_dir, "Path to save log")
tf.flags.DEFINE_integer("epochs", hp.num_epochs, "Numbers of train epoch")
tf.flags.DEFINE_float("dropout_keep_rate", hp.dropout_keep_rate, "Dropout keep rate")
tf.flags.DEFINE_float("accu_threshold", hp.accu_threshold, "Accurate score threshold")
tf.flags.DEFINE_integer("embedding_type", hp.embedding_type, "1: trainable=true, 0: trainable=false")
tf.flags.DEFINE_string("embedding_file", hp.embedding_file, "pre-trained embedding")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "allow_soft_placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "log_device_placement")

# Model parameters
tf.flags.DEFINE_integer("batch_size", hp.batch_size, "Batch size for training")
tf.flags.DEFINE_integer("test_batch_size", hp.test_batch_size, "Batch size for validation")
tf.flags.DEFINE_float("learning_rate", hp.learning_rate, "Learning rate")
tf.flags.DEFINE_integer("max_seq_len", hp.max_len, "Max length for sequence")
tf.flags.DEFINE_integer("min_word_freq", hp.min_word_freq, "Min word freq")
tf.flags.DEFINE_integer("num_blocks", hp.num_blocks, "Number of blocks for multihead attention")
tf.flags.DEFINE_integer("num_heads", hp.num_heads, "Number of heads for multihead attention")
tf.flags.DEFINE_integer("embedding_size", hp.emb_size, "Size of embedding for token/position")
tf.flags.DEFINE_integer("hidden_size", hp.hidden_size, "Size of hidden nodes")
tf.flags.DEFINE_integer("fc_size", hp.fc_size, "Size of fc layer")
tf.flags.DEFINE_integer("ff_size", hp.ff_size, "Size of feed-forward layer")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_class", 1, "Number of classes")

FLAGS = tf.flags.FLAGS

def initLogging():
    if not os.path.exists(FLAGS.model_path):
        os.system("mkdir %s" % FLAGS.model_path)
    # 训练日志
    train_log = FLAGS.log_path + "/train.log"
    # tensorboard日志
    tensorboard_log_path = FLAGS.log_path + "/tensorboard"
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

def save_model(saver, model_path, sess):
    tf_model_path = FLAGS.model_path + "tf_model"
    if os.path.exists(tf_model_path):
        os.remove(tf_model_path)
    saver.save(sess, tf_model_path)

def examples(querys_a, ids_a, querys_b, ids_b, labels):
    logging.info("Load datas like this:\n")
    for query_a, id_a, query_b, id_b, label in zip(querys_a, ids_a, querys_b, ids_b, labels):
        logging.info("query_a:\t%s" % " ".join(query_a))
        logging.info("id_a:\t%s" % " ".join([str(id_) for id_ in id_a]))
        logging.info("query_b:\t%s" % " ".join(query_b))
        logging.info("id_b:\t%s" % " ".join([str(id_) for id_ in id_b]))
        if label[0] == 1:
            label_ = 1
        else:
            label_ = 0
        logging.info("label:\t%d" % label_)
        logging.info("*" * 100)

def train():
    # 初始化日志
    initLogging()

    # 加载vocab
    word2id, id2word = data_helper.create_vocab(FLAGS.train_file, FLAGS.model_path)
    vocab_size = len(word2id.keys())
    logging.info("Create vocab done, vocab size=%d" % vocab_size)

    # 加载预训练词向量
    if FLAGS.embedding_file != None:
        pretrained_emb = data_helper.create_vocab_emb(word2id, FLAGS.embedding_file, FLAGS.embedding_size)
    else:
        pretrained_emb = None

    # 加载训练数据
    train_ids_a, train_ids_b, train_querys_a, train_querys_b, train_labels = data_helper.create_data(FLAGS.train_file, word2id, keep_query=False)
    train_data_size = len(train_ids_a)
    logging.info("Load train file done, size=%d" % train_data_size)
    examples(train_querys_a[:3], train_ids_a[:3], train_querys_b[:3], train_ids_b[:3], train_labels[:3])
    # 加载测试数据
    test_ids_a, test_ids_b, test_querys_a, test_querys_b, test_labels = data_helper.create_data(FLAGS.valid_file, word2id, keep_query=False)
    test_data_size = len(test_ids_a)
    logging.info("Load test file done, size=%d" % test_data_size)

    logging.info("Train start")

    # Build a graph and transformer object
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            gpu_options=gpu_options,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = st.SiameseTransformer(FLAGS.num_heads,
                                          FLAGS.num_blocks,
                                          FLAGS.max_seq_len,
                                          FLAGS.ff_size,
                                          FLAGS.fc_size,
                                          FLAGS.hidden_size,
                                          FLAGS.embedding_size,
                                          vocab_size,
                                          FLAGS.num_class,
                                          FLAGS.embedding_type,
                                          pretrained_emb
                                          )
            model.build()
            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(model.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=model.global_step, name="train_op")

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            def metrix(tp, tn, fp, fn):
                accu = (tp + tn) / (tp + tn + fp + fn)
                recall = tp / (tp + fn)
                f1 = 2*tp / (2*tp + fp + fn)
                if (tp+fp) != 0:
                    prec = tp / (tp + fp)
                else:
                    prec = 0.0
                return accu, recall, f1, prec

            def get_accu(scores, labels):
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                for score, label in zip(scores, labels):
                    if label == 1 and score >= FLAGS.accu_threshold:
                        tp += 1
                    elif label == 1 and score < FLAGS.accu_threshold:
                        fn += 1
                    elif label == 0 and score >= FLAGS.accu_threshold:
                        fp += 1
                    elif label == 0 and score < FLAGS.accu_threshold:
                        tn += 1
                return tp, fp, tn, fn

            def train_step(batch_ids_a, batch_ids_b, batch_labels):
                feed_dict = {
                    model.input_query_a: batch_ids_a,
                    model.input_query_b: batch_ids_b,
                    model.Y: batch_labels,
                    model.dropout_keep_rate: FLAGS.dropout_keep_rate,
                }
                # _, loss, accuracy, logits, softmax_scores, predictions  = sess.run(
                #     [train_op, model.loss, model.accuracy, model.logits, model.softmax_scores, model.predictions], feed_dict=feed_dict
                # )
                # print logits
                # print softmax_scores
                # print predictions
                _, loss, scores = sess.run(
                    [train_op, model.loss, model.scores], feed_dict=feed_dict
                )
                tp, fp, tn, fn = get_accu(scores[:], batch_labels[:])
                return loss, tp, fp, tn, fn

            def validation_step(batch_ids_a, batch_ids_b, batch_labels):
                feed_dict = {
                    model.input_query_a: batch_ids_a,
                    model.input_query_b: batch_ids_b,
                    model.Y: batch_labels,
                    model.dropout_keep_rate: 1.0,
                }
                # loss, accuracy, recall, precision, f1, auc, logits = sess.run(
                #     [model.loss, model.accuracy, model.recall, model.precision, model.F1, model.AUC, model.logits],
                #     feed_dict=feed_dict
                # )
                # print logits
                loss, scores = sess.run(
                    [model.score_losses, model.scores],
                    feed_dict=feed_dict
                )
                tp, fp, tn, fn = get_accu(scores[:], batch_labels[:])
                return loss, tp, fp, tn, fn

            with tf.device("/gpu:0"):
                batch_per_epoch = train_data_size / FLAGS.batch_size
                if train_data_size % FLAGS.batch_size != 0:
                    batch_per_epoch += 1

                test_batch_sum = test_data_size / FLAGS.test_batch_size
                if test_data_size % FLAGS.test_batch_size != 0:
                    test_batch_sum += 1

                best_val_loss = 10000000.0
                best_val_accu = 0.0
                best_val_recall = 0.0
                best_val_prec = 0.0
                best_val_f1 = 0.0
                best_epoch = -1
                for epoch in range(FLAGS.epochs):
                    # Generate train batches
                    # batches_ids_a, batches_ids_b, batches_labels = data_helper.batch_iter(train_ids_a,
                    #                                                                       train_ids_b,
                    #                                                                       train_labels,
                    #                                                                       FLAGS.batch_size)
                    batches = data_helper.batch_iter(list(zip(train_ids_a, train_ids_b, train_labels)), FLAGS.batch_size)
                    total_loss = 0.0
                    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
                    # for batch_ids_a, batch_ids_b, batch_labels in zip(batches_ids_a, batches_ids_b, batches_labels):
                    for idx, batch in enumerate(batches):
                        batch_ids_a, batch_ids_b, batch_labels = zip(*batch)
                        batch_labels = np.array(batch_labels)[:, 0]
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
                    # Generate validation batches
                    # batches_ids_a, batches_ids_b, batches_labels = data_helper.batch_iter(test_ids_a,
                    #                                                                       test_ids_b,
                    #                                                                       test_labels,
                    #                                                                       FLAGS.batch_size)
                    batches = data_helper.batch_iter(list(zip(test_ids_a, test_ids_b, test_labels)), FLAGS.batch_size)
                    # for batch_ids_a, batch_ids_b, batch_labels in zip(batches_ids_a, batches_ids_b, batches_labels):
                    for batch in batches:
                        batch_ids_a, batch_ids_b, batch_labels = zip(*batch)
                        batch_labels = np.array(batch_labels)[:, 0]
                        loss_, _tp, _fp, _tn, _fn = validation_step(batch_ids_a, batch_ids_b, batch_labels)
                        total_loss += loss_
                        tp += _tp
                        tn += _tn
                        fp += _fp
                        fn += _fn
                    total_loss = total_loss / test_batch_sum
                    accu, recall, f1, prec = metrix(tp, tn, fp, fn)
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
                logging.info("Best epoch=%d, loss=%f, accu=%.4f, recall=%.4f, prec=%.4f, f1=%.4f",
                             best_epoch, best_val_loss, best_val_accu, best_val_recall, best_val_prec, best_val_f1)
        logging.info("Training done")

if __name__ == "__main__":
    train()
