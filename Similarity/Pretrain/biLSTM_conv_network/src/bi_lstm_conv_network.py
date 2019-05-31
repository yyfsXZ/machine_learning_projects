#!/usr/env/bin python
#coding=utf-8
"""
    @file   :           bi_lstm_conv_network.py
    @Author :           Xiang Zhang
    @Date   :           10:45 AM 2019/5/31
    @Description:       模型使用char粒度的embedding输入层；隐藏层使用一层Bi-LSTM + CNN的网络结构，然后经过全连接层到输出层
"""

import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from CommonLibs import ModelUtils

class BILSTM_CONV_NETWORK:
    def __init__(self,
                 vocab_size,
                 seq_length,
                 d_emb,
                 d_hidden_lstm,
                 d_hidden_conv,
                 d_fc,
                 pretrained_embedding=None,
                 pretrain_emb_trainable=True,
                 d_class=2):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.d_embedding = d_emb
        self.d_hidden_lstm = d_hidden_lstm
        self.d_hidden_conv = d_hidden_conv  # [[kernel_height, channel_size, stride], ...]
        self.d_fc = d_fc
        self.pretrained_embedding = pretrained_embedding
        self.pretrain_emb_trainable = pretrain_emb_trainable
        self.d_class = d_class

        if pretrained_embedding != None:
            self.d_embedding = pretrained_embedding.shape[1]
            self.vocab_size = pretrained_embedding.shape[0]

        self.input_a = tf.placeholder(tf.int32, [None, self.seq_length], name="input_ids_a")
        self.input_b = tf.placeholder(tf.int32, [None, self.seq_length], name="input_ids_b")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        self.input_y = tf.placeholder(tf.int32, [None, self.d_class], name="input_y")

        self.embeddings = ModelUtils.get_token_embedding(self.vocab_size,
                                                         self.d_embedding,
                                                         self.pretrained_embedding,
                                                         self.pretrain_emb_trainable,
                                                         scope="embeddings")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

    def hidden_layer(self, input_ids, output_name):
        self.input_embeddings = tf.nn.embedding_lookup(self.embeddings,input_ids)

        outputs = self.input_embeddings # [batch_size, seq_length, d_embedding]

        with tf.variable_scope("bi_lstm_layers", reuse=tf.AUTO_REUSE):
            for layer_id, d_lstm_hidden in enumerate(self.d_hidden_lstm):
                with tf.variable_scope("layer_%d" % layer_id, reuse=tf.AUTO_REUSE):
                    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(d_lstm_hidden)
                    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(d_lstm_hidden)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=(1-self.dropout_prob))
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=(1-self.dropout_prob))
                    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, outputs, dtype=tf.float32)
                    outputs = tf.concat(outputs, axis=2)    # [batch_size, seq_length, d_lstm_hidden*2]
        print "lstm outputs - {}".format(outputs.get_shape())

        outputs = tf.expand_dims(outputs, -1)
        print "lstm outputs expand dim - {}".format(outputs.get_shape())
        list_conv_outputs = []
        # input_shapes = [outputs_a.get_shape()[-1]] + [shape[1] for shape in self.d_hidden_conv]    # shape[-1] means channels for convolution layer
        with tf.variable_scope("convolution_layers", reuse=tf.AUTO_REUSE):
            for shape_id, shape in enumerate(self.d_hidden_conv):
                with tf.variable_scope("%dth_convolution" % shape_id, reuse=tf.AUTO_REUSE):
                    with tf.variable_scope("conv2d", reuse=tf.AUTO_REUSE):
                        conv_height = shape[0]
                        conv_channel_num = shape[1]
                        conv_stride = shape[2]
                        conv_width = outputs.get_shape()[2]
                        conv_outputs = tf.layers.conv2d(inputs=outputs,
                                                        filters=conv_channel_num,
                                                        kernel_size=[conv_height, conv_width],
                                                        strides=conv_stride,
                                                        activation=tf.nn.relu,
                                                        use_bias=True)
                        print "{}th convolution outputs - {}".format(shape_id, conv_outputs.get_shape())

                    conv_outputs = tf.transpose(conv_outputs, [0, 3, 1, 2])
                    with tf.variable_scope("max_pooling", reuse=tf.AUTO_REUSE):
                        pool_height = 1
                        pool_width = conv_outputs.get_shape()[2]
                        pool_stride = 1
                        conv_outputs = tf.layers.max_pooling2d(inputs=conv_outputs,
                                                               pool_size=[pool_height, pool_width],
                                                               strides=pool_stride)
                        print "{}th pooling outputs - {}".format(shape_id, conv_outputs.get_shape())
                    conv_outputs = tf.squeeze(conv_outputs, [2, 3])
                    list_conv_outputs.append(conv_outputs)
        conv_outputs = tf.concat(list_conv_outputs, axis=1, name=output_name)
        print "convolution outputs - {}".format(conv_outputs.get_shape())
        return conv_outputs

    def build_output_layer(self, fc_inputs, fc_inputs_pre, d_in):
        outputs = fc_inputs
        outputs_pre = fc_inputs_pre
        dims = [d_in] + self.d_fc
        # full connect layers
        for layer_id, d_out in enumerate(self.d_fc):
            with tf.variable_scope("fc_layer_%d" % layer_id, reuse=tf.AUTO_REUSE):
                d_in = dims[layer_id]
                W = tf.Variable(tf.truncated_normal(shape=[d_in, d_out],
                                                      stddev=0.1, dtype=tf.float32), name="fc_w_%d" % layer_id)
                B = tf.Variable(tf.constant(value=0.1, shape=[d_out], dtype=tf.float32), name="fc_b_%d" % layer_id)
                outputs = tf.nn.relu(tf.nn.xw_plus_b(outputs, W, B))
                outputs_pre = tf.nn.relu(tf.nn.xw_plus_b(outputs_pre, W, B))
                print outputs.get_shape()
        # dropout
        outputs = tf.layers.dropout(outputs, rate=self.dropout_prob)
        # output_layer
        with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
            W = tf.Variable(tf.truncated_normal(shape=[int(outputs.get_shape()[-1]), self.d_class],
                                                stddev=0.1, dtype=tf.float32), name="output_w")
            B = tf.Variable(tf.constant(value=0.1, shape=[self.d_class], dtype=tf.float32), name="output_b")
            self.logits = tf.nn.xw_plus_b(outputs, W, B)
            self.logits_pre = tf.nn.xw_plus_b(outputs_pre, W, B)
        # return logits, logits_pre

    def run_loss_func(self):
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.labels = tf.argmax(self.input_y, 1, name="labels")
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            self.softmax_score_losses = tf.reduce_mean(losses, name="pred_losses")
            self.l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                      name="l2_losses") * 0.0000001
            self.loss = tf.add(self.softmax_score_losses, self.l2_losses, name="loss")

    def build(self):
        # self.build_hidden_layer(self.input_a, self.input_b)
        self.outputs_a = self.hidden_layer(self.input_a, "outputs_a")
        self.outputs_b = self.hidden_layer(self.input_b, "outputs_b")

        self.outputs_a_pre = tf.placeholder(tf.float32, self.outputs_a.get_shape(), name="outputs_a_pre")
        self.outputs_b_pre = tf.placeholder(tf.float32, self.outputs_b.get_shape(), name="outputs_b_pre")

        self.fc_inputs = tf.concat([self.outputs_a, self.outputs_b], axis=1, name="fc_inputs")
        self.fc_inputs_pre = tf.concat([self.outputs_a_pre, self.outputs_b_pre], axis=1, name="fc_inputs_pre")

        d_fc_layer_in = sum([shape[1] for shape in self.d_hidden_conv]) * 2
        self.build_output_layer(self.fc_inputs, self.fc_inputs_pre, d_fc_layer_in)
        # self.logits = self.build_output_layer(self.fc_inputs, "logits")
        # self.logits_pre = self.build_output_layer(self.fc_inputs_pre, "logits_pre")
        self.probs = tf.nn.softmax(self.logits, name="probs")
        self.probs_pre = tf.nn.softmax(self.logits_pre, name="probs_pre")

        self.run_loss_func()


if __name__ == "__main__":
    d_hidden_lstm = [256]
    d_hidden_conv = [[2, 100, 1], [3, 200, 1]]

    sess = tf.Session()
    with sess.as_default():
        model = BILSTM_CONV_NETWORK(
            vocab_size=1000,
            seq_length=20,
            d_emb=123,
            d_hidden_lstm=d_hidden_lstm,
            d_hidden_conv=d_hidden_conv,
            d_fc=[345, 678]
        )
        model.build()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ids_a = np.ones([1, 20])
        ids_b = np.ones([1, 20])
        print ids_a
        probs, outputs_a, outputs_b = sess.run([model.probs, model.outputs_a, model.outputs_b], feed_dict={model.input_a: np.array(ids_a),
                                                                                                           model.input_b: np.array(ids_b),
                                                                                                           model.dropout_prob: 0.0,
                                                                                                           model.input_y: np.array([[0, 1], [1, 0]]),})
        print outputs_a[:, :10]
        print outputs_b[:, :10]
        print probs

        probs_pre = sess.run([model.probs_pre], feed_dict={model.input_a: np.array(ids_a),
                                                           model.input_b: np.array(ids_b),
                                                           model.outputs_a_pre: outputs_a,
                                                           model.outputs_b_pre: outputs_b,
                                                           model.dropout_prob: 0.0})
        print probs_pre




