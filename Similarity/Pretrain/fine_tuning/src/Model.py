#!/usr/env/bin python
#coding=utf-8

import tensorflow as tf
import numpy as np

class Model:
    def __init__(self,
                 d_emb,
                 d_hiddens,
                 d_fc):
        self.d_emb = d_emb
        self.d_hiddens = d_hiddens
        self.d_fc = d_fc

        self.inputs_a = tf.placeholder(tf.float32, [None, self.d_emb], name="inputs_a")
        self.inputs_b = tf.placeholder(tf.float32, [None, self.d_emb], name="inputs_b")
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

    def build_hidden_layer(self):
        dims = [self.d_emb] + self.d_hiddens
        self.outputs_a = tf.nn.dropout(self.inputs_a, keep_prob=self.dropout_keep_rate)
        self.outputs_b = tf.nn.dropout(self.inputs_b, keep_prob=self.dropout_keep_rate)
        with tf.variable_scope("hidden_layer", reuse=tf.AUTO_REUSE):
            for layer_id, d_out in enumerate(self.d_hiddens):
                d_in = dims[layer_id]
                with tf.variable_scope("layer_%d" % layer_id, reuse=tf.AUTO_REUSE):
                    W = tf.Variable(tf.truncated_normal(shape=[d_in, d_out],
                                                        stddev=0.1, dtype=tf.float32), name="w")
                    B = tf.Variable(tf.constant(value=0.1, shape=[d_out], dtype=tf.float32), name="b")
                    self.outputs_a = tf.nn.relu(tf.nn.xw_plus_b(self.outputs_a, W, B))
                    self.outputs_b = tf.nn.relu(tf.nn.xw_plus_b(self.outputs_b, W, B))

    def build_fc_layer(self):
        self.outputs_a = tf.nn.dropout(self.outputs_a, keep_prob=self.dropout_keep_rate)
        self.outputs_b = tf.nn.dropout(self.outputs_b, keep_prob=self.dropout_keep_rate)
        with tf.variable_scope("fc_layer", reuse=tf.AUTO_REUSE):
            d_in = self.d_hiddens[-1]
            d_out = self.d_fc
            W = tf.Variable(tf.truncated_normal(shape=[d_in, d_out],
                                                stddev=0.1, dtype=tf.float32), name="w")
            B = tf.Variable(tf.constant(value=0.1, shape=[d_out], dtype=tf.float32), name="b")
            self.fc_outputs_a = tf.nn.xw_plus_b(self.outputs_a, W, B, name="outputs_a")
            self.fc_outputs_b = tf.nn.xw_plus_b(self.outputs_b, W, B, name="outputs_b")

    def build_output(self):
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            pooled_a = tf.sqrt(tf.reduce_sum(tf.square(self.fc_outputs_a), 1))
            pooled_b = tf.sqrt(tf.reduce_sum(tf.square(self.fc_outputs_b), 1))
            pooled_ab = tf.reduce_sum(self.fc_outputs_a*self.fc_outputs_b, 1)
            self.scores = tf.div(pooled_ab, pooled_a*pooled_b + 1e-8, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.square(self.scores - self.input_y)
            self.score_losses = tf.reduce_mean(losses, name="score_losses")
            self.l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * 0.0000001
            self.loss = tf.add(self.score_losses, self.l2_losses, name="loss")

    def build(self):
        # 构建隐层
        self.build_hidden_layer()
        # 构建全连接层
        self.build_fc_layer()
        # 计算输出
        self.build_output()

if __name__ == "__main__":
    emb_size = 16
    hidden_sizes = [64, 128]
    fc_size = 256

    sess = tf.Session()
    with sess.as_default():
        model = Model(d_emb=emb_size,
                      d_hiddens=hidden_sizes,
                      d_fc=256)
        model.build()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        inputs_a = np.random.rand(1, emb_size)*2 - 1
        inputs_b = np.random.rand(1, emb_size)*2 - 1
        print inputs_a[0]
        print inputs_b[0]

        # inputs_a = np.ones([1, emb_size])
        # inputs_b = np.ones([1, emb_size])
        scores, fc_outputs_a, fc_outputs_b = sess.run([model.scores, model.fc_outputs_a, model.fc_outputs_b],
                                                      feed_dict={model.inputs_a: inputs_a,
                                                                 model.inputs_b: inputs_b,
                                                                 model.dropout_keep_rate: 1.0})
        print scores

