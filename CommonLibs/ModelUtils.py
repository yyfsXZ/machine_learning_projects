#!/usr/env/bin python
#coding=utf-8

import os
import tensorflow as tf

def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
    return outputs

def spe_xw_plus_b(inputs, w, b):
    """
        @Author:            Xiang Zhang
        @Date:              10:49 AM 2019/5/10
        @Description:       Auto ajust inputs.shape[0] to w

    """
    w_ = tf.tile(w, [tf.shape(inputs)[0], 1])
    w = tf.reshape(w_, [tf.shape(inputs)[0], tf.shape(w)[0], tf.shape(w)[1]])
    return tf.nn.xw_plus_b(inputs, w, b)

def train_step(loss, learning_rate, global_step, decay_step=-1, decay_ratio=-1, norm_ratio=1.25, decay=True):
    """
        @Author:            Xiang Zhang
        @Date:              10:21 AM 2019/5/13
        @Description:       train step for tensorflow
        @params:
            loss: loss for model
            learning_rate:

    """
    if decay:
        if decay_step > 0:
            # 梯度衰减，每轮后衰减为原始值*decay_step
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_ratio, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads, vars = zip(*optimizer.compute_gradients(loss))
    if norm_ratio > 1:
        # 梯度归一，防止梯度爆炸或梯度消失
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=norm_ratio)
    return optimizer.apply_gradients(zip(grads, vars), global_step=global_step, name="train_op")

def get_token_embedding(vocab_size,
                        embedding_size,
                        pretrained_embedding=None,
                        trainable=True,
                        zero_pad=True,
                        scope="token_embedding"):
    """
        @Author:            Xiang Zhang
        @Date:              11:53 AM 2019/5/16
        @Description:       generate token embeddings for embedding layer
            vocab_size: vocal size
            embedding_size: input embedding size
            pretrained_embedding: input pretrained embedding; if None, will init embedding by random methods
            trainable: if pretrained_embedding != None and trainable=True, generate pretrained embedding for trainable
            zero_pad: whether need to paddle on for token_index=0
            scope: scope name
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if pretrained_embedding is None:
            embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0,
                                                       dtype=tf.float32), trainable=True, name="embedding")
        else:
            if not trainable:
                embeddings = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
            else:
                embeddings = tf.Variable(pretrained_embedding, trainable=True, dtype=tf.float32, name="embedding")
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, embedding_size]),
                                    embeddings[1:, :]), 0)
        return embeddings

def save_model(saver, model_path, sess):
    tf_model_path = model_path + "tf_model"
    if os.path.exists(tf_model_path):
        os.remove(tf_model_path)
    saver.save(sess, tf_model_path)
