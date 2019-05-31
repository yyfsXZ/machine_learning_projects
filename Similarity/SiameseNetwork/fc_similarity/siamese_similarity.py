#!/usr/env/bin python
#coding=utf-8

"""
    @file   :           siamese_transformer.py
    @Author :           Xiang Zhang
    @Date   :           6:22 PM 2019/2/27
    @Description:
"""

import tensorflow as tf
import numpy as np

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
    w_ = tf.tile(w, [tf.shape(inputs)[0], 1])
    w = tf.reshape(w_, [tf.shape(inputs)[0], tf.shape(w)[0], tf.shape(w)[1]])
    return tf.nn.xw_plus_b(inputs, w, b)

def residual(querys, outputs, W, B):
    residual_result = tf.matmul(querys, tf.transpose(outputs, [0, 2, 1]))
    residual_result = spe_xw_plus_b(residual_result, W, B) + outputs
    return residual_result

class SiameseTransformer:
    def __init__(self,
                 num_heads,
                 num_blocks,
                 sequence_len,
                 ff_size,
                 fc_size,
                 hidden_size,
                 embedding_size,
                 vocab_size,
                 num_class=2,
                 embedding_type=1,
                 pretrained_embedding=None,
                 use_position=True,
                 ):
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.sequence_len = sequence_len
        self.d_ff = ff_size
        self.d_fc = fc_size
        self.d_hidden = hidden_size
        self.d_emb = embedding_size
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.use_position = use_position

        self.input_query_a = tf.placeholder(tf.int32, [None, self.sequence_len], name="input_query_a")
        self.input_query_b = tf.placeholder(tf.int32, [None, self.sequence_len], name="input_query_b")
        self.Y = tf.placeholder(tf.int32, [None, self.num_class], name="input_Y")
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        self.token_embedding = self.get_token_embedding(pretrained_embedding, embedding_type)
        self.init_variables()

    def init_variables(self):
        self.W_Q = []
        self.W_K = []
        self.W_V = []
        self.W_ff_relu = []
        self.W_ff = []
        self.W_r = []   # residual layer W
        self.B_Q = []
        self.B_K = []
        self.B_V = []
        self.B_ff_relu = []
        self.B_ff = []
        self.B_r = []   # residual layer B

        for i in range(self.num_blocks):
            with tf.variable_scope("dense_variable_Q_%d" % i, reuse=tf.AUTO_REUSE):
                W_Q = tf.Variable(tf.truncated_normal(shape=[self.d_emb, self.d_hidden],
                                                           stddev=0.1, dtype=tf.float32), name="W_Q_%d" % i)
                B_Q = tf.Variable(tf.constant(value=0.1, shape=[self.d_hidden], dtype=tf.float32), name="B_Q_%d" % i)
                self.W_Q.append(W_Q)
                self.B_Q.append(B_Q)
            with tf.variable_scope("dense_variable_K_%d" % i, reuse=tf.AUTO_REUSE):
                W_K = tf.Variable(tf.truncated_normal(shape=[self.d_emb, self.d_hidden],
                                                           stddev=0.1, dtype=tf.float32), name="W_K_%d" % i)
                B_K = tf.Variable(tf.constant(value=0.1, shape=[self.d_hidden], dtype=tf.float32), name="B_K_%d" % i)
                self.W_K.append(W_K)
                self.B_K.append(B_K)
            with tf.variable_scope("dense_variable_V_%d" % i, reuse=tf.AUTO_REUSE):
                W_V = tf.Variable(tf.truncated_normal(shape=[self.d_emb, self.d_hidden],
                                                           stddev=0.1, dtype=tf.float32), name="W_V_%d" % i)
                B_V = tf.Variable(tf.constant(value=0.1, shape=[self.d_hidden], dtype=tf.float32), name="B_V_%d" % i)
                self.W_V.append(W_V)
                self.B_V.append(B_V)
            with tf.variable_scope("dense_variable_ff_relu_%d" % i, reuse=tf.AUTO_REUSE):
                W_ff_relu = tf.Variable(tf.truncated_normal(shape=[self.d_hidden, self.d_ff],
                                                                 stddev=0.1, dtype=tf.float32), name="W_ff_relu_%d" % i)
                B_ff_relu = tf.Variable(tf.constant(value=0.1, shape=[self.d_ff], dtype=tf.float32), name="B_ff_relu_%d" % i)
                self.W_ff_relu.append(W_ff_relu)
                self.B_ff_relu.append(B_ff_relu)
            with tf.variable_scope("dense_variable_ff_out_%d" % i, reuse=tf.AUTO_REUSE):
                W_ff = tf.Variable(tf.truncated_normal(shape=[self.d_ff, self.d_hidden],
                                                            stddev=0.1, dtype=tf.float32), name="W_ff_out_%d" % i)
                B_ff = tf.Variable(tf.constant(value=0.1, shape=[self.d_hidden], dtype=tf.float32), name="B_ff_out_%d" % i)
                self.W_ff.append(W_ff)
                self.B_ff.append(B_ff)
            with tf.variable_scope("residual_%d" % i, reuse=tf.AUTO_REUSE):
                W_r_ = tf.Variable(tf.truncated_normal(shape=[self.sequence_len, self.d_hidden],
                                                       stddev=0.1, dtype=tf.float32), name="W_residual_%d" % i)
                B_r_ = tf.Variable(tf.constant(value=0.1, shape=[self.d_hidden], dtype=tf.float32), name="B_residual_%d" % i)
                self.W_r.append(W_r_)
                self.B_r.append(B_r_)

    def ff(self, inputs, block_id):
        """
            @Author:            Xiang Zhang(zhangxiang8@xiaomi.com)
            @Date:              11:25 AM 2019/2/28
            @Description:       feed_forward method

        """
        with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.nn.relu(spe_xw_plus_b(inputs, self.W_ff_relu[block_id], self.B_ff_relu[block_id]))

            # Outer layer
            outputs = spe_xw_plus_b(outputs, self.W_ff[block_id], self.B_ff[block_id])
            # Residual connection
            outputs += inputs

            # Normalize
            outputs = ln(outputs)

        return outputs

    def get_positional_embedding(self,
                                 inputs,
                                 mask=True):
        """
            @Author:            Xiang Zhang
            @Date:              11:25 AM 2019/2/28
            @Description:       position embedding

        """
        E = inputs.get_shape().as_list()[-1] # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
        with tf.variable_scope("positional_embedding", reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
                for pos in range(self.sequence_len)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if mask:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.to_float(outputs)

    def get_token_embedding(self,
                            pretrained_embedding=None,
                            embedding_type=1,
                            zero_pad=True):
        with tf.variable_scope("token_embedding", reuse=tf.AUTO_REUSE):
            if pretrained_embedding is None:
                # embeddings = tf.get_variable('weight_mat',
                #                              dtype=tf.float32,
                #                              shape=(self.vocab_size, self.d_hidden),
                #                              initializer=tf.contrib.layers.xavier_initializer())

                embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.d_emb], minval=-1.0, maxval=1.0,
                                                           dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    embeddings = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                else:
                    embeddings = tf.Variable(pretrained_embedding, trainable=True, dtype=tf.float32, name="embedding")
            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, self.d_emb]),
                                        embeddings[1:, :]), 0)
        return embeddings

    # def spe_xw_plus_b(self,
    #                   inputs,
    #                   w,
    #                   b):
    #     w_ = tf.tile(w, [tf.shape(inputs)[0], 1])
    #     w = tf.reshape(w_, [tf.shape(inputs)[0], tf.shape(w)[0], tf.shape(w)[1]])
    #     return tf.nn.xw_plus_b(inputs, w, b)

    def multihead_attention(self,
                            querys,
                            keys,
                            values,
                            block_id):
        """
            @Author:            Xiang Zhang(zhangxiang8@xiaomi.com)
            @Date:              11:31 AM 2019/2/28
            @Description:       multi-head attention

        """
        with tf.variable_scope("multihead_attention", reuse=tf.AUTO_REUSE):
            # Linear projections
            # Q = tf.nn.xw_plus_b(querys, self.W_Q[block_id], self.B_Q[block_id]) # (N, T_q, d_model)
            # K = tf.nn.xw_plus_b(keys, self.W_K[block_id], self.B_K[block_id]) # (N, T_k, d_model)
            # V = tf.nn.xw_plus_b(values, self.W_V[block_id], self.B_V[block_id]) # (N, T_v, d_model)
            Q = spe_xw_plus_b(querys, self.W_Q[block_id], self.B_Q[block_id]) # (N, T_q, d_model)
            K = spe_xw_plus_b(keys, self.W_K[block_id], self.B_K[block_id]) # (N, T_k, d_model)
            V = spe_xw_plus_b(values, self.W_V[block_id], self.B_V[block_id]) # (N, T_v, d_model)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2 ) # (N, T_q, d_model)

            # Residual connection
            # outputs += querys
            outputs = residual(querys, outputs, self.W_r[block_id], self.B_r[block_id])

            # Normalize
            outputs = ln(outputs)
        return outputs

    def scaled_dot_product_attention(self, Q, K, V):
        """
            @Author:            Xiang Zhang
            @Date:              11:46 AM 2019/2/28
            @Description:       self-attention

        """
        with tf.variable_scope("scaled_dot_product_attention", reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # key masking
            outputs = self.mask(outputs, Q, K, type="key")

            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # query masking
            outputs = self.mask(outputs, Q, K, "query")

            # dropout
            outputs = tf.layers.dropout(outputs, rate=self.dropout_keep_rate)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
        return outputs

    def mask(self,
             inputs,
             Q=None,
             K=None,
             type=None):
        padding_num = -2 ** 32 + 1
        if type == "key":
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(K), axis=-1))  # (N, T_k)
            masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
            masks = tf.tile(masks, [1, tf.shape(Q)[1], 1])  # (N, T_q, T_k)

            # Apply masks to inputs
            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
        elif type == "query":
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))  # (N, T_q)
            masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(K)[1]])  # (N, T_q, T_k)

            # Apply masks to inputs
            outputs = inputs*masks
        return outputs

    def build_output_layer(self,
                           inputs,
                           inputs_st,
                           ):
        # with tf.variable_scope("reduce", reuse=tf.AUTO_REUSE):
        #     W = tf.Variable(tf.truncated_normal(shape=[self.sequence_len, 1],
        #                                         stddev=0.1, dtype=tf.float32), name="W_reduce")
        #     B = tf.Variable(tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="B_reduce")
        #     outputs = self.spe_xw_plus_b(tf.transpose(inputs, [0, 2, 1]), W, B)    # (N, 2*d_hidden, 1)
        #     outputs = tf.squeeze(outputs, axis=2)   # (N, 2*d_hidden)
        #
        #     outputs_st = self.spe_xw_plus_b(tf.transpose(inputs_st, [0, 2, 1]), W, B)    # (N, 2*d_hidden, 1)
        #     outputs_st = tf.squeeze(outputs_st, axis=2)   # (N, 2*d_hidden)
        #     # print outputs.get_shape()

        outputs = inputs
        outputs_st = inputs_st
        for i in range(len(self.d_fc)):
            if i == 0:
                d_input = self.d_hidden*2
                d_output = self.d_fc[i]
            else:
                d_input = self.d_fc[i-1]
                d_output = self.d_fc[i]
            print "fc_%d, d_input=%d, d_output=%d" % (i, d_input, d_output)
            with tf.variable_scope("fc_%d" % i, reuse=tf.AUTO_REUSE):
                W = tf.Variable(tf.truncated_normal(shape=[d_input, d_output],
                                                    stddev=0.1, dtype=tf.float32), name="W_fc")
                B = tf.Variable(tf.constant(value=0.1, shape=[d_output], dtype=tf.float32), name="B_fc")
                outputs = tf.nn.relu(tf.nn.xw_plus_b(outputs, W, B))
                outputs_st = tf.nn.relu(tf.nn.xw_plus_b(outputs_st, W, B))

        outputs = tf.layers.dropout(outputs, rate=self.dropout_keep_rate)
        outputs_st = tf.layers.dropout(outputs_st, rate=self.dropout_keep_rate)

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W = tf.Variable(tf.truncated_normal(shape=[self.d_fc[-1], self.num_class],
                                                stddev=0.1, dtype=tf.float32), name="W_output")
            B = tf.Variable(tf.constant(value=0.1, shape=[self.num_class], dtype=tf.float32), name="B_output")
            self.logits = tf.nn.xw_plus_b(outputs, W, B, name="logits")
            self.softmax_scores = tf.nn.softmax(self.logits, name="softmax_scores")

            self.logits_st = tf.nn.xw_plus_b(outputs_st, W, B, name="logits_st")
            self.softmax_scores_st = tf.nn.softmax(self.logits_st, name="softmax_scores_st")

            self.predictions = tf.argmax(self.softmax_scores, 1, name="predictions")
            self.labels = tf.argmax(self.Y, 1, name="labels")
            self.topKPreds = tf.nn.top_k(self.softmax_scores, k=1, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits)
            self.softmax_score_losses = tf.reduce_mean(losses, name="logits_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * 0.0000001
            self.loss = tf.add(self.softmax_score_losses, l2_losses, name="loss")

    def build_output_layer_bi(self,
                           inputs_r,
                           inputs_l
                           ):
        with tf.variable_scope("reduce", reuse=tf.AUTO_REUSE):
            W = tf.Variable(tf.truncated_normal(shape=[self.sequence_len, 1],
                                                stddev=0.1, dtype=tf.float32), name="W_reduce")
            B = tf.Variable(tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="B_reduce")
            outputs_r = spe_xw_plus_b(tf.transpose(inputs_r, [0, 2, 1]), W, B)    # (N, 2*d_hidden, 1)
            outputs_r = tf.squeeze(outputs_r, axis=2)   # (N, 2*d_hidden)
            outputs_l = spe_xw_plus_b(tf.transpose(inputs_l, [0, 2, 1]), W, B)    # (N, 2*d_hidden, 1)
            outputs_l = tf.squeeze(outputs_l, axis=2)   # (N, 2*d_hidden)
            # print outputs.get_shape()

        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            W = tf.Variable(tf.truncated_normal(shape=[self.d_hidden*2, self.d_fc],
                                                stddev=0.1, dtype=tf.float32), name="W_fc")
            B = tf.Variable(tf.constant(value=0.1, shape=[self.d_fc], dtype=tf.float32), name="B_fc")
            outputs_r = tf.nn.relu(tf.nn.xw_plus_b(outputs_r, W, B))
            outputs_r = tf.layers.dropout(outputs_r, rate=self.dropout_keep_rate)
            outputs_l = tf.nn.relu(tf.nn.xw_plus_b(outputs_l, W, B))
            outputs_l = tf.layers.dropout(outputs_l, rate=self.dropout_keep_rate)
            # print outputs.get_shape()

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W = tf.Variable(tf.truncated_normal(shape=[self.d_fc, self.num_class],
                                                stddev=0.1, dtype=tf.float32), name="W_output")
            B = tf.Variable(tf.constant(value=0.1, shape=[self.num_class], dtype=tf.float32), name="B_output")
            self.logits_r = tf.nn.xw_plus_b(outputs_r, W, B, name="logits_r")
            self.logits_l = tf.nn.xw_plus_b(outputs_l, W, B, name="logits_l")
            self.softmax_scores_r = tf.nn.softmax(self.logits_r, name="softmax_scores_r")
            self.softmax_scores_l = tf.nn.softmax(self.logits_l, name="softmax_scores_l")

            div = tf.constant(2, dtype=tf.float32)
            self.softmax_scores = tf.divide(tf.add(self.softmax_scores_r, self.softmax_scores_l, name=None), div, name="softmax_scores")

            self.predictions = tf.argmax(self.softmax_scores, 1, name="predictions")
            self.labels = tf.argmax(self.Y, 1, name="labels")
            self.topKPreds = tf.nn.top_k(self.softmax_scores, k=1, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits_r) + tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits_l)
            self.softmax_score_losses = tf.reduce_mean(losses, name="logits_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * 0.0000001
            self.loss = tf.add(self.softmax_score_losses, l2_losses, name="loss")

    def metrix(self):
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Number of correct predictions
        with tf.name_scope("num_correct"):
            correct = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"), name="num_correct")

        # Calculate Fp
        with tf.name_scope("fp"):
            fp = tf.metrics.false_positives(labels=tf.argmax(self.Y, 1), predictions=self.predictions)
            self.fp = tf.reduce_sum(tf.cast(fp, "float"), name="fp")

        # Calculate Fn
        with tf.name_scope("fn"):
            fn = tf.metrics.false_negatives(labels=tf.argmax(self.Y, 1), predictions=self.predictions)
            self.fn = tf.reduce_sum(tf.cast(fn, "float"), name="fn")

        # Calculate Recall
        with tf.name_scope("recall"):
            self.recall = self.num_correct / (self.num_correct + self.fn)

        # Calculate Precision
        with tf.name_scope("precision"):
            self.precision = self.num_correct / (self.num_correct + self.fp)

        # Calculate F1
        with tf.name_scope("F1"):
            self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

        # Calculate AUC
        with tf.name_scope("AUC"):
            self.AUC = tf.metrics.auc(self.softmax_scores, self.Y, name="AUC")

    def build_encoder(self, inputs, use_position=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # embedding
            enc = tf.nn.embedding_lookup(self.token_embedding, inputs)
            enc += self.d_hidden**0.5 # scale

            # position emb
            if use_position:
                enc += self.get_positional_embedding(enc, self.sequence_len)

            enc = tf.layers.dropout(enc, self.dropout_keep_rate)

            # blocks
            for block_id in range(self.num_blocks):
                with tf.variable_scope("blocks_%d" % block_id, reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = self.multihead_attention(querys=enc,
                                                   keys=enc,
                                                   values=enc,
                                                   block_id=block_id)
                    # feed forward
                    enc = self.ff(enc, block_id)
            return enc

    def build(self):
        # encoder
        self.enc_output_a = self.build_encoder(self.input_query_a, use_position=self.use_position)
        self.enc_output_b = self.build_encoder(self.input_query_b, use_position=self.use_position)

        with tf.variable_scope("reduce", reuse=tf.AUTO_REUSE):
            W = tf.Variable(tf.truncated_normal(shape=[self.sequence_len, 1],
                                                stddev=0.1, dtype=tf.float32), name="W_reduce")
            B = tf.Variable(tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="B_reduce")
            self.enc_output_a = spe_xw_plus_b(tf.transpose(self.enc_output_a, [0, 2, 1]), W, B)    # (N, d_hidden, 1)
            self.enc_output_b = spe_xw_plus_b(tf.transpose(self.enc_output_b, [0, 2, 1]), W, B)    # (N, d_hidden, 1)
            self.enc_output_a = tf.squeeze(self.enc_output_a, axis=2, name="encoder_output_a")   # (N, d_hidden)
            self.enc_output_b = tf.squeeze(self.enc_output_b, axis=2, name="encoder_output_b")   # (N, d_hidden)

        self.enc_output_st_a = tf.placeholder(tf.float32, [None, self.d_hidden], name="enc_output_st_a")
        self.enc_output_st_b = tf.placeholder(tf.float32, [None, self.d_hidden], name="enc_output_st_b")

        self.enc = tf.concat([self.enc_output_a, self.enc_output_b], axis=1, name="encoder_output")  # (N,  2*d_hidden)
        self.enc_st = tf.concat([self.enc_output_st_a, self.enc_output_st_b], axis=1, name="encoder_output_fc")     # (N, 2*d_hidden)

        # output layer
        self.build_output_layer(self.enc, self.enc_st)
        # self.build_output_layer_bi(self.enc_r, self.enc_l)

        # metric
        self.metrix()

