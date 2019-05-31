#!/usr/env/bin python
#coding=utf-8
import tensorflow as tf
import os

from hyperparams import Hyperparams as hp

MODEL_ROOT = hp.model_path
MODEL_DIR = "%s/model-57"
PB_MODEL_DIR = "%s_pb" % MODEL_ROOT
if os.path.exists(PB_MODEL_DIR):
    os.system("rm -rf %s" % PB_MODEL_DIR)
os.system("mkdir %s" % PB_MODEL_DIR)
os.system("cp %s/word2id.txt %s" % (MODEL_ROOT, PB_MODEL_DIR))

ENCODER_MODEL_DIR = "%s/encoder" % PB_MODEL_DIR
ST_MODEL_DIR = "%s/siamese_transformer" % PB_MODEL_DIR
if os.path.exists(ENCODER_MODEL_DIR):
    os.system("rm -rf %s" % ENCODER_MODEL_DIR)
if os.path.exists(ST_MODEL_DIR):
    os.system("rm -rf %s" % ST_MODEL_DIR)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("%s.meta" % MODEL_DIR)
        # saver = tf.train.import_meta_graph("{0}.meta".format(MODEL_DIR))
        saver.restore(sess, MODEL_DIR)

        encoder_model_inputs = {
            "input_query_a" : tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("input_query_a").outputs[0]),
            "dropout_keep_rate" : tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("dropout_keep_rate").outputs[0]),
        }
        encoder_model_outputs = {
            "reduce/encoder_output_a": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("reduce/encoder_output_a").outputs[0]),
        }

        encdoer_signature = tf.saved_model.signature_def_utils.build_signature_def(encoder_model_inputs, encoder_model_outputs, 'inference_sig_name')
        encoder_builder = tf.saved_model.builder.SavedModelBuilder(ENCODER_MODEL_DIR)
        encoder_builder.add_meta_graph_and_variables(
            sess,
            ["serve"],
            {'inference_signature': encdoer_signature}
        )
        encoder_builder.save()

        st_inputs = {
            "enc_output_st_a" : tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("enc_output_st_a").outputs[0]),
            "encoder_output_st_b" : tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("enc_output_st_b").outputs[0]),
            "dropout_keep_rate" : tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("dropout_keep_rate").outputs[0]),
        }
        st_outputs = {
            "output/softmax_scores_st": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("output/scores_st").outputs[0])
        }

        st_signature = tf.saved_model.signature_def_utils.build_signature_def(st_inputs, st_outputs, 'inference_sig_name')

        st_builder = tf.saved_model.builder.SavedModelBuilder(ST_MODEL_DIR)
        st_builder.add_meta_graph_and_variables(
            sess,
            ["serve"],
            {'inference_signature': st_signature}
        )
        st_builder.save()
