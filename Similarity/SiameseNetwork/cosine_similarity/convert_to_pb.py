#!/usr/env/bin python
#coding=utf-8
import tensorflow as tf
import os

from hyperparams import Hyperparams as hp

MODEL_ROOT = hp.model_path
model_id = sys.argv[1]
MODEL_DIR = "%s/model-%s" % (MODEL_ROOT, model_id)
PB_MODEL_DIR = "%s_pb" % MODEL_ROOT
if os.path.exists(PB_MODEL_DIR):
    os.system("rm -rf %s" % PB_MODEL_DIR)

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
        encoder_builder = tf.saved_model.builder.SavedModelBuilder(PB_MODEL_DIR)
        encoder_builder.add_meta_graph_and_variables(
            sess,
            ["serve"],
            {'inference_signature': encdoer_signature}
        )
        encoder_builder.save()
os.system("cp ../model/word2id.txt %s" % PB_MODEL_DIR)