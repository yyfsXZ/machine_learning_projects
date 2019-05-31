#!/usr/env/bin python
#coding=utf-8
import sys
import tensorflow as tf
import os
import json

from hyperparams import Hyperparams

prj_name = sys.argv[1]
hp = Hyperparams(prj_name)
model_id = sys.argv[2]
RAW_MODEL_DIR = hp.train_params["model_path"]
MODEL_DIR = "%s/model-%s" % (RAW_MODEL_DIR, model_id)
PB_MODEL_DIR = "%s_pb" % RAW_MODEL_DIR
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

        model_inputs = {
            "inputs_a": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("inputs_a").outputs[0]),
            "dropout_keep_rate": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("dropout_keep_rate").outputs[0]),
        }
        model_outputs = {
            "fc_layer/outputs_a": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("fc_layer/outputs_a").outputs[0]),
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(model_inputs, model_outputs, 'inference_sig_name')
        builder = tf.saved_model.builder.SavedModelBuilder(PB_MODEL_DIR)
        builder.add_meta_graph_and_variables(
            sess,
            ["serve"],
            {'inference_signature': signature}
        )
        builder.save()

