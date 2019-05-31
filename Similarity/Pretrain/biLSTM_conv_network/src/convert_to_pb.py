#!/usr/env/bin python
#coding=utf-8
import sys
import tensorflow as tf
import os
import json

hp = Hyperparams()

model_id = sys.argv[1]
RAW_MODEL_DIR = hp.train_params["model_path"]
with open("%s/model.conf" % RAW_MODEL_DIR, 'r') as fp:
    model_conf = json.loads(fp.readline().strip('\r\n'))
    fp.close()
model_conf["epoch"] = int(model_id)

MODEL_DIR = "%s/model-%s" % (RAW_MODEL_DIR, model_id)
PB_MODEL_DIR = "%s_pb" % RAW_MODEL_DIR
if os.path.exists(PB_MODEL_DIR):
    os.system("rm -rf %s" % PB_MODEL_DIR)
os.system("mkdir %s" % PB_MODEL_DIR)
os.system("cp %s/%s %s" % (RAW_MODEL_DIR, model_conf["vocabDic"], PB_MODEL_DIR))

with open("%s/model.conf" % PB_MODEL_DIR, 'w') as wp:
    print >> wp, json.dumps(model_conf, ensure_ascii=False)
    wp.close()

ENCODER_MODEL_DIR = "%s/%s" % (PB_MODEL_DIR, model_conf["encoderModelPath"])
ST_MODEL_DIR = "%s/%s" % (PB_MODEL_DIR, model_conf["similarityModelPath"])
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
            "input_ids_a": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("input_ids_a").outputs[0]),
            "dropout_prob": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("dropout_prob").outputs[0]),
        }
        encoder_model_outputs = {
            "outputs_a": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("outputs_a").outputs[0]),
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
            "outputs_a_pre": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("outputs_a_pre").outputs[0]),
            "outputs_b_pre": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("outputs_b_pre").outputs[0]),
            "dropout_prob": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("dropout_prob").outputs[0]),
        }
        st_outputs = {
            "probs_pre": tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("probs_pre").outputs[0])
        }

        st_signature = tf.saved_model.signature_def_utils.build_signature_def(st_inputs, st_outputs, 'inference_sig_name')

        st_builder = tf.saved_model.builder.SavedModelBuilder(ST_MODEL_DIR)
        st_builder.add_meta_graph_and_variables(
            sess,
            ["serve"],
            {'inference_signature': st_signature}
        )
        st_builder.save()

