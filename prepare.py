import os
import sys
import argparse
import glob
import json
import shutil

import numpy as np
from keras.models import load_model

import tensorflow as tf
# needed in tensorflow 2.0
#from tensorflow.keras import backend as k
from keras import backend as k

# this prepares a trained model and means/sigmas for use with CMSSW
k.set_learning_phase(0)


parser = argparse.ArgumentParser(description='Prepare')
parser.add_argument('convertDir', type=str,
                    help='Directory of input numpy arrays')
parser.add_argument('trainDir', type=str,
                    help='Directory of trained model')

args = parser.parse_args()


inDir = args.convertDir
outDir = args.trainDir

model = load_model('{}/KERAS_check_best_model.h5'.format(outDir))
print(model.inputs)
print(model.outputs)

with k.get_session() as sess:

    outputs = ["ID_pred/Softmax"] # names of output operations you want to use later
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), outputs)
    tf.train.write_graph(constant_graph, outDir, "constantgraph.pb", as_text=False)
    
    simpleDir = "{}/simplegraph".format(outDir)
    if os.path.exists(simpleDir): shutil.rmtree(simpleDir)
    builder = tf.saved_model.builder.SavedModelBuilder(simpleDir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()

# save the means
# note: this relies on the order of the output json being the same as the input vector
# should explicitly set the order (and break apart into input blocks)
with open('{}/means.json'.format(inDir)) as f:
    result = json.load(f)
means = result['means']
stds = result['stds']
with open('{}/mean_sigmas.txt'.format(outDir),'w') as f:
    for k in means:
        f.write('{},{},{}\n'.format(k,means[k],stds[k]))
    
