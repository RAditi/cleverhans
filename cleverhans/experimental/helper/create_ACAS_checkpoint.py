from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans_tutorials.tutorial_models import ModelBasicMLP
FLAGS = flags.FLAGS


def read_nnet(nnetFile):
    '''
    Read a .nnet file and obtain the weights 
    
    Args:
        nnetFile (str): A .nnet file to convert to read from 
    '''
    
    # Open NNet file
    f = open(nnetFile,'r')
    
    # Skip header lines
    line = f.readline()
    while line[:2]=="//":
        line = f.readline()
        
    # Extract information about network architecture
    record = line.split(',')
    numLayers   = int(record[0])
    inputSize   = int(record[1])

    line = f.readline()
    record = line.split(',')
    layerSizes = np.zeros(numLayers+1,'int')
    for i in range(numLayers+1):
        layerSizes[i]=int(record[i])

    # Skip the normalization information
    # Unused parameter
    f.readline()
    line = f.readline()
    # Min 
    record = line.split(',')
    record = record[:-1]
    input_min = np.matrix([float(a) for a in record])


    # Max
    line = f.readline()
    record = line.split(',')
    record = record[:-1]
    input_max = np.matrix([float(a) for a in record])

    # Mean
    line = f.readline()
    record = line.split(',')
    record = record[:-1]
    input_mean = [float(a) for a in record]
    output_mean = input_mean[-1]
    input_mean = np.matrix(input_mean[:-1])
    # Mean

    line = f.readline()
    record = line.split(',')
    record = record[:-1]
    input_range = [float(a) for a in record]
    output_range = input_range[-1]
    input_range = np.matrix(input_range[:-1])

    # Initialize list of weights and biases
    weights = [np.zeros((layerSizes[i],layerSizes[i+1])) for i in range(numLayers)]
    biases  = [np.zeros(layerSizes[i+1]) for i in range(numLayers)]

    # Read remainder of file and place each value in the correct spot in a weight matrix or bias vector
    layer=0
    i=0
    j=0
    line = f.readline()
    record = line.split(',')
    while layer+1 < len(layerSizes):
        while i<layerSizes[layer+1]:
            while record[j]!="\n":
                weights[layer][j,i] = float(record[j])
                j+=1
            j=0
            i+=1
            line = f.readline()
            record = line.split(',')

        i=0
        while i<layerSizes[layer+1]:
            biases[layer][i] = float(record[0])
            i+=1
            line = f.readline()
            record = line.split(',')

        layer+=1
        i=0
        j=0
    f.close()
    return weights, biases, input_min, input_max, input_mean, input_range, output_mean, output_range


def create_checkpoint(weights, biases, input_min, input_max, input_mean, input_range, output_mean, output_range):
    model = ModelBasicMLP('model', input_shape=[5, 1], 
                          nb_classes=5, nb_layers=6, nb_hidden=[50, 50, 50, 50, 50, 50])
    var_list = model.get_params()
    index = 0
    init_ops = []
    for i in range(6):
        weight_var = var_list[index]
        bias_var = var_list[index + 1]
        index = index + 2 
        init_ops.append(tf.assign(weight_var, tf.constant(weights[i].astype(np.float32))))
        init_ops.append(tf.assign(bias_var, tf.constant(biases[i].astype(np.float32))))
    # Final layer (adding normalization)
    weight_var = var_list[index]
    bias_var = var_list[index+1]
    init_ops.append(tf.assign(weight_var, tf.constant((output_range*weights[6]).astype(np.float32))))
    init_ops.append(tf.assign(bias_var, tf.constant((output_range*biases[6] + output_mean).astype(np.float32))))
    return init_ops, model
    


def main(argv=None):
  config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))
  weights, biases, input_min, input_max, input_mean, input_range, output_mean, output_range = read_nnet(FLAGS.nnetFile)
  init_ops, model = create_checkpoint(weights, biases, input_min, input_max, input_mean, input_range, output_mean, output_range)

  
  # Opt variable 
  init_value = np.random.rand(1, 5)
  opt_x = tf.Variable(init_value, dtype=tf.float32, trainable=True)
  #opt_x = tf.placeholder(dtype=tf.float32, shape=[1, 5])

  norm_x = tf.minimum(opt_x, tf.convert_to_tensor(input_max, dtype=tf.float32))
  norm_x = tf.maximum(opt_x, tf.convert_to_tensor(input_min, dtype=tf.float32))
  norm_x = (opt_x - tf.convert_to_tensor(input_mean, dtype=tf.float32))/tf.convert_to_tensor(input_range, dtype=tf.float32)
  preds = model.get_logits(norm_x)
  objective = preds[0, 0]
  sess.run(tf.global_variables_initializer())
  sess.run(init_ops)
  np_input = np.matrix([15299.0,-1.142,-1.142,600.0,500.0])
  # np_input = (np_input - input_mean)/input_range

  #print(np_input)
  #val_obj = sess.run(objective, feed_dict={opt_x: np_input})
  #print(val_obj)
  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=0.0001)
  train_step = optimizer.minimize(-objective, var_list=[opt_x])
  sess.run(tf.global_variables_initializer())
  sess.run(init_ops) 

  input_min = np.ravel(input_min)
  input_max = np.ravel(input_max)
  input_min[0] = 55947
  input_min[3] = 1145
  input_max[4] = 60

  #print("Input min", input_min)
  #print("input max")
  proj1 = tf.assign(opt_x, tf.minimum(opt_x, tf.convert_to_tensor(input_max, dtype=tf.float32)))
  proj2 = tf.assign(opt_x, tf.maximum(opt_x, tf.convert_to_tensor(input_min, dtype=tf.float32)))
  proj_ops = tf.group([proj1, proj2])

  for i in range(100000):
      sess.run(train_step)
      sess.run(proj_ops)
      print(sess.run(objective)*output_range + output_mean)
  
  saver = tf.train.Saver()
  save_path = saver.save(sess, "../models/" + FLAGS.model_name + ".ckpt")
  print("Model saved in path: %s" % save_path)
  

if __name__ == '__main__':
  flags.DEFINE_string('nnetFile', None, 'Name of nnet file')
  flags.DEFINE_string('model_name', 'temp', 'Name of checkpoint')

  tf.app.run()

