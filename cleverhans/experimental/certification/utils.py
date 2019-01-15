"""File containing some simple helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os 
import scipy.io as sio 
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

def diag(diag_elements):
  """Function to create tensorflow diagonal matrix with input diagonal entries.

  Args:
    diag_elements: tensor with diagonal elements

  Returns:
    tf matrix with diagonal entries as diag_elements
  """
  return tf.diag(tf.reshape(diag_elements, [-1]))



def conv2ff(input_shape, layer_weight):
  """
  input_shape: 3 dimensional array of [num_rows, num_cols, num_channels]
  """

  input_num_elements = input_shape[0] * input_shape[1] * input_shape[2]
  kernel = layer_weight
  
  # TF graph to compute convolution
  flattened_input = tf.placeholder(tf.float32, shape=[1, input_num_elements])  # batch size = 1
  input_image = tf.reshape(flattened_input, [1] + input_shape)
  output_image = tf.nn.conv2d(input_image, kernel, strides=[1, 2, 2, 1], padding='SAME')
  flattened_output = tf.reshape(output_image, [1, -1])
  output_num_elements = int(flattened_output.shape[1])

  # construct the convolutional matrix
  conv_matrix = np.zeros([output_num_elements, input_num_elements], dtype=np.float32)
  with tf.Session() as sess:
    for i in range(input_num_elements):
      input_vector = np.zeros((1, input_num_elements), dtype=np.float32)
      input_vector[0, i] = 1.0
      output_vector = sess.run(flattened_output, feed_dict={flattened_input: input_vector})
      conv_matrix[:,i] = output_vector.flatten()

  # verify that result is the same
  random_input = np.random.random((1, input_num_elements))
  # with tf.gfile.Open(FLAGS.test_input) as f:
  #   test_input = np.load(f)
  with tf.Session() as sess:
    conv_output = sess.run(flattened_output, feed_dict={flattened_input: np.transpose(np.reshape(random_input, [-1, 1]))})
    matmul_output = np.reshape(np.matmul(conv_matrix, random_input.flatten()), (1, -1))

  print('Absolute difference between outputs: ', np.amax(np.abs(conv_output - matmul_output)))
  print('Relative difference between outputs: ', np.amax(np.abs(conv_output - matmul_output) / (np.abs(conv_output) + 1e-7)))
  return conv_matrix 


def l1_column(matrix):
  return tf.reduce_sum(tf.abs(matrix), axis=0)
