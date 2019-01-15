from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools 

import os
import logging
import numpy as np
import tensorflow as tf

import torch 
import torch.nn as nn

np.random.seed(3)

num_channels = 1
num_rows = 10
num_columns = 10
num_examples = 1
num_filters = 2

weights = np.random.rand(num_filters, num_channels, 4, 4)
bias = np.random.rand(num_filters)
x_value  = np.random.rand(num_examples, num_rows, num_columns, num_channels) 


#### Pytorch version #### 
# device = torch.device('cpu')
# x_torch = torch.from_numpy(x_value.transpose(0, 3, 1, 2)).float()
# conv1 = nn.Conv2d(1, num_filters, 4, stride=2, padding=1)
# params = [p for p in conv1.parameters()]
# params[0].data = torch.tensor(weights).float()
# params[1].data = torch.tensor(bias).float()
# y_conv = conv1(x_torch)
# relu = nn.ReLU()
# y_torch = relu(y_conv)
# y_torch = y_torch.view(y_torch.size(0), -1)
# torch_output = y_torch.data

x_tensor = tf.constant(x_value)

my_conv = functools.partial(
        tf.layers.conv2d, activation=tf.nn.relu)

y = my_conv(x_tensor, num_filters, kernel_size = 4, strides=[2,2], padding='same')
# y = tf.transpose(y, perm=[0, 3, 1, 2])
# y = tf.layers.flatten(y) 

variables = [v for v in tf.trainable_variables()]
init_ops = [tf.assign(variables[0], tf.constant(weights.transpose(2, 3, 1, 0)))]
init_ops.append(tf.assign(variables[1], tf.constant(bias)))

sess = tf.Session()
sess.run(init_ops)
tf_output = sess.run(tf.reshape(y, [-1, 1]))

# Matrix multiply version 
filter_tensor = tf.constant(weights.transpose(2, 3, 1, 0))
bias_tensor = tf.constant(bias)
custom_output = tf.nn.conv2d(x_tensor, filter_tensor, strides=[1,2,2,1], padding='SAME')
new_bias_tensor = tf.tile(tf.reshape(bias_tensor, [-1, 1]), [num_examples*5*5, 1])
custom_output = tf.reshape(custom_output, [-1, 1]) + new_bias_tensor 
custom_output = sess.run(custom_output)

print(np.shape(custom_output))
print(np.shape(tf_output))
print(np.max(np.ravel(np.abs(custom_output-tf_output))))
