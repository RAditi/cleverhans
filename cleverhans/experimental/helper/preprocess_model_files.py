""" 
File with util files for pre-processing a checkpoint and setting up 
for certification
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python import pywrap_tensorflow

from cleverhans.loss import CrossEntropy, WeightedSum
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans_tutorials.tutorial_models import ModelBasicMLP

FLAGS = flags.FLAGS
BATCH_SIZE = 128
NB_FILTERS = 64
NB_LAYERS = 1
NB_HIDDEN = 100
  

def evaluate_model(save_path, ckpt_path, batch_size, nb_layers, nb_hidden, 
                  train_start=0, train_end=60000,
                  test_start=0, test_end=10000):

  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  with tf.Session() as sess:
    model = ModelBasicMLP('model2', nb_classes, nb_layers, nb_hidden)
    fgsm = FastGradientMethod(model, sess=sess)
    pgd = ProjectedGradientDescent(model, sess=sess)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    print('Model successfully restored')
    preds = model.get_logits(x)
    eval_params = {'batch_size': batch_size, 'is_correct_indices':True}
    clean_acc, clean_correct_indices =  model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    print('Clean test accuracy: %0.4f' % (clean_acc))

    fgsm_params = {
      'eps': 0.1,
      'clip_min': 0.,
      'clip_max': 1.
    } 
    pgd_params = {
      'eps':0.1, 
      'clip_min':0., 
      'clip_max':1.
    } 

    fgsm_adv_x = fgsm.generate(x, **fgsm_params)
    fgsm_preds_adv = model.get_logits(fgsm_adv_x)
    pgd_adv_x = pgd.generate(x, **pgd_params)
    pgd_preds_adv = model.get_logits(pgd_adv_x)

    fgsm_acc, fgsm_correct_indices = model_eval(sess, x, y, fgsm_preds_adv, x_test, y_test, args=eval_params)
    print('FGSM Test accuracy: %0.4f' % (fgsm_acc))
    np.savetxt(os.path.join(save_path, 'FGSM_indices'), np.ravel(fgsm_correct_indices).astype(int))

    pgd_acc, pgd_correct_indices  = model_eval(sess, x, y, pgd_preds_adv, x_test, y_test, args=eval_params) 
    print('PGD accuracy: %0.4f' % (pgd_acc))
    np.savetxt(os.path.join(save_path, 'PGD_indices'), np.ravel(pgd_correct_indices).astype(int))
  

def create_json(save_path, ckpt_path, nb_layers):
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  
  for key in var_to_shape_map:
    print("tensor_name: ", key)
    
  layer_info = []
  for i in range(nb_layers):
    current_layer_info = {}
    current_layer_info["weight_var"] = "model2/dense/kernel"
    current_layer_info["bias_var"] = "model2/dense/bias"
    current_layer_info["type"] = "ff_relu"
    current_layer_info["is_transpose"] = True
    layer_info.append(current_layer_info)

  last_layer_info = {}
  last_layer_info["weight_var"] = "model2/logits/kernel"
  last_layer_info["bias_var"] = "model2/logits/bias"
  last_layer_info["type"] = "ff"
  last_layer_info["is_transpose"] = True
  layer_info.append(last_layer_info)

  with open(os.path.join(save_path, 'description.json'), 'w') as outfile:  
    json.dump(layer_info, outfile)
  

def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)
  if not os.path.isdir(FLAGS.save_path):
    os.mkdir(FLAGS.save_path)
    
  create_json(FLAGS.save_path, FLAGS.ckpt_path, FLAGS.nb_layers)
  evaluate_model(FLAGS.save_path, FLAGS.ckpt_path, FLAGS.batch_size, 
                 FLAGS.nb_layers, FLAGS.nb_hidden)
  

if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_layers', NB_LAYERS,
                       'Number of hidden layers')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                        'Size of training batches')
  flags.DEFINE_integer('nb_hidden', NB_HIDDEN,
                       'Number of hidden nodes in each layer')
  flags.DEFINE_string('ckpt_path', None, 'Path of checkpoint to restore')
  flags.DEFINE_string('save_path', None, 
                      'Folder to save the json file and attack performance')


  tf.app.run()
