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
from cleverhans_tutorials.tutorial_models import ModelBasicCNN, ModelSmallCNN, ModelVerySmallCNN
from cleverhans_tutorials.tutorial_models import ModelBasicMLP

FLAGS = flags.FLAGS
BATCH_SIZE = 128
NB_FILTERS = 16
NB_LAYERS = 1
  

def evaluate_model(save_path, ckpt_path, model_type, model_name, 
                   batch_size, nb_layers, nb_hidden, nb_filters, 
                   train_start=0, train_end=60000,
                   test_start=0, test_end=10000):

  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  np.random.seed(3)
  
  p = np.random.permutation(10000);
  x_test = x_test[p, :];
  y_test = y_test[p, :];

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  with tf.Session() as sess:
    if(model_type == 'fc'):
      model = ModelBasicMLP('model1', nb_classes, nb_layers, nb_hidden)
    elif (model_type == 'small_cnn'):
      model = ModelSmallCNN('model1', nb_classes, nb_hidden, nb_filters)
    elif (model_type == 'very_small_cnn'):
      model = ModelVerySmallCNN('model2', nb_classes, nb_hidden, nb_filters)

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
    pgd_predictions = sess.run(pgd_preds_adv, feed_dict={x:x_test})
    test_predictions = sess.run(preds, feed_dict={x:x_test})
    np.save(os.path.join(save_path, 'PGD_logits'), pgd_predictions.astype(float))
    np.save(os.path.join(save_path, 'test_logits'), test_predictions.astype(float))
    np.savetxt(os.path.join(save_path, 'PGD_indices'), np.ravel(pgd_correct_indices).astype(int))
    

def create_json(save_path, ckpt_path, model_type, model_name , nb_layers=1):
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  
  for key in var_to_shape_map:
    print("tensor_name: ", key)
    
  layer_info = []
  if(model_type=='fc'):
    first_layer_info = {}
    first_layer_info["weight_var"] = model_name + "/dense/kernel"
    first_layer_info["bias_var"] = model_name + "/dense/bias"
    first_layer_info["type"] = "ff_relu"
    first_layer_info["is_transpose"] = True
    layer_info.append(first_layer_info)

    for i in range(1, nb_layers):
      current_layer_info = {}
      current_layer_info["weight_var"] = model_name + "/dense_" + str(i) + "/kernel"
      current_layer_info["bias_var"] = model_name + "/dense_" +str(i) + "/bias"
      current_layer_info["type"] = "ff_relu"
      current_layer_info["is_transpose"] = True
      layer_info.append(current_layer_info)

    last_layer_info = {}
    last_layer_info["weight_var"] = model_name + "/dense_" + str(nb_layers) + "/kernel"
    last_layer_info["bias_var"] = model_name + "/dense_" + str(nb_layers) + "/bias"
    last_layer_info["type"] = "ff"
    last_layer_info["is_transpose"] = True
    layer_info.append(last_layer_info)
    
  elif (model_type=='small_cnn'):
    first_layer_info = {}
    first_layer_info["weight_var"] = model_name + "/conv2d/kernel"
    first_layer_info["bias_var"] = model_name + "/conv2d/bias"
    first_layer_info["type"] = "conv"
    first_layer_info["is_transpose"] = True
    layer_info.append(first_layer_info)

    second_layer_info = {}
    second_layer_info["weight_var"] = model_name + "/conv2d_1/kernel"
    second_layer_info["bias_var"] = model_name + "/conv2d_1/bias"
    second_layer_info["type"] = "conv"
    second_layer_info["is_transpose"] = True
    layer_info.append(second_layer_info)

    third_layer_info = {}
    third_layer_info["weight_var"] = model_name + "/dense/kernel"
    third_layer_info["bias_var"] = model_name + "/dense/bias"
    third_layer_info["type"] = "ff_relu"
    third_layer_info["is_transpose"] = True
    layer_info.append(third_layer_info)

    fourth_layer_info = {}
    fourth_layer_info["weight_var"] = model_name + "/dense_1/kernel"
    fourth_layer_info["bias_var"] = model_name + "/dense_1/bias"
    fourth_layer_info["type"] = "ff"
    fourth_layer_info["is_transpose"] = True
    layer_info.append(fourth_layer_info)

  elif (model_type=='very_small_cnn'):
    first_layer_info = {}
    first_layer_info["weight_var"] = model_name + "/conv2d/kernel"
    first_layer_info["bias_var"] = model_name + "/conv2d/bias"
    first_layer_info["type"] = "conv"
    first_layer_info["is_transpose"] = True
    layer_info.append(first_layer_info)

    second_layer_info = {}
    second_layer_info["weight_var"] = model_name + "/dense/kernel"
    second_layer_info["bias_var"] = model_name + "/dense/bias"
    second_layer_info["type"] = "ff"
    second_layer_info["is_transpose"] = True
    layer_info.append(second_layer_info)


  with open(os.path.join(save_path, 'description.json'), 'w') as outfile:  
    json.dump(layer_info, outfile)

def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)
  if not os.path.isdir(FLAGS.save_path):
    os.mkdir(FLAGS.save_path)

  hidden_dimensions = [int(item) for item in FLAGS.nb_hidden.split(',')]    
  create_json(FLAGS.save_path, FLAGS.ckpt_path, FLAGS.model_type, FLAGS.model_name, FLAGS.nb_layers)
  evaluate_model(FLAGS.save_path, FLAGS.ckpt_path, FLAGS.model_type, FLAGS.model_name, FLAGS.batch_size, 
                 FLAGS.nb_layers, hidden_dimensions, FLAGS.nb_filters)
  

if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_layers', NB_LAYERS,
                       'Number of hidden layers')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                        'Size of training batches')
  flags.DEFINE_string('nb_hidden', None,
                       'Number of hidden nodes in each layer')
  flags.DEFINE_string('ckpt_path', None, 'Path of checkpoint to restore')
  flags.DEFINE_string('model_name', None, 'Name of model')
  flags.DEFINE_string('save_path', None, 
                      'Folder to save the json file and attack performance')
  flags.DEFINE_string('model_type', 'fc', 
                      'Type of the model being processed')


  tf.app.run()
