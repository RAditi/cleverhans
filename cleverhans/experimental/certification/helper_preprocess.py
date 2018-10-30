""" 
File with util files for pre-processing a checkpoint 
For a tensorflow checkpoint as input: 
it computes the PGD test error and obtains indices of test points that can be certified
and creates a separate folder with these inputs with a text file containing mapping of true and adversarial label for each input 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans_tutorials.tutorial_models import ModelBasicMLP

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
# Aditi: Check what this does 
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
NB_LAYERS = 1
NB_HIDDEN = 100
  

def evaluate_model(ckpt_path, nb_layers, nb_hidden, 
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


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                      'Number of epochs to train model')
  flags.DEFINE_integer('nb_layers', NB_LAYERS,
                       'Number of hidden layers')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                        'Size of training batches')
  flags.DEFINE_integer('nb_hidden', NB_HIDDEN,
                       'Number of hidden nodes in each layer')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))
  flags.DEFINE_string('ckpt_path', None, 'Path of checkpoint to restore')

  evaluate_model(FLAGS.ckpt_path, FLAGS.nb_layers, FLAGS.nb_hidden)
  tf.app.run()
