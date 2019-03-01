"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import numpy as np
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy, WeightedSum, MyWeightDecay
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet, decay
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
EPS = 0.01
NB_GD_ITER = 15
EPS_ITER =  EPS/5.0
CLEAN_WEIGHT = 1
ADV_WEIGHT = 1
MOMENTUM = 0.9
WEIGHT_DECAY_WEIGHT = 0.0002
STEP_SIZE_SCHEDULE = [[0, 0.001], [40000, 0.0001], [60000, 0.00001]]
EVALUATE_EPOCH = 10

def cifar10_adv_train(train_start=0, train_end=60000, test_start=0,
                     test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE,
                     nb_filters=NB_FILTERS, 
                     num_threads=None,
                     label_smoothing=0.1):
  """
  CIFAR10 cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Remember that the input is scaled to [0, 1]
  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an MNIST model
  train_params = {
      'nb_epochs': FLAGS.nb_epochs,
      'batch_size': FLAGS.batch_size,
      'learning_rate': FLAGS.learning_rate
  }
  eval_params = {'batch_size': batch_size}
  pgd_params = {
      'eps': FLAGS.eps,
      'eps_iter': FLAGS.eps_iter,
      'nb_iter': FLAGS.nb_gd_iter,  
      'clip_min': 0.,
      'clip_max': 1.
  }
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('%s : %0.4f' % (report_key, acc))
    return acc

  # Create a new model and train it to be robust to PGD 
  model = make_wresnet(nb_classes=10, input_shape=(None, 32, 32, 3), scope='test')
  # model = ModelAllConvolutional('model1', 10, 10,
  #                               input_shape=[32, 32, 3])

  
  # Generated the adversarial logits 
  pgd = ProjectedGradientDescent(model, sess=sess)
  pgd_x = pgd.generate(x, **pgd_params)
  pgd_preds = model.get_logits(pgd_x)
  
  def attack(x):
    return pgd.generate(x, **pgd_params)

  adv_loss = CrossEntropy(model, smoothing=label_smoothing, attack=attack)
  clean_loss = CrossEntropy(model, smoothing=label_smoothing)
  weight_decay_loss = MyWeightDecay(model)
  
  final_loss = WeightedSum(model, ([FLAGS.clean_weight, adv_loss], [FLAGS.weight_decay_weight, weight_decay_loss]))
  
  preds = model.get_logits(x)
  adv_x = attack(x)
  adv_x = tf.stop_gradient(adv_x)
  preds_adv = model.get_logits(adv_x)

  model_dir = FLAGS.model_dir 
  if not os.path.exists(model_dir):
      os.makedirs(model_dir)

  saver = tf.train.Saver(max_to_keep=3)
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)


  def evaluate():
    # Accuracy of adversarially trained model on legitimate test inputs
    test_clean_accuracy = do_eval(preds, x_test, y_test, 'test_clean_eval', False)
    # Accuracy of the adversarially trained model on adversarial examples
    test_adv_accuracy = do_eval(preds_adv, x_test, y_test, 'test_adv_eval', True)
    # Accuracy on train clean 
    train_clean_accuracy = do_eval(preds, x_train, y_train, 'train_clean_eval', False)
    # Accuracy on train adv 
    train_adv_accuracy = do_eval(preds_adv, x_train, y_train, 'train_adv_eval', True)

    summary = tf.Summary(value=[
      # tf.Summary.Value(tag='loss train', simple_value= average_train_loss),
      # tf.Summary.Value(tag='loss test', simple_value= average_test_loss),
      # tf.Summary.Value(tag='loss adv train', simple_value= average_adv_train_loss),
      # tf.Summary.Value(tag='loss adv test', simple_value= average_adv_test_loss),
      tf.Summary.Value(tag='accuracy train', simple_value= train_clean_accuracy),
      tf.Summary.Value(tag='accuracy adv train', simple_value= train_adv_accuracy),
      tf.Summary.Value(tag='accuracy adv test', simple_value= test_adv_accuracy),
      tf.Summary.Value(tag='accuracy test', simple_value= test_clean_accuracy)])
    summary_writer.add_summary(summary, global_step.eval(sess))


  # Setting up optimizer
  boundaries = [int(sss[0]) for sss in STEP_SIZE_SCHEDULE]
  boundaries = boundaries[1:]
  values = [sss[1] for sss in STEP_SIZE_SCHEDULE]
  global_step = tf.contrib.framework.get_or_create_global_step()
  learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
  momentum = FLAGS.momentum
  optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  
  # Perform and evaluate adversarial training
  train(sess, final_loss, None, None,
        dataset_train=dataset_train, dataset_size=dataset_size,
        evaluate=evaluate, evaluate_epoch=FLAGS.evaluate_epoch,
        optimizer=optimizer, 
        args=train_params, rng=rng,
        var_list=model.get_params())

    
  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  cifar10_adv_train(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))
  flags.DEFINE_float('eps', EPS,
                       'Size of perturbation')
  flags.DEFINE_float('eps_iter', EPS_ITER,
                       'Size of perturbation at each step')
  flags.DEFINE_float('clean_weight', CLEAN_WEIGHT,
                       'Weight of loss of clean examples')
  flags.DEFINE_float('adv_weight', ADV_WEIGHT,
                       'Weight of loss of adversarial examples')
  flags.DEFINE_float('momentum', MOMENTUM,
                       'Momentum for optimizer')
  flags.DEFINE_float('weight_decay_weight', WEIGHT_DECAY_WEIGHT,
                       'Penalty of weight decay')
  flags.DEFINE_integer('nb_gd_iter', NB_GD_ITER,
                       'Number of gradient descent iterations')
  flags.DEFINE_integer('evaluate_epoch', EVALUATE_EPOCH, 
                       'After how many epochs to evaluate')
  flags.DEFINE_string('model_dir', 'temp',
                       'Directory to store results')

  

  tf.app.run()
