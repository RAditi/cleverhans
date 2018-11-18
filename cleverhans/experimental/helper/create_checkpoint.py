from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy, WeightedSum
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans_tutorials.tutorial_models import ModelBasicMLP

FLAGS = flags.FLAGS
NB_LAYERS = 1 
BATCH_SIZE=50

def create_checkpoint(weights_dir, 
                      model_name,
                      nb_hidden, 
                      nb_layers=NB_LAYERS, 
                      train_start=0, 
                      train_end=60000,
                      test_start=0, 
                      test_end=10000, 
                      batch_size=BATCH_SIZE, 
                      label_smoothing=0.1, 
                      testing=False):
    
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  config_args = {}
  
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get MNIST data
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

  eval_params = {'batch_size': batch_size}
  fgsm_params = {
      'eps': 0.1,
      'clip_min': 0.,
      'clip_max': 1.
  }
  pgd_params = {
      'eps':0.1,
      'eps_iter': 0.05, 
      'nb_iter': 40, 
      'clip_min':0.,
      'clip_max':1.
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
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  model = ModelBasicMLP('model1', nb_classes, nb_layers, nb_hidden)
  preds = model.get_logits(x)
  loss = CrossEntropy(model, smoothing=label_smoothing)
  saver = tf.train.Saver()

  def evaluate():
    do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

  # train(sess, loss, x_train, y_train, evaluate=evaluate,
  #       args=train_params, rng=rng, var_list=model.get_params())
  # Loading weights 
  init_ops = []
  var_list = model.get_params()
  for layer in range(nb_layers + 1):
      weight_var = np.load(os.path.join(weights_dir, "W" + str(layer) + ".npy"))
      bias_var = np.load(os.path.join(weights_dir, "bias_W" + str(layer) + ".npy"))
      weight_var = weight_var.astype(np.float32)
      bias_var = bias_var.astype(np.float32)
      init_ops.append(tf.assign(var_list[2*layer], tf.constant(weight_var.T)))
      init_ops.append(tf.assign(var_list[2*layer+1], tf.constant(bias_var)))

  sess.run(init_ops)
  # Calculate training error
  if testing:
    do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

  # Initialize the Fast Gradient Sign Method (FGSM) attack object and
  # graph
  fgsm = FastGradientMethod(model, sess=sess)
  fgsm_adv_x = fgsm.generate(x, **fgsm_params)
  fgsm_preds_adv = model.get_logits(fgsm_adv_x)
  # Initalize PGD attack object 
  pgd = ProjectedGradientDescent(model, sess=sess) 
  pgd_adv_x = pgd.generate(x, **pgd_params)
  pgd_preds_adv = model.get_logits(pgd_adv_x)
  
    
  # Evaluate the accuracy of the MNIST model on adversarial examples
  do_eval(pgd_preds_adv, x_test, y_test, 'test_pgd_adv_eval', True)
  do_eval(fgsm_preds_adv, x_test, y_test, 'test_fgsm_adv_eval', True)
  do_eval(preds, x_test, y_test, 'test_clean_eval', False)

  save_path = saver.save(sess, "../models/" + model_name + ".ckpt")
  print("Model saved in path: %s" % save_path)

  # Calculate training error
  if testing:
    do_eval(pgd_preds_adv, x_train, y_train, 'train_clean_train_pgd_adv_eval')
    do_eval(fgsm_preds_adv, x_train, y_train, 'train_clean_train_fgsm_adv_eval')

  return report



def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)
  hidden_dimensions = [int(item) for item in FLAGS.nb_hidden.split(',')]
  create_checkpoint(weights_dir=FLAGS.weights_dir, 
                    model_name=FLAGS.model_name, 
                    nb_layers=FLAGS.nb_layers, 
                    nb_hidden=hidden_dimensions, 
                    batch_size=FLAGS.batch_size)
  

if __name__ == '__main__':
  flags.DEFINE_integer('nb_layers', NB_LAYERS,
                       'Number of hidden layers')
  flags.DEFINE_string('nb_hidden', None,
                       'Number of hidden nodes in each layer, separated by commas')
  flags.DEFINE_string('weights_dir', None, 'Directory with numpy weights')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                        'Size of training batches')
  flags.DEFINE_string('model_name', None, 'Nmae of the checkpoint')

  tf.app.run()
