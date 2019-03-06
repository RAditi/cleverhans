"""
Code to run adversarial training for different models and datasets 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.loss import CrossEntropy, WeightedSum, WeightDecay
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.dataset import CIFAR10, MNIST

# All parameters go here 
FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
EPS = 0.3
NB_GD_ITER = 15
EPS_ITER = EPS/10 
DATA_AUGMENTATION = False
LR_TYPE = 'exp_decay'
INIT_LEARNING_RATE = 0.001
EVALUATE_EPOCH = 10
CLEAN_WEIGHT = 1
ADV_WEIGHT = 0
MOMENTUM = 0.9
WEIGHT_DECAY_WEIGHT = 0.0002
LR_DECAY = 0.9

def adv_train(dataset='MNIST',
	      model_name='basicCNN', 
	      nb_epochs=NB_EPOCHS, 
	      batch_size=BATCH_SIZE,
	      learning_rate=LEARNING_RATE,
	      nb_filters=NB_FILTERS, 
	      data_augmentation=DATA_AUGMENTATION, 
	      num_threads=None,
	      label_smoothing=0.1):
    """
    Function to perform training (possibly adversarial)
    :param dataset: one of 'CIFAR, MNIST'
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

    #Loading data 
    if dataset=='MNIST':
  	data = MNIST()
    elif dataset=='CIFAR10':
  	data = CIFAR10()
    else:
  	logging.info("Dataset not implemented")
  	return

    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    print(dataset_train)
    # If performing data augmentation with random shifts and horizontal flip
    if data_augmentation:
  	dataset_train = dataset_train.map(
    	  lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')
    x_test, y_test = data.get_set('test')

    img_rows, img_cols, nb_channels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]
    
    # Remember that the input is scaled to [0, 1]
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nb_channels))
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

    # Generic evaluation function 
    def do_eval(preds, x_set, y_set, report_key):
        acc, loss = model_eval(sess, x, y, preds, final_loss, x_set, y_set, args=eval_params)
        setattr(report, report_key, acc)
        # if is_adv is None:
        #   report_text = None
        # elif is_adv:
        #   report_text = 'adversarial'
        # else:
        #   report_text = 'legitimate'
        print('Test accuracy on %s examples: %0.4f' % (report_key, acc))
        return acc, loss

    if model_name=='basicCNN':
  	model = ModelBasicCNN('model', nb_classes, nb_filters)
    elif model_name == 'wresnet':
	model = make_wresnet(nb_classes, input_shape=(None, img_rows, img_cols, nb_channels), scope=None)
    elif model_name== 'allConv':
  	model = modelAllConvolutional(scope, nb_classes, nb_filters, input_shape=(None, img_rows, img_cols, nb_channels))
    else: 
  	logging.info("Model not implemented")

    # Generated the adversarial logits 
    pgd = ProjectedGradientDescent(model, sess=sess)
    pgd_x = pgd.generate(x, **pgd_params)
    pgd_preds = model.get_logits(pgd_x)

    def attack(x):
        return pgd.generate(x, **pgd_params)

    clean_loss = CrossEntropy(model, smoothing=label_smoothing)
    adv_loss = CrossEntropy(model, smoothing=label_smoothing, attack=attack)
    weight_decay_loss = WeightDecay(model)

    final_loss = WeightedSum(model, ([FLAGS.clean_weight, clean_loss], [FLAGS.weight_decay_weight, weight_decay_loss]))

    preds = model.get_logits(x)
    adv_x = attack(x)
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = model.get_logits(adv_x)

    model_dir = FLAGS.model_dir 
    if not os.path.exists(model_dir):
        os.makedires(model_dir)
        
    saver = tf.train.Saver(max_to_keep=3)
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    global_step = tf.contrib.framework.get_or_create_global_step()
    if(FLAGS.lr_type == 'step'):
  	boundaries = [int(sss[0]) for sss in STEP_SIZE_SCHEDULE]
  	boundaries = boundaries[1:]
  	values = [sss[1] for sss in STEP_SIZE_SCHEDULE]
  	learning_rate = tf.train.piecewise_constant(
    	    tf.cast(global_step, tf.int32),
    	    boundaries,
    	    values)
    elif FLAGS.lr_type == 'exp_decay':
  	num_steps_epoch = ceil((train_end - train_start)/batch_size)
  	learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, 
  		                                   tf.cast(global_step, tf.int32), 
  		                                   num_steps_epoch, 
  		                                   FLAGS.lr_decay, 
                                                   staircase=False)
    else: 
        learning_rate = FLAGS.learning_rate

    if FLAGS.optimizer =='mom':
	momentum = FLAGS.momentum
  	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  
    elif FLAGS.optimizer=='adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)

    elif FLAGS.optimizer == 'rmsprop':
  	momentum = FLAGS.momentum
  	decay = 0.9
  	optimizer = tf.train.RMSPropOptimizer(learning_rate, 
  		                              decay, 
  		                              momentum)
    else:
  	optimizer = tf.train.AdamOptimizer(learning_rate)

    
    def evaluate():
        # Accuracy of adversarially trained model on legitimate test inputs
        test_clean_accuracy, test_clean_loss = do_eval(preds, x_test, y_test, 'test_clean_eval')
        # Accuracy of the adversarially trained model on adversarial examples
        test_adv_accuracy, test_adv_loss = do_eval(preds_adv, x_test, y_test, 'test_adv_eval')
        # Accuracy on train clean 
        train_clean_accuracy, train_clean_loss = do_eval(preds, x_train, y_train, 'train_clean_eval')
        # Accuracy on train adv 
        train_adv_accuracy, train_adv_loss = do_eval(preds_adv, x_train, y_train, 'train_adv_eval')
    
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='loss train', simple_value=train_clean_loss),
            tf.Summary.Value(tag='loss test', simple_value=test_clean_loss),
            tf.Summary.Value(tag='loss adv train', simple_value=train_adv_loss),
            tf.Summary.Value(tag='loss adv test', simple_value=test_adv_loss),
            tf.Summary.Value(tag='accuracy train', simple_value= train_clean_accuracy),
            tf.Summary.Value(tag='accuracy adv train', simple_value= train_adv_accuracy),
            tf.Summary.Value(tag='accuracy adv test', simple_value= test_adv_accuracy),
            tf.Summary.Value(tag='accuracy test', simple_value= test_clean_accuracy)])
        summary_writer.add_summary(summary, global_step.eval(sess))

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
    adv_train(dataset=FLAGS.dataset, 
	      model_name=FLAGS.model_name, 
	      nb_epochs=FLAGS.nb_epochs, 
	      batch_size=FLAGS.batch_size,
	      learning_rate=FLAGS.learning_rate,
	      nb_filters=FLAGS.nb_filters, 
	      data_augmentation=FLAGS.data_augmentation)
    
if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', NB_FILTERS,
                         'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE,
                         'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    # flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
    #                   ('If True, backprop through adversarial example '
    #                    'construction process during adversarial training'))
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
    flags.DEFINE_string('lr_type', 'One of exp_decay or step',
                        'Directory to store results')
    flags.DEFINE_float('init_learning_rate', INIT_LEARNING_RATE, 
  	               'Initial learning rate')
    flags.DEFINE_float('lr_decay', LR_DECAY, 
  	               'Initial learning rate')
    flags.DEFINE_string('optimizer', 'adam', 'Which optimizer to use')
    flags.DEFINE_string('dataset', 'MNIST', 'Which dataset')
    flags.DEFINE_string('model_name', 'basicCNN', 'Which model')
    flags.DEFINE_bool('data_augmentation', False,
                      'whether to use data augmentation')

    tf.app.run()
