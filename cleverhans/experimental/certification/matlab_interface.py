"""Code with matlab interface for the current problem"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np 
import scipy.io as sio
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import cg, cgs
from scipy.sparse.linalg import LinearOperator
import tensorflow as tf

from cleverhans.experimental.certification import utils
from cleverhans.experimental.certification import neural_net_params 


flags = tf.app.flags
FLAGS = flags.FLAGS

class MatlabInterface(object):
    """ Class to handle matlab interfacing for debugging certification"""

    def __init__(self, matlab_folder):
        self.folder = matlab_folder 
        if not tf.gfile.IsDirectory(matlab_folder):
            tf.gfile.MkDir(matlab_folder)
    
    def save_weights(self, nn_params, sess):
        numpy_weights = [sess.run(w) for w in nn_params.weights]
        numpy_biases = [sess.run(b) for b in nn_params.biases]
        numpy_net_sizes = nn_params.sizes
        sio.savemat(os.path.join(self.folder, 'weights.mat'), {'weights':numpy_weights})
        sio.savemat(os.path.join(self.folder, 'biases.mat'), {'biases':numpy_biases})
        sio.savemat(os.path.join(self.folder, 'sizes.mat'), {'sizes':numpy_net_sizes})
        
    def save_dual_params(self, dual_object, sess):
        dual_params = {}
        dual_params['test_input'] = sess.run(dual_object.test_input)
        dual_params['epsilon'] = sess.run(dual_object.epsilon)
        dual_params['input_minval'] = sess.run(dual_object.input_minval)
        dual_params['input_maxval'] = sess.run(dual_object.input_maxval)
        dual_params['true_class'] = dual_object.true_class
        dual_params['adv_class'] = dual_object.adv_class
        dual_params['final_linear'] = sess.run(dual_object.final_linear)
        dual_params['final_constant'] = sess.run(dual_object.final_constant)
        dual_params['lower'] = [sess.run(l) for l in dual_object.lower]
        dual_params['upper'] = [sess.run(u) for u in dual_object.upper]
        sio.savemat(os.path.join(self.folder, 'dual_params.mat'), {'dual_params':dual_params})
        
        
