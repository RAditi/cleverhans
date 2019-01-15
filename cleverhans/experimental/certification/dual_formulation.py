"""Code with dual formulation for certification problem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np 
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import cg, cgs
from scipy.sparse.linalg import LinearOperator
import tensorflow as tf
from cleverhans.experimental.certification import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

class DualFormulation(object):
  """DualFormulation is a class that creates the dual objective function
  and access to matrix vector products for the matrix that is constrained
  to be Positive semidefinite """

  def __init__(self, sess, neural_net_param_object, test_input, true_class,
               adv_class, input_minval, input_maxval, epsilon, new_init=False):
    """Initializes dual formulation class.

    Args:
      dual_var: dictionary of dual variables containing a) lambda_pos
        b) lambda_neg, c) lambda_quad, d) lambda_lu
      neural_net_param_object: NeuralNetParam object created for
        the network under consideration
      test_input: clean example to certify around
      true_class: the class label of the test input
      adv_class: the label that the adversary tried to perturb input to
      input_minval: minimum value of valid input range
      input_maxval: maximum value of valid input range
      epsilon: Size of the perturbation (scaled for [0, 1] input)
    """
    self.sess = sess
    self.nn_params = neural_net_param_object
    self.test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
    self.true_class = true_class
    self.adv_class = adv_class
    self.input_minval = tf.convert_to_tensor(input_minval, dtype=tf.float32)
    self.input_maxval = tf.convert_to_tensor(input_maxval, dtype=tf.float32)
    self.epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    self.final_linear = (self.nn_params.final_weights[adv_class, :]
                         - self.nn_params.final_weights[true_class, :])
    self.final_linear = tf.reshape(self.final_linear,
                                   shape=[tf.size(self.final_linear), 1])
    self.final_constant = (self.nn_params.final_bias[adv_class]
                           - self.nn_params.final_bias[true_class])

    # Computing lower and upper bounds
    # Note that lower and upper are of size nn_params.num_hidden_layers + 1
    if(not new_init):
      self.lower = []
      self.upper = []
      self.pre_lower = []
      self.pre_upper = []

      # Initializing at the input layer with \ell_\infty constraints
      self.lower.append(
        tf.maximum(self.test_input - self.epsilon, self.input_minval))
      self.upper.append(
        tf.minimum(self.test_input + self.epsilon, self.input_maxval))
      self.pre_upper.append(self.upper[0])
      self.pre_lower.append(self.lower[0])

      for i in range(0, self.nn_params.num_hidden_layers):
        current_lower = 0.5*(
          self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
          + self.nn_params.forward_pass(self.lower[i] - self.upper[i], i,
                                        is_abs=True)) + self.nn_params.biases[i]
        current_upper = 0.5*(
          self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
          + self.nn_params.forward_pass(self.upper[i] -self.lower[i], i,
                                        is_abs=True)) + self.nn_params.biases[i]
        self.pre_lower.append(current_lower)
        self.pre_upper.append(current_upper)
        self.lower.append(tf.nn.relu(current_lower))
        self.upper.append(tf.nn.relu(current_upper))
    
    else :
      pass
      # # For fast passes through the network 
      # self.layer_inputs = []
      # self.layer_outputs = []
      # for i in range(self.nn_params.num_hidden_layers):
      #   self.layer_inputs.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i],1)))
      #   self.layer_outputs.append(self.nn_params.forward_pass(self.layer_inputs[i], i))

      # self.lower = []
      # self.upper = []
      # self.switch_indices = []
      # current_lower = sess.run(tf.maximum(self.test_input - self.epsilon, self.input_minval))
      # current_upper = sess.run(tf.minimum(self.test_input + self.epsilon, self.input_maxval))
      # switch_indices = (np.multiply(current_lower, current_upper) > 0).astype(int)
      # self.lower.append(np.reshape(current_lower, [-1, 1]))
      # self.upper.append(np.reshape(current_upper, [-1, 1]))
      # self.switch_indices.append(np.reshape(switch_indices, [-1, 1]))
      
      # first_matrix = np.eye(self.nn_params.sizes[0])
      # first_matrix_cols = []
      # for i in range(self.nn_params.sizes[0]):
      #   first_matrix_cols.append(sess.run(self.layer_outputs[0], feed_dict={self.layer_inputs[0]:np.reshape(first_matrix[:, i], [-1, 1])}))
      # first_matrix = np.hstack(first_matrix_cols)

    self.positive_indices = []
    self.negative_indices = []
    self.switch_indices = []

    for i in range(0, self.nn_params.num_hidden_layers + 1):
      self.positive_indices.append(tf.cast(self.pre_lower[i] >= 0, dtype=tf.float32))
      self.negative_indices.append(tf.cast(self.pre_upper[i] <= 0, dtype=tf.float32))
      self.switch_indices.append(tf.cast(tf.multiply(self.pre_lower[i], self.pre_upper[i])<0, dtype=tf.float32))
      # self.positive_indices.append(tf.zeros([self.nn_params.sizes[i], 1], tf.float32))
      # self.negative_indices.append(tf.zeros([self.nn_params.sizes[i], 1], tf.float32))
      # self.switch_indices.append(tf.ones([self.nn_params.sizes[i], 1], tf.float32))

    print("Positive size", np.sum(sess.run(self.positive_indices[i])))
    print("Negative size", np.sum(sess.run(self.negative_indices[i])))
    print("Switch size", np.sum(sess.run(self.switch_indices[i])))

    # Computing the optimization terms
    self.vector_g = None
    self.scalar_f = None
    self.matrix_h = None
    self.matrix_m = None
    self.matrix_m_dimension = 1 + np.sum(self.nn_params.sizes)
    
    # The primal vector in the SDP can be thought of as [layer_1, layer_2..]
    # In this concatenated version, dual_index[i] that marks the start
    # of layer_i
    # This is useful while computing implicit products with matrix H
    self.dual_index = [0]
    for i in range(self.nn_params.num_hidden_layers + 1):
      self.dual_index.append(self.dual_index[-1] +
                             self.nn_params.sizes[i])

  def initialize_dual(self, name, init_dual_folder=None,
                    random_init_variance=0.0, init_nu=200.0):
    """Function to initialize the dual variables of the class.
    
    Args:
    neural_net_params_object: Object with the neural net weights, biases
    and types
    positive_indices: Indices where l and u are both positive 
    negative_indices: Indices where l and u are both negative 
    switch_indices: Indices where l and u are of opposite sign 
    init_dual_file: Path to file containing dual variables, if the path
    is empty, perform random initialization
    Expects numpy dictionary with
    lambda_pos_0, lambda_pos_1, ..
    lambda_neg_0, lambda_neg_1, ..
    lambda_quad_0, lambda_quad_1, ..
    lambda_lu_0, lambda_lu_1, ..
    random_init_variance: variance for random initialization
    init_nu: Value to initialize nu variable with
    
    Returns:
    dual_var: dual variables initialized appropriately.
    """
    self.lambda_pos = []
    self.lambda_neg = []
    self.lambda_quad = []
    self.lambda_lu = []

    if init_dual_folder is None:
      for i in range(0, self.nn_params.num_hidden_layers + 1):
        # Lambda_pos
        initializer = (np.random.uniform(0, random_init_variance, size=
                                       (self.nn_params.sizes[i], 1))).astype(np.float32)
        if(FLAGS.use_matlab or False):
          initializer = sio.loadmat('matlab_vs_cnn/lambda_pos_' + str(i+1))
          initializer = initializer['val_lambda_pos'].astype(np.float32)

        # current_lambda_pos = []
        # trainable_values = self.sess.run(1 - self.negative_indices[i]).astype(bool)
        # for j in range(self.nn_params.sizes[i]):
        #   current_lambda_pos.append(tf.get_variable('lambda_pos_' + str(i) + '_' + str(j), 
        #                                             initializer = initializer[i], 
        #                                             dtype=tf.float32, 
        #                                             trainable=trainable_values[j]))
        # self.lambda_pos.append(tf.reshape(tf.concat(current_lambda_pos, axis=0),  [-1, 1]))
        self.lambda_pos.append(tf.get_variable(name+'lambda_pos_' + str(i), initializer=initializer, 
                                               dtype=tf.float32))

        # Lambda_neg
        initializer = (np.random.uniform(0, random_init_variance, size=(
          self.nn_params.sizes[i], 1))).astype(np.float32)

        if(FLAGS.use_matlab or False):
          initializer = sio.loadmat('matlab_vs_cnn/lambda_neg_' + str(i+1))
          initializer = initializer['val_lambda_neg'].astype(np.float32)

        # current_lambda_neg = []
        # trainable_values = self.sess.run(1 - self.positive_indices[i]).astype(bool)

        # for j in range(self.nn_params.sizes[i]):
        #   current_lambda_neg.append(tf.get_variable('lambda_neg_' + str(i) + '_' + str(j), 
        #                                             initializer = initializer[i], 
        #                                             dtype=tf.float32, 
        #                                             trainable=trainable_values[j]))
        # self.lambda_neg.append(tf.reshape(tf.concat(current_lambda_neg, axis=0), [-1, 1]))
        self.lambda_neg.append(tf.get_variable(name+'lambda_neg_' + str(i), initializer=initializer, 
                                               dtype=tf.float32))

        # Lambda_quad
        initializer = (np.random.uniform(0, random_init_variance, size=(
          self.nn_params.sizes[i], 1))).astype(np.float32)
        if(FLAGS.use_matlab or False):
          initializer = sio.loadmat('matlab_vs_cnn/lambda_quad_' + str(i+1))
          initializer = initializer['val_lambda_quad'].astype(np.float32)

        # current_lambda_quad = []
        # trainable_values = self.sess.run(self.switch_indices[i]).astype(bool)

        # for j in range(self.nn_params.sizes[i]):
        #   current_lambda_quad.append(tf.get_variable('lambda_quad_' + str(i) + '_' + str(j), 
        #                                             initializer = initializer[i], 
        #                                             dtype=tf.float32, 
        #                                             trainable=trainable_values[j]))
        # self.lambda_quad.append(tf.reshape(tf.concat(current_lambda_quad, axis=0), [-1, 1]))
        self.lambda_quad.append(tf.get_variable(name+'lambda_quad_' + str(i), initializer=initializer, 
                                                dtype=tf.float32))
      
        #Lambda_lu
        initializer = (np.random.uniform(0, random_init_variance, size=(
          self.nn_params.sizes[i], 1))).astype(np.float32)
        # initializer = np.ones((self.nn_params.sizes[i], 1)).astype(np.float32)
        if(FLAGS.use_matlab or False):
          initializer = sio.loadmat('matlab_vs_cnn/lambda_lu_' + str(i+1))
          initializer = initializer['val_lambda_lu'].astype(np.float32)
        self.lambda_lu.append(tf.get_variable(name+'lambda_lu_' + str(i),
                                       initializer=initializer,
                                       dtype=tf.float32))

      if(FLAGS.use_matlab or False):
        init_nu = sio.loadmat('matlab_vs_cnn/nu.mat')
        init_nu = init_nu['val_nu'].astype(np.float32)
      nu = tf.get_variable(name+'nu', initializer=init_nu)
      self.nu = tf.reshape(nu, shape=(1, 1))
    else:
      # Loading from folder
      init_lambda_pos = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_pos.npy'))
      init_lambda_neg = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_neg.npy'))
      init_lambda_quad = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_quad.npy'))
      init_lambda_lu = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_lu.npy'))
      init_nu = np.load(os.path.join(FLAGS.init_dual_folder, 'nu.npy'))

      for i in range(0, self.nn_params.num_hidden_layers + 1):
        self.lambda_pos.append(
          tf.get_variable('lambda_pos_' + str(i),
                          initializer=init_lambda_pos[i],
                          dtype=tf.float32))
        self.lambda_neg.append(
          tf.get_variable('lambda_neg_' + str(i),
                          initializer=init_lambda_neg[i],
                          dtype=tf.float32))
        self.lambda_quad.append(
          tf.get_variable('lambda_quad_' + str(i),
                          initializer=init_lambda_quad[i],
                          dtype=tf.float32))
        self.lambda_lu.append(
          tf.get_variable('lambda_lu_' + str(i),
                          initializer=init_lambda_lu[i],
                          dtype=tf.float32))
      nu = tf.get_variable('nu', initializer=1.0*init_nu)
      self.nu = tf.reshape(nu, shape=(1, 1))
      self.dual_var = {'lambda_pos': self.lambda_pos, 'lambda_neg': self.lambda_neg,
              'lambda_quad': self.lambda_quad, 'lambda_lu': self.lambda_lu, 'nu': self.nu}


  def initialize_placeholder(self):
    """ Function to intialize dual placeholders """
    self.lambda_pos = []
    self.lambda_neg = []
    self.lambda_quad = []
    self.lambda_lu = []


    for i in range(0, self.nn_params.num_hidden_layers + 1):
      # Lambda_pos
      self.lambda_pos.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
      # Lambda_neg
      self.lambda_neg.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
      # Lambda_quad 
      self.lambda_quad.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
      # Lambda_lu
      self.lambda_lu.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))

    self.nu = tf.placeholder(tf.float32, shape=(1, 1))


  def set_differentiable_objective(self):
    """Function that constructs minimization objective from dual variables."""
    # Checking if graphs are already created
    if self.vector_g is not None:
      return

    # Computing the scalar term
    bias_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers):
      bias_sum = bias_sum + tf.reduce_sum(
          tf.multiply(self.nn_params.biases[i], self.lambda_pos[i+1]))
    lu_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers+1):
      lu_sum = lu_sum + tf.reduce_sum(
          tf.multiply(tf.multiply(self.lower[i], self.upper[i]),
                      self.lambda_lu[i]))

    self.scalar_f = - bias_sum - lu_sum + self.final_constant

    # Computing the vector term
    g_rows = []
    for i in range(0, self.nn_params.num_hidden_layers):
      if i > 0:
        current_row = (self.lambda_neg[i] + self.lambda_pos[i] -
                       self.nn_params.forward_pass(self.lambda_pos[i+1],
                                                   i, is_transpose=True) +
                       tf.multiply(self.lower[i]+self.upper[i],
                                   self.lambda_lu[i]) +
                       tf.multiply(self.lambda_quad[i],
                                   self.nn_params.biases[i-1]))
      else:
        current_row = (-self.nn_params.forward_pass(self.lambda_pos[i+1],
                                                    i, is_transpose=True)
                       + tf.multiply(self.lower[i]+self.upper[i],
                                     self.lambda_lu[i]))
      g_rows.append(current_row)

    # Term for final linear term
    g_rows.append((self.lambda_pos[self.nn_params.num_hidden_layers] +
                   self.lambda_neg[self.nn_params.num_hidden_layers] +
                   self.final_linear +
                   tf.multiply((self.lower[self.nn_params.num_hidden_layers]+
                                self.upper[self.nn_params.num_hidden_layers]),
                               self.lambda_lu[self.nn_params.num_hidden_layers])
                   + tf.multiply(
                       self.lambda_quad[self.nn_params.num_hidden_layers],
                       self.nn_params.biases[
                           self.nn_params.num_hidden_layers-1])))
    self.vector_g = tf.concat(g_rows, axis=0)
    self.unconstrained_objective = self.scalar_f + 0.5*self.nu 

  def get_psd_product(self, vector):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix M

    Returns:
      result_product: Matrix product of M and vector
    """
    # For convenience, think of x as [\alpha, \beta]
    alpha = tf.reshape(vector[0], shape=[1, 1])
    beta = vector[1:]
    # Computing the product of matrix_h with beta part of vector
    # At first layer, h is simply diagonal
    h_beta_rows = []
    for i in range(self.nn_params.num_hidden_layers):
      # Split beta of this block into [gamma, delta]
      gamma = beta[self.dual_index[i]:self.dual_index[i+1]]
      delta = beta[self.dual_index[i+1]:self.dual_index[i+2]]

      # Expanding the product with diagonal matrices
      if i == 0:
        h_beta_rows.append(tf.multiply(2*self.lambda_lu[i], gamma) -
                           self.nn_params.forward_pass(
                               tf.multiply(self.lambda_quad[i+1], delta),
                               i, is_transpose=True))
      else:
        h_beta_rows[i] = (h_beta_rows[i] +
                          tf.multiply(self.lambda_quad[i] +
                                      self.lambda_lu[i], gamma) -
                          self.nn_params.forward_pass(
                              tf.multiply(self.lambda_quad[i+1], delta),
                              i, is_transpose=True))

      new_row = (tf.multiply(self.lambda_quad[i+1] + self.lambda_lu[i+1], delta)
                 - tf.multiply(self.lambda_quad[i+1],
                               self.nn_params.forward_pass(gamma, i)))
      h_beta_rows.append(new_row)

    # Last boundary case
    h_beta_rows[self.nn_params.num_hidden_layers] = (
        h_beta_rows[self.nn_params.num_hidden_layers] +
        tf.multiply((self.lambda_quad[self.nn_params.num_hidden_layers] +
                     self.lambda_lu[self.nn_params.num_hidden_layers]),
                    delta))

    h_beta = tf.concat(h_beta_rows, axis=0)

    # Constructing final result using vector_g
    self.set_differentiable_objective()
    result = tf.concat([alpha*self.nu+tf.reduce_sum(
        tf.multiply(beta, self.vector_g))
                        , tf.multiply(alpha, self.vector_g) + h_beta], axis=0)
    return result


  def get_reliable_eigvals_H(self, sess):
    """Function to compute reliable eigen values of H."""
    if(FLAGS.use_scipy_eig or True):
      # Constructing a linear operator for products with H 
      dim = self.matrix_m_dimension
      input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
      output_vector = self.get_psd_product(input_vector)

      def matrix_vector_product(np_vector):
        # Inserting 0 at the beginning to reuse the product with large matrix M
        np_vector = np.insert(np_vector, 0, 0)
        np_vector = np.reshape(np_vector, [-1, 1])
        output_np_vector = sess.run(output_vector, feed_dict={input_vector:np_vector})
        return output_np_vector[1:]

      eigval_linear_operator = LinearOperator((dim-1, dim-1), matvec=matrix_vector_product)
      np_eigen_val, np_eigen_vector = eigs(eigval_linear_operator, k=1, which='SR', tol=1E-4)
      min_eigen_val = np.min(np_eigen_val)
      return tf.convert_to_tensor(min_eigen_val, dtype=tf.float32)
    
    # (TODO): have to implement a version based on 
    else:
      self.get_full_psd_matrix()
      eigen_vals = tf.self_adjoint_eigvals(self.matrix_h)
      min_eigen_val = tf.reduce_min(eig_vals)
      return min_eigen_val

 
  def get_full_psd_matrix(self):
    """Function that retuns the tf graph corresponding to the entire matrix M.


    Returns:
      matrix_h: unrolled version of tf matrix corresponding to H
      matrix_m: unrolled tf matrix corresponding to M
    """
    # Computing the matrix term
    h_columns = []
    for i in range(self.nn_params.num_hidden_layers + 1):
      current_col_elems = []
      for j in range(i):
        current_col_elems.append(tf.zeros([self.nn_params.sizes[j],
                                           self.nn_params.sizes[i]]))

    # For the first layer, there is no relu constraint
      if i == 0:
        current_col_elems.append(utils.diag(self.lambda_lu[i]))
      else:
        current_col_elems.append(utils.diag(self.lambda_lu[i] +
                                            self.lambda_quad[i]))
      if i < self.nn_params.num_hidden_layers:
        current_col_elems.append((
            (tf.matmul(utils.diag(-1*self.lambda_quad[i+1]),
                       self.nn_params.weights[i]))))
      for j in range(i + 2, self.nn_params.num_hidden_layers + 1):
        current_col_elems.append(tf.zeros([self.nn_params.sizes[j],
                                           self.nn_params.sizes[i]]))
      current_column = tf.concat(current_col_elems, 0)
      h_columns.append(current_column)

    self.matrix_h = tf.concat(h_columns, 1)
    self.set_differentiable_objective()
    self.matrix_h = (self.matrix_h + tf.transpose(self.matrix_h))

    self.matrix_m = tf.concat([tf.concat([self.nu, tf.transpose(self.vector_g)],
                                         axis=1),
                               tf.concat([self.vector_g, self.matrix_h],
                                         axis=1)], axis=0)
    return self.matrix_h, self.matrix_m

  def save_dual(self, folder, sess):
    """ Function to save the current values of the dual variable"""
    if not tf.gfile.IsDirectory(folder):
      tf.gfile.MkDir(folder)
    [current_lambda_pos, current_lambda_neg, current_lambda_lu, 
     current_lambda_quad, current_nu] = sess.run([self.lambda_pos, 
                                                  self.lambda_neg, 
                                                  self.lambda_lu, 
                                                  self.lambda_quad, 
                                                  self.nu])
    np.save(os.path.join(folder, 'lambda_pos'), current_lambda_pos)
    np.save(os.path.join(folder, 'lambda_neg'), current_lambda_neg)
    np.save(os.path.join(folder, 'lambda_lu'), current_lambda_lu)
    np.save(os.path.join(folder, 'lambda_quad'), current_lambda_quad)
    np.save(os.path.join(folder, 'nu'), current_nu)
    print('Saved the current dual variables in folder:', folder)

                                                    
