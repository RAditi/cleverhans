"""Code for setting up the optimization problem for certification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
from numpy.linalg import cholesky
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs 
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import inv
import tensorflow as tf
from tensorflow.contrib import autograph

from cleverhans.experimental.certification import dual_formulation
flags = tf.app.flags
FLAGS = flags.FLAGS

# Bound on lowest value of certificate to check for numerical errors
LOWER_CERT_BOUND = -40.0


class Optimization(object):
  """Class that sets up and runs the optimization of dual_formulation"""

  def __init__(self, dual_formulation_object, sess, optimization_params, nn_params_ff):
    """Initialize the class variables.

    Args:
      dual_formulation_object: Instance of DualFormulation that contains the
        dual variables and objective
      sess: tf session to be used to run
      optimization_params: Dictionary with the following
        eig_num_iter - Number of iterations to run for computing minimum eigen
          value
        eig_learning_rate - Learning rate for minimum eigen value iterations
        init_smooth - Starting value of the smoothness parameter (typically
        around 0.001)
        smooth_decay - The factor by which to decay after every outer loop epoch
        optimizer - one of gd, adam, momentum or adagrad
    """
    self.dual_object = dual_formulation_object
    self.params = optimization_params
    self.penalty_placeholder = tf.placeholder(tf.float32, shape=[])

    # The dimensionality of matrix M is the sum of sizes of all layers + 1
    # The + 1 comes due to a row and column of M representing the linear terms
    self.eig_init_vec_placeholder = tf.placeholder(
        tf.float32, shape=[1 + self.dual_object.dual_index[-1], 1])
    self.smooth_placeholder = tf.placeholder(tf.float32, shape=[])
    self.eig_num_iter_placeholder = tf.placeholder(tf.int32, shape=[])
    self.sess = sess
    self.nn_params_ff = nn_params_ff

  def eig_one_step(self, current_vector):
    """Function that performs one step of gd (variant) for min eigen value.

    Args:
      current_vector: current estimate of the eigen vector with minimum eigen
        value

    Returns:
      updated vector after one step
    """
    grad = 2*self.dual_object.get_psd_product(current_vector)
    # Current objective = (1/2)*v^T (2*M*v); v = current_vector
    # grad = 2*M*v
    current_objective = tf.reshape(tf.matmul(tf.transpose(current_vector),
                                             grad) / 2., shape=())

    # Project the gradient into the tangent space of the constraint region.
    # This way we do not waste time taking steps that try to change the
    # norm of current_vector
    grad = grad - current_vector*tf.matmul(tf.transpose(current_vector), grad)
    grad_norm = tf.norm(grad)
    grad_norm_sq = tf.square(grad_norm)

    # Computing normalized gradient of unit norm
    norm_grad = grad / grad_norm

    # Computing directional second derivative (dsd)
    # dsd = 2*g^T M g, where g is normalized gradient
    directional_second_derivative = (
        tf.reshape(2*tf.matmul(tf.transpose(norm_grad),
                               self.dual_object.get_psd_product(norm_grad)),
                   shape=()))

    # Computing grad^\top M grad [useful to compute step size later]
    # Just a rescaling of the directional_second_derivative (which uses
    # normalized gradient
    grad_m_grad = directional_second_derivative*grad_norm_sq / 2

    # Directional_second_derivative/2 = objective when vector is norm_grad
    # If this is smaller than current objective, simply return that
    if directional_second_derivative / 2. < current_objective:
      return norm_grad

    # If curvature is positive, jump to the bottom of the bowl
    if directional_second_derivative > 0.:
      step = -1. * grad_norm / directional_second_derivative
    else:
      # If the gradient is very small, do not move
      if grad_norm_sq <= 1e-16:
        step = 0.0
      else:
        # Make a heuristic guess of the step size
        step = -2. * tf.reduce_sum(current_vector*grad) / grad_norm_sq
        # Computing gain using the gradient and second derivative
        gain = -(2 * tf.reduce_sum(current_vector*grad) +
                 (step*step) * grad_m_grad)

        # Fall back to pre-determined learning rate if no gain
        if gain < 0.:
          step = -self.params['eig_learning_rate'] * grad_norm
    current_vector = current_vector + step * norm_grad
    return tf.nn.l2_normalize(current_vector)

  def multi_steps(self):
    """Function that runs one step iteratively to compute min eig value.

    Returns:
      current_vector: the estimate of eigen vector with minimum eigen value
    """
    current_vector = self.eig_init_vec_placeholder
    counter = 0
    while counter < self.eig_num_iter_placeholder:
      # TODO: figure out how to fix following
      # Without argument self autograph throws error. At the same this
      # function call should not require self and it's a lint error.
      # pylint: disable=too-many-function-args
      current_vector = self.eig_one_step(self, current_vector)
      # pylint: enable=too-many-function-args
      counter += 1
    return current_vector

  def tf_min_eig_vec(self):
    """Function for min eigen vector using tf's full eigen decomposition."""
    # Full eigen decomposition requires the explicit psd matrix M
    _, matrix_m = self.dual_object.get_full_psd_matrix()
    [eig_vals, eig_vectors] = tf.self_adjoint_eig(matrix_m)
    index = tf.argmin(eig_vals)
    return tf.reshape(eig_vectors[:, index],
                      shape=[eig_vectors.shape[0].value, 1])

  def tf_smooth_eig_vec(self):
    """Function that returns smoothed version of min eigen vector."""
    _, matrix_m = self.dual_object.get_full_psd_matrix()
    # Easier to think in terms of max so negating the matrix
    [eig_vals, eig_vectors] = tf.self_adjoint_eig(-matrix_m)
    exp_eig_vals = tf.exp(tf.divide(eig_vals, self.smooth_placeholder))
    scaling_factor = tf.reduce_sum(exp_eig_vals)
    # Multiplying each eig vector by exponential of corresponding eig value
    # Scaling factor normalizes the vector to be unit norm
    eig_vec_smooth = tf.divide(tf.matmul(eig_vectors,
                                         tf.diag(tf.sqrt(exp_eig_vals))),
                               tf.sqrt(scaling_factor))
    return tf.reshape(tf.reduce_sum(eig_vec_smooth, axis=1),
                      shape=[eig_vec_smooth.shape[0].value, 1])

  def get_min_eig_vec_proxy(self, use_tf_eig=False):
    """Computes the min eigen value and corresponding vector of matrix M.

    Args:
      use_tf_eig: Whether to use tf's default full eigen decomposition

    Returns:
      eig_vec: Minimum absolute eigen value
      eig_val: Corresponding eigen vector
    """
    if use_tf_eig:
      # If smoothness parameter is too small, essentially no smoothing
      # Just output the eigen vector corresponding to min
      return self.tf_min_eig_vec()
      return tf.cond(self.smooth_placeholder < 1E-8,
                     self.tf_min_eig_vec,
                     self.tf_smooth_eig_vec)
    # Using autograph to automatically handle the control flow of multi_steps()
    multi_steps_tf = autograph.to_graph(self.multi_steps)
    estimated_eigen_vector = multi_steps_tf(self)
    return estimated_eigen_vector

  def scipy_eig_vec(self):
    """ Function to compute eigen vector with scipy """

    dim = self.dual_object.matrix_m_dimension
    input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
    output_vector = self.dual_object.get_psd_product(input_vector)

    def matrix_vector_product(np_vector):
      np_vector = np.reshape(np_vector, [np.size(np_vector), 1])
      output_np_vector = self.sess.run(output_vector, feed_dict={input_vector:np_vector})
      return output_np_vector
                                         
    dual_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
    np_eigen_val, np_eigen_vector = eigs(dual_linear_operator, k=1, which='SR',tol=1E-4)
    index = np.argmin(np_eigen_val)
    return np.reshape(np_eigen_vector[:, index], [-1, 1]), np_eigen_val[index]

    
  def binary_search(self, lambda_feed_dict):
    """ Function to compute the binary search solution  
    from the dual variables with values in lambda_feed_dict 
    """
    # Computing the value of nu by binary search 
    current_min_eig = 1E1
    #Step one: compute current min eig 
    dim = self.pdualconv_object.matrix_m_dimension
    input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
    output_vector = self.pdualconv_object.get_psd_product(input_vector)

    def matrix_vector_product(np_vector):
      np_vector = np.reshape(np_vector, [-1, 1])
      lambda_feed_dict.update({input_vector:np_vector})
      output_np_vector = self.sess.run(output_vector, feed_dict=lambda_feed_dict)
      return output_np_vector

    M_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
    min_eigen_val, min_eigen_vector = eigs(M_linear_operator, k=1, which='SR',tol=1E-3)
    print("First minimum eigen value", min_eigen_val)

    # TODO: Add code for decreasing nu if needed 
    while(min_eigen_val < 1E-5):
      lambda_feed_dict[self.pdualconv_object.nu] = 2*lambda_feed_dict[self.pdualconv_object.nu]
      input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
      output_vector = self.pdualconv_object.get_psd_product(input_vector)

      def matrix_vector_product(np_vector):
        np_vector = np.reshape(np_vector, [-1, 1])
        lambda_feed_dict.update({input_vector:np_vector})
        output_np_vector = self.sess.run(output_vector, lambda_feed_dict)
        return output_np_vector

      M_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
      min_eigen_val, min_eigen_vector = eigs(M_linear_operator, k=1, which='SR', tol = 1E-3)

    small = 0.5*lambda_feed_dict[self.pdualconv_object.nu]
    large = lambda_feed_dict[self.pdualconv_object.nu]
    current = 0.5*(small + large)

    current_scalar_f = self.sess.run(self.pdualconv_object.scalar_f, feed_dict=lambda_feed_dict)
    smallest_value = 0.5*small + current_scalar_f
    largest_value = 0.5*large + current_scalar_f
      
    if(smallest_value > 0.01):
      print("Certificate is not possibly negative")

    else: 
      # Perform binary search to refine
      while(large-small > FLAGS.tol and small < 0.01 - 2*current_scalar_f):
        print(large-small)
        lambda_feed_dict[self.pdualconv_object.nu] = current
        input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
        output_vector = self.pdualconv_object.get_psd_product(input_vector)
        
        def matrix_vector_product(np_vector):
          np_vector = np.reshape(np_vector, [-1, 1])
          lambda_feed_dict.update({input_vector:np_vector})
          output_np_vector = self.sess.run(output_vector, lambda_feed_dict)
          return output_np_vector

        M_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
        min_eigen_val, min_eigen_vector = eigs(M_linear_operator, k=1, which='SR')
        if(min_eigen_val > 1E-8):
          large = current 
          print("Changing large")
        else:
          small = current 
          print("Changing small")
        current = 0.5*(small + large)
        
    projected_certificate = current_scalar_f + 0.5*large 
    print("Current scalar f", current_scalar_f)
    print("Conv dual certificate", projected_certificate)
    return projected_certificate


  def project_dual_conv(self, slack):
    """ new version of the function so that the computation graph is built only once"""

    # Obtain numpy versions of the projected dual variables
    print("projecting for slack:", slack)
    min_eig = self.sess.run(self.dual_object.get_reliable_eigvals_H(self.sess))

    print("Original scalar f", self.sess.run(self.dual_object.scalar_f))
    print("Pre projection min eigen value", min_eig)
    projected_lambda_pos = self.sess.run([x for x in self.dual_object.lambda_pos])
    projected_lambda_neg = self.sess.run([x for x in self.dual_object.lambda_neg])
    projected_lambda_quad = self.sess.run([x for x in self.dual_object.lambda_quad])
    projected_lambda_lu = self.sess.run([x for x in self.dual_object.lambda_lu])
    projected_nu = self.sess.run(self.dual_object.nu)

    for i in range(self.dual_object.nn_params.num_hidden_layers+1):
      # Since lambda_lu appears only in diagonal terms, can subtract to
      # make PSD and feasible
      projected_lambda_lu[i] = np.maximum(projected_lambda_lu[i] + 0.5*np.maximum(-min_eig, 0) + 1E-5, 0)
      projected_lambda_lu[i] = np.maximum(projected_lambda_lu[i], slack)
      # Adjusting lambda_neg wherever possible so that lambda_neg + lambda_lu
      # remains close to unchanged
      # projected_lambda_neg[i] = tf.maximum(0.0, projected_lambda_neg[i] +
      #                                      (0.5*min_eig - slack)*
      #                                      (self.lower[i] + self.upper[i]))

    # Creating feed dict for projected dual variables
    lambda_feed_dict = {}
    # for i in range(self.dual_object.nn_params.num_hidden_layers + 1):
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualconv_object.lambda_pos, 
                                                                                           projected_lambda_pos)})
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualconv_object.lambda_neg, 
                                                                                           projected_lambda_neg)})
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualconv_object.lambda_quad, 
                                                                                           projected_lambda_quad)})
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualconv_object.lambda_lu, 
                                                                                      projected_lambda_lu)})

    lambda_feed_dict.update({self.pdualconv_object.nu: projected_nu})

    return self.binary_search(lambda_feed_dict)



  def project_dual_ff(self, slack):
    """ new version of the function so that the computation graph is built only once"""

    # Obtain numpy versions of the projected dual variables
    print("Original scalar f", self.sess.run(self.dual_object.scalar_f))
    projected_lambda_pos = self.sess.run([x for x in self.dual_object.lambda_pos])
    projected_lambda_neg = self.sess.run([x for x in self.dual_object.lambda_neg])
    projected_lambda_quad = self.sess.run([x for x in self.dual_object.lambda_quad])
    projected_lambda_lu = self.sess.run([x for x in self.dual_object.lambda_lu])
    new_projected_lambda_lu = [np.copy(x) for x in projected_lambda_lu]
    new_projected_lambda_neg = [np.copy(x) for x in projected_lambda_neg]
    lower = self.sess.run([x for x in self.dual_object.lower])
    upper = self.sess.run([x for x in self.dual_object.upper])

    positive_indices = self.sess.run([x for x in self.dual_object.positive_indices])
    negative_indices = self.sess.run([x for x in self.dual_object.negative_indices])
    switch_indices = self.sess.run([x for x in self.dual_object.switch_indices])

    projected_nu = self.sess.run(self.dual_object.nu)

    # Creating feed dict for projected dual variables
    lambda_feed_dict = {}
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualff_object.lambda_pos, 
                                                                                           projected_lambda_pos)})
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualff_object.lambda_neg, 
                                                                                           projected_lambda_neg)})
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualff_object.lambda_quad, 
                                                                                           projected_lambda_quad)})
    lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualff_object.lambda_lu, 
                                                                                           projected_lambda_lu)})
    lambda_feed_dict.update({self.pdualff_object.nu: projected_nu})
    old_vector_g = self.sess.run(self.pdualff_object.vector_g, lambda_feed_dict)
    old_matrix_h = self.sess.run(self.pdualff_object.matrix_h, lambda_feed_dict)
    old_scalar_f = self.sess.run(self.pdualff_object.scalar_f, lambda_feed_dict)
    old_matrix_m = self.sess.run(self.pdualff_object.matrix_m, lambda_feed_dict)
    np.save('old_matrix_m_5', old_matrix_m)
    
    # min_eig_val_h, min_eig_vec_h = eigs(old_matrix_h, k=1, which='SR',tol=1E-6)
    small = 0 
    large = 0.1
    while( np.abs(large - small) >  0.00001):
      current = 0.5*(large + small)
      # print("Current", current)
      try: 
        cholesky(old_matrix_h + current*np.eye(5588))
      except:
        small = current
        # print("Inc small")
      else:
        large = current
        # print("Dec large")

    print("Min eig h:", large)
    min_eig_val_h = -large 
    slack = 1e-6
    for min_eig_val_M in [0, -0.0001, -0.0005, -0.001]:
    # for min_eig_val_M in [0]:
      print("Adding value:", min_eig_val_M)
      for i in range(self.dual_object.nn_params.num_hidden_layers+1):
        new_projected_lambda_lu[i] = projected_lambda_lu[i] + 0.5*np.maximum(-min_eig_val_M -min_eig_val_h, 0) + slack
        new_projected_lambda_neg[i] = projected_lambda_neg[i] + np.multiply((lower[i] + upper[i]), 
                                                                           (projected_lambda_lu[i] - 
                                                                            new_projected_lambda_lu[i]))
        new_projected_lambda_neg[i] = (np.multiply(negative_indices[i], 
                                                   new_projected_lambda_neg[i])+ 
                                      np.multiply(switch_indices[i], np.maximum(new_projected_lambda_neg[i], 0)))

      lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualff_object.lambda_lu, new_projected_lambda_lu)})    
      lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualff_object.lambda_neg, new_projected_lambda_neg)})    
      scalar_f = self.sess.run(self.pdualff_object.scalar_f, lambda_feed_dict)
      print("New scalar f:", scalar_f)
      new_vector_g = self.sess.run(self.pdualff_object.vector_g, lambda_feed_dict)
      new_matrix_h = self.sess.run(self.pdualff_object.matrix_h, lambda_feed_dict)
      # new_matrix_m = self.sess.run(self.pdualff_object.matrix_m, lambda_feed_dict)
      # np.save("New matrix M", new_matrix_m)
      # print("New scalar f", scalar_f)
      # second_term = np.matmul(np.matmul(np.transpose(old_vector_g), inv(csc_matrix(new_matrix_h)).toarray()), old_vector_g)
      # print("Check second term ", second_term)
      second_term = np.matmul(np.matmul(np.transpose(new_vector_g), inv(csc_matrix(new_matrix_h)).toarray()), new_vector_g) + 0.05
      print("Second term", second_term)
      lambda_feed_dict.update({self.pdualff_object.nu: second_term})
      new_matrix_m = self.sess.run(self.pdualff_object.matrix_m, lambda_feed_dict)
      np.save('new_matrix_m', new_matrix_m)
      try: 
        cholesky(new_matrix_m)
      except:
        print("Matrix is not PSD -- error")
      else:
        print("Matrix is PSD -- everything good")

      # np.save("new_matrix_m", new_matrix_m)
      # min_eig_val_M, min_eig_vec_M = eigs(new_matrix_m, k=1, which='SR',tol=1E-6)
      # print("new min eigen value", min_eig_val_M)
      # np.save("new_matrix_m", new_matrix_m)
      # print("Second term", second_term)
      projected_certificate = scalar_f + 0.5*second_term
      print("FF dual certificate", projected_certificate)

    return projected_certificate

  def prepare_one_step(self):
    """Create tensorflow op for running one step of descent."""
    # Initialize all variables 
    # Create the objective
    dim = self.dual_object.matrix_m_dimension 
    self.dual_object.set_differentiable_objective()
    if(FLAGS.use_scipy_eig):
      self.eig_vec_estimate = tf.placeholder(tf.float32, shape=(dim, 1))
      # self.matmul_eig_vec_estimate = self.get_min_eig_vec_proxy()
    else:
      self.eig_vec_estimate = self.get_min_eig_vec_proxy()

    self.stopped_eig_vec_estimate = tf.stop_gradient(self.eig_vec_estimate)
    # Eig value is v^\top M v, where v is eigen vector
    self.eig_val_estimate = tf.matmul(tf.transpose(
        self.stopped_eig_vec_estimate),
                                      self.dual_object.get_psd_product(
                                          self.stopped_eig_vec_estimate))
    # Penalizing negative of min eigen value because we want min eig value
    #self.matmul_eig_val_estimate = tf.matmul(tf.transpose(
    #    self.matmul_eig_vec_estimate),
    #                                  self.dual_object.get_psd_product(
    #                                      self.matmul_eig_vec_estimate))

    # to be positive
    self.total_objective = (self.dual_object.unconstrained_objective +
                            0.5*(tf.square(
                                tf.maximum(-1*self.penalty_placeholder*
                                           self.eig_val_estimate, 0))))

    # self.debug_value1 = tf.reduce_min(self.dual_object.lambda_neg[1])    
    # self.debug_value2 = tf.reduce_min(self.dual_object.lambda_neg[2])
    # self.debug_value3 = tf.reduce_min(self.dual_object.lambda_neg[3])
    global_step = tf.Variable(0, trainable=False)

    # Set up learning rate
    # # Learning rate decays after every outer loop
    # learning_rate = tf.train.exponential_decay(
    #     self.params['init_learning_rate'],
    #     global_step, self.params['inner_num_steps'],
    #     self.params['learning_rate_decay'], staircase=True)
    # # Constant learning rate 
    # learning_rate = self.params['init_learning_rate']

    self.learning_rate = tf.placeholder(tf.float32, shape=[])

    # Set up the optimizer
    if self.params['optimizer'] == 'adam':
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate)
    elif self.params['optimizer'] == 'adagrad':
      self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
    elif self.params['optimizer'] == 'momentum':
      self.optimizer = tf.train.MomentumOptimizer(
          learning_rate=self.learning_rate,
          momentum=self.params['momentum_parameter'], use_nesterov=True)
    else:
      self.optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=self.learning_rate)

    # Write out the projection step
    self.train_step = self.optimizer.minimize(self.total_objective,
                                              global_step=global_step)
    # Initializing all variables 
    self.sess.run(tf.global_variables_initializer())
    # All dual variables are positive
    proj_ops = []
    for i in range(self.dual_object.nn_params.num_hidden_layers + 1):
      
      proj_ops.append(self.dual_object.lambda_pos[i].assign(
        tf.multiply(self.dual_object.positive_indices[i], 
                    self.dual_object.lambda_pos[i])+ 
        tf.multiply(self.dual_object.switch_indices[i], 
                    tf.nn.relu(self.dual_object.lambda_pos[i]))))

      proj_ops.append(self.dual_object.lambda_neg[i].assign(
        tf.multiply(self.dual_object.negative_indices[i], 
                    self.dual_object.lambda_neg[i])+ 
        tf.multiply(self.dual_object.switch_indices[i], 
                    tf.nn.relu(self.dual_object.lambda_neg[i]))))

      proj_ops.append(self.dual_object.lambda_quad[i].assign(
        tf.multiply(self.dual_object.switch_indices[i], 
                    tf.nn.relu(self.dual_object.lambda_quad[i]))))
      
        # if(i > 0):
      proj_ops.append(self.dual_object.lambda_lu[i].assign(
        tf.nn.relu(self.dual_object.lambda_lu[i])))
      
        # proj_ops.append(self.dual_object.lambda_lu[i].assign(
        #   tf.multiply(self.dual_object.switch_indices[i], 
        #               tf.nn.relu(self.dual_object.lambda_lu[i]))))
        # else:

        # proj_ops.append(self.dual_object.lambda_pos[i].assign(
        #   tf.zeros([self.dual_object.nn_params.sizes[i], 1])))
        # proj_ops.append(self.dual_object.lambda_neg[i].assign(
        #   tf.zeros([self.dual_object.nn_params.sizes[i], 1])))
        # proj_ops.append(self.dual_object.lambda_quad[i].assign(
        #   tf.zeros([self.dual_object.nn_params.sizes[i], 1])))
        # proj_ops.append(self.dual_object.lambda_lu[i].assign(
        #   tf.zeros([self.dual_object.nn_params.sizes[i], 1])))

    # with tf.control_dependencies([self.train_step]):
    #   # print(proj_ops)
    #   self.proj_step = tf.group(proj_ops)
      # self.proj_step = tf.group([v.assign(tf.maximum(v, 0))
      #                            for v in tf.trainable_variables()])

    # Control dependencies ensures that train_step is executed first
    self.proj_step = tf.group(proj_ops)
    self.opt_one_step = self.proj_step
    # Run the initialization of all variables
    # TODO: do we need to do it here or can do outside of this class?
    # Create folder for saving stats if the folder is not None
    if(self.params['stats_folder'] is not None and
       not tf.gfile.IsDirectory(self.params['stats_folder'])):
      print("Made directory")
      tf.gfile.MkDir(self.params['stats_folder'])
    # Creating the projection conv object 
    self.pdualconv_object = dual_formulation.DualFormulation(self.sess,
                                            self.dual_object.nn_params,
                                            self.dual_object.test_input,
                                            self.dual_object.true_class,
                                            self.dual_object.adv_class,
                                            self.dual_object.input_minval,
                                            self.dual_object.input_maxval,
                                            self.dual_object.epsilon)

    self.pdualconv_object.initialize_placeholder()
    self.pdualconv_object.set_differentiable_objective()

    self.pdualff_object = dual_formulation.DualFormulation(self.sess,
                                            self.nn_params_ff,
                                            self.dual_object.test_input,
                                            self.dual_object.true_class,
                                            self.dual_object.adv_class,
                                            self.dual_object.input_minval,
                                            self.dual_object.input_maxval,
                                            self.dual_object.epsilon)

    self.pdualff_object.initialize_placeholder()
    self.pdualff_object.get_full_psd_matrix()
    # self.pdualff_object.second_term = 0.5*tf.matmul(tf.matmul(tf.transpose(self.pdualff_object.vector_g),
    #                              tf.matrix_inverse(self.pdualff_object.matrix_h)),
    #                                        self.pdualff_object.vector_g)
    
  def run_one_step(self, eig_init_vec_val, eig_num_iter_val,
                   smooth_val, penalty_val, learning_rate_val):
    """Run one step of gradient descent for optimization.

    Args:
      eig_init_vec_val: Start value for eigen value computations
      eig_num_iter_val: Number of iterations to run for eigen computations
      smooth_val: Value of smoothness parameter
      penalty_val: Value of penalty for the current step
      learning_rate_val: Value of the learning rate

    Returns:
     found_cert: True is negative certificate is found, False otherwise
    """
    # Project onto feasible set of dual variables
    if self.current_step % self.params['projection_steps'] == 0 and False :
      # self.sess.run(self.proj_step)
      list_slacks = [0]
      # valid_certificates = [self.project_dual_ff(slack) for slack in list_slacks]
      valid_certificates = [self.project_dual_ff(slack) for slack in list_slacks]
      projected_certificate = min(valid_certificates)

      if(FLAGS.use_scipy_eig):
        current_np_vector, current_scipy_eig_val = self.scipy_eig_vec()
        current_certificate = self.sess.run(projected_certificate, feed_dict={self.smooth_placeholder:smooth_val, 
                                                                              self.eig_vec_estimate:current_np_vector})
      else:
        # current_certificate = self.sess.run(projected_certificate, feed_dict={self.smooth_placeholder: smooth_val})
        current_certificate = projected_certificate
      print('Current certificate', current_certificate)

      # Sometimes due to either overflow or instability in inverses,
      # the returned certificate is large and negative -- keeping a check
      if LOWER_CERT_BOUND < current_certificate < 0:
        print('Found certificate of robustness')
        return True

    # Running step
    step_feed_dict = {self.eig_init_vec_placeholder: eig_init_vec_val,
                      self.eig_num_iter_placeholder: eig_num_iter_val,
                      self.smooth_placeholder: smooth_val,
                      self.penalty_placeholder: penalty_val,
                      self.learning_rate: learning_rate_val}
    
    # Scipy eig requires a different step feed dict because the eig vector 
    # comes from outside as a placeholder
    if FLAGS.use_scipy_eig: 
      current_np_vector, current_scipy_eig_val = self.scipy_eig_vec()
      step_feed_dict = {self.eig_init_vec_placeholder: eig_init_vec_val,
                        self.eig_num_iter_placeholder: eig_num_iter_val,
                        self.smooth_placeholder: smooth_val,
                        self.penalty_placeholder: penalty_val, 
                        self.eig_vec_estimate: current_np_vector,
                        self.learning_rate: learning_rate_val}

    self.sess.run(self.train_step, feed_dict=step_feed_dict)
    self.sess.run(self.proj_step)
    self.current_eig_vec_val = self.sess.run(self.eig_vec_estimate, feed_dict=step_feed_dict)

    if self.current_step % self.params['print_stats_steps'] == 0:
      [self.current_total_objective, self.current_unconstrained_objective,
       self.current_eig_vec_val,
       self.current_eig_val_estimate, 
       self.current_nu] = self.sess.run(
         [self.total_objective,
          self.dual_object.unconstrained_objective,
          self.eig_vec_estimate,
          self.eig_val_estimate, 
          self.dual_object.nu], feed_dict=step_feed_dict)

      if(self.current_step % 1000 == 0):
        _, self.current_scipy_eig_val = self.scipy_eig_vec()
      else:
        self.current_scipy_eig_val = 0 

      stats = {'total_objective': float(self.current_total_objective),
               'unconstrained_objective': float(self.current_unconstrained_objective),
               'min_eig_val_estimate': float(self.current_eig_val_estimate), 
               'current_nu': float(self.current_nu), 
               'scipy_min_eig_val': float(np.real(self.current_scipy_eig_val))}

      print('Current_inner_step', self.current_step)
      print(stats)
      if self.params['stats_folder'] is not None:
        stats = json.dumps(stats)
        with tf.gfile.Open((self.params['stats_folder']+ '/' +
                            str(self.current_outer_step) + '_' 
                            + str(self.current_step) + '.json'), mode='w')as file_f:
          file_f.write(stats)
          self.dual_object.save_dual(self.params['stats_folder'] + '/' +  
                                     str(self.current_outer_step), self.sess)
    return False

  def run_optimization(self):
    """Run the optimization, call run_one_step with suitable placeholders.

    Returns:
      True if certificate is found
      False otherwise
    """
    self.prepare_one_step()
    penalty_val = self.params['init_penalty']
    # Don't use smoothing initially - very inaccurate for large dimension
    self.smooth_on = False
    smooth_val = 0
    learning_rate_val = self.params['init_learning_rate']

    self.current_outer_step = 1
    while self.current_outer_step <= self.params['outer_num_steps']:
      print('Running outer step', self.current_outer_step)
      print('Penalty val', penalty_val)
      print('Tolerance val', FLAGS.tol)
      print('Learning rate val', learning_rate_val)
      # Running inner loop of optimization with current_smooth_val,
      # current_penalty as smoothness parameters and penalty respectively
      self.current_step = 0
      # Run first step with random eig initialization and large number of steps
      init_vector = np.random.random(
        size=(1 + self.dual_object.dual_index[-1], 1))
      found_cert = self.run_one_step(
          init_vector,
          self.params['large_eig_num_steps'],
          smooth_val,
          penalty_val, 
      learning_rate_val)
      if found_cert:
        return True
      # while self.current_total_objective > -0.2:
      while self.current_step <= self.params['inner_num_steps']:
        self.current_step = self.current_step + 1
        init_vector = np.random.random(
              size=(1 + self.dual_object.dual_index[-1], 1))
        init_vector = self.current_eig_vec_val
        found_cert = self.run_one_step(init_vector,
                                       self.params['small_eig_num_steps'],
                                       smooth_val,
                                       penalty_val, 
                                       learning_rate_val)
        if found_cert:
          return -1
      # Update penalty only if it looks like current objective is optimizes
      print("Current total objective", self.current_total_objective)
      if(self.current_step % 1000 == 0):
        FLAGS.small_eig_num_steps = 1.5*FLAGS.small_eig_num_steps
      if self.current_total_objective <-0.1:
        # if(penalty_val < 1000):
        #   penalty_val = penalty_val*self.params['beta']
        # else: 
        #   penalty_val = penalty_val + 500
        penalty_val = penalty_val*self.params['beta']
        learning_rate_val = learning_rate_val*self.params['learning_rate_decay']
      else:
        learning_rate_val = learning_rate_val*0.5
        pass 
        # To get more accurate gradient estimate
        # self.params['small_eig_num_steps'] = (1.2*self.params['small_eig_num_steps'])
        # self.params['inner_num_steps'] = self.params['inner_num_steps'] + 100

      FLAGS.tol = np.maximum(0.5*FLAGS.tol, 0.1)
      # If eigen values seem small enough, turn on smoothing
      # useful only when performing full eigen decomposition
      if np.abs(self.current_eig_val_estimate) < 0.01:
        smooth_val = self.params['smoothness_parameter']
      self.current_outer_step = self.current_outer_step + 1
    return False
