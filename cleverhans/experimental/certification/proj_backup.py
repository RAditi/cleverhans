  def project_dual(self, slack, nn_params_ff, sess):
    print("Projecting with slack:", slack)
    """Function that projects the input dual variables onto the feasible set.
    Args:
      sess: tf session to run things (for scipy computations)
    Returns:
      projected_dual: Feasible dual solution corresponding to current dual
      projected_certificate: Objective value of feasible dual
    """
    # TODO: consider whether we can use shallow copy of the lists without
    # using tf.identity
    projected_lambda_pos = [tf.identity(x) for x in self.lambda_pos]
    projected_lambda_neg = [tf.identity(x) for x in self.lambda_neg]
    projected_lambda_quad = [tf.identity(x) for x in self.lambda_quad]
    projected_lambda_lu = [tf.identity(x) for x in self.lambda_lu]
    projected_nu = tf.identity(self.nu)

    # Minimum eigen value of H
    # TODO: Write this in terms of matrix multiply
    # matrix H is a submatrix of M, thus we just need to extend existing code
    # for computing matrix-vector product (see get_psd_product function).
    # Then use the same trick to compute smallest eigenvalue.
    min_eig = self.get_reliable_eigvals_H(sess)
    print("Original scalar f", sess.run(self.scalar_f))
    print("Pre projection min eigen value", sess.run(min_eig))
    for i in range(self.nn_params.num_hidden_layers+1):
      # Since lambda_lu appears only in diagonal terms, can subtract to
      # make PSD and feasible
      projected_lambda_lu[i] = tf.maximum(tf.nn.relu((projected_lambda_lu[i] +
                                                      0.5*tf.maximum(-min_eig, 0) + 1E-5)), slack)
      # Adjusting lambda_neg wherever possible so that lambda_neg + lambda_lu
      # remains close to unchanged
      # projected_lambda_neg[i] = tf.maximum(0.0, projected_lambda_neg[i] +
      #                                      (0.5*min_eig - slack)*
      #                                      (self.lower[i] + self.upper[i]))

    # projected_dual_var = {'lambda_pos': projected_lambda_pos,
    #                       'lambda_neg': projected_lambda_neg,
    #                       'lambda_lu': projected_lambda_lu,
    #                       'lambda_quad': projected_lambda_quad,
    #                       'nu': projected_nu}

    print("############## Convolution version #############")
    FLAGS.binary_search = True
    projected_dual_object = DualFormulation(sess,
                                            self.nn_params,
                                            self.test_input,
                                            self.true_class,
                                            self.adv_class,
                                            self.input_minval,
                                            self.input_maxval,
                                            self.epsilon)

    # Initializing with the projected solution
    projected_dual_object.lambda_pos = projected_lambda_pos 
    projected_dual_object.lambda_neg = projected_lambda_neg 
    projected_dual_object.lambda_quad = projected_lambda_quad 
    projected_dual_object.lambda_lu = projected_lambda_lu 
    projected_dual_object.nu = projected_nu 
    projected_certificate = projected_dual_object.compute_certificate(sess)

    print("############## Non convolution version #############")
    # Checking the non conv version
    FLAGS.binary_search = False
    projected_dual_object = DualFormulation(sess,
                                            nn_params_ff,
                                            self.test_input,
                                            self.true_class,
                                            self.adv_class,
                                            self.input_minval,
                                            self.input_maxval,
                                            self.epsilon)

    # Initializing with the projected solution
    projected_dual_object.lambda_pos = projected_lambda_pos 
    projected_dual_object.lambda_neg = projected_lambda_neg 
    projected_dual_object.lambda_quad = projected_lambda_quad 
    projected_dual_object.lambda_lu = projected_lambda_lu 
    projected_dual_object.nu = projected_nu 

    projected_certificate = projected_dual_object.compute_certificate(sess)

    return projected_certificate



 def compute_certificate(self, sess):
    """Function to compute the certificate associated with feasible solution."""
    self.set_differentiable_objective()
    min_eig = self.get_reliable_eigvals_H(sess)

    if(FLAGS.perform_cg):
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

      H_linear_operator = LinearOperator((dim-1, dim-1), matvec=matrix_vector_product)
      Np_vector_g = sess.run(self.vector_g)

      # np_Hinv_g, info = cg(H_linear_operator, np_vector_g, tol=1E-2, maxiter=10000000)
      # print("Info of conjugate gradient operator", info)
      # tf_Hinv_g = tf.convert_to_tensor(np.reshape(np_Hinv_g, [-1, 1]), dtype=tf.float32)
      # projected_certificate = (
      #   self.scalar_f +
      #   0.5*tf.matmul(tf.transpose(self.vector_g),
      #                           tf_Hinv_g))

    elif (FLAGS.binary_search):
      # print("Nu before projection", sess.run(self.nu))
      current_min_eig = 1E1
      #Step one: compute current min eig 
      dim = self.matrix_m_dimension
      input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
      output_vector = self.get_psd_product(input_vector)

      def matrix_vector_product(np_vector):
        np_vector = np.reshape(np_vector, [-1, 1])
        output_np_vector = sess.run(output_vector, feed_dict={input_vector:np_vector})
        return output_np_vector

      M_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
      min_eigen_val, min_eigen_vector = eigs(M_linear_operator, k=1, which='SR',tol=1E-3)
      print("First minimum eigen value", min_eigen_val)

      # TODO: Add code for decreasing nu if needed 
      
      while(min_eigen_val < 1E-5):
        self.nu = 2*self.nu
        input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
        output_vector = self.get_psd_product(input_vector)

        def matrix_vector_product(np_vector):
          np_vector = np.reshape(np_vector, [-1, 1])
          output_np_vector = sess.run(output_vector, feed_dict={input_vector:np_vector})
          return output_np_vector

        M_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
        min_eigen_val, min_eigen_vector = eigs(M_linear_operator, k=1, which='SR', tol = 1E-6)

      small = sess.run(0.5*self.nu)
      large = sess.run(self.nu)
      current = 0.5*(small + large)

      current_scalar_f = sess.run(self.scalar_f)
      smallest_value = 0.5*small + current_scalar_f
      largest_value = 0.5*large + current_scalar_f
      
      if(smallest_value > 0.5):
        print("Certificate is not possibly negative")

      else: 
        # Perform binary search to refine
        while(large-small > FLAGS.tol):
          self.nu = tf.convert_to_tensor(current, dtype=tf.float32)
          input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
          output_vector = self.get_psd_product(input_vector)

          def matrix_vector_product(np_vector):
            np_vector = np.reshape(np_vector, [-1, 1])
            output_np_vector = sess.run(output_vector, feed_dict={input_vector:np_vector})
            return output_np_vector

          M_linear_operator = LinearOperator((dim, dim), matvec=matrix_vector_product)
          min_eigen_val, min_eigen_vector = eigs(M_linear_operator, k=1, which='SR')
          if(min_eigen_val > 1E-8):
            large = current 
          else:
            small = current 
          current = 0.5*(small + large)

      projected_certificate = self.scalar_f + 0.5*self.nu
      print("Scalar f", sess.run(self.scalar_f))
      print("Nu", sess.run(0.5*self.nu))
      print("Certificate one", sess.run(projected_certificate))

      # self.get_full_psd_matrix()
      # projected_certificate = (
      #   self.scalar_f +
      #   0.5*tf.matmul(tf.matmul(tf.transpose(self.vector_g),
      #                           tf.matrix_inverse(self.matrix_h)),
      #                 self.vector_g))
      # print("Certificate two", sess.run(projected_certificate))

    # TODO: replace matrix_inverse with functin which uses matrix-vector product
    else:
      self.get_full_psd_matrix()
      print("Obtained the full computation graph")
      projected_certificate = (
        self.scalar_f +
        0.5*tf.matmul(tf.matmul(tf.transpose(self.vector_g),
                                tf.matrix_inverse(self.matrix_h)),
                      self.vector_g))

      scalar_f = sess.run(scalar_f)
      second_term = sess.run(tf.matmul(tf.matmul(tf.transpose(self.vector_g),
                                                        tf.matrix_inverse(self.matrix_h)),
                                              self.vector_g))
      print("Scalar f", scalar_f)
      print("Second term", 0.5*second_term)
      print("Inversion certificate", scalar_f + 0.5*second_term)

    return projected_certificate
