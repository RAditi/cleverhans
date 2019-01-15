      self.lower = []
      self.upper = []
      self.switch_indices = []
      # Initializing at the input layer with \ell_\infty constraints
      current_lower = tf.maximum(self.test_input - self.epsilon, self.input_minval)
      current_upper = tf.minimum(self.test_input + self.epsilon, self.input_maxval)
      self.lower.append(current_lower)
      self.upper.append(current_upper)
      self.switch_indices.append(tf.to_float(tf.multiply(current_lower, current_upper) > 0))

      first_matrix = tf.eye(self.nn_params.sizes[0])
      first_matrix = tf.transpose(self.nn_params.matrix_forward_pass(first_matrix, 0))
      
      current_lower = (tf.transpose(self.nn_params.forward_pass(self.test_input, 0))+ 
      tf.transpose(self.nn_params.biases[0]) - self.epsilon*utils.l1_column(first_matrix))
      
      current_upper = (tf.transpose(self.nn_params.forward_pass(self.test_input, 0))+ 
      tf.transpose(self.nn_params.biases[0]) + self.epsilon*utils.l1_column(first_matrix))
      
      self.lower.append(current_lower)
      self.upper.append(current_upper)
      self.switch_indices.append(tf.to_float( tf.multiply(current_lower, current_upper) > 0))

      print("Current bounds shape", current_lower.get_shape())
      nu = []
      gamma = []
      nu.append(np.ones(2))
      gamma.append(tf.reshape(self.nn_params.biases[0], [1, -1]))

      for i in range(1, self.nn_params.num_hidden_layers-1):
        print("Bound computation", i)
        # np_lower = sess.run(current_lower)
        # np_upper = sess.run(current_upper)
        # print("Numpy computations done")
        # positive_indices = (np_lower > 0).astype(int)
        # negative_indices = (np_upper < 0).astype(int)
        # switch_indices = (np.multiply(np_lower,np_upper)<=0).astype(int)
        # all_diag_elems = tf.convert_to_tensor(positive_indices + np.multiply(switch_indices, np.divide(np_upper, np_upper-np_lower)), dtype=tf.float32)
        # switch_diag_elems = tf.convert_to_tensor(np.multiply(switch_indices, np.divide(np_upper, np_upper-np_lower)), dtype=tf.float32)

        switch_indices = tf.to_float( tf.multiply(current_lower, current_upper) > 0)
        print("Switch indices shape", switch_indices.get_shape())
        positive_indices = tf.to_float( current_lower > 0)
        all_diag_elems = positive_indices + tf.multiply(switch_indices, tf.divide(current_upper, 
                                                                                  current_upper - current_lower))
        switch_diag_elems = tf.multiply(switch_indices, tf.divide(current_upper, 
                                                                                  current_upper - current_lower))
        all_diag_elems = tf.reshape(all_diag_elems, [1, -1])
        switch_diag_elems = tf.reshape(switch_diag_elems, [1, -1])

        print("Initializing")
        nu.append(tf.transpose(self.nn_params.matrix_forward_pass(utils.diag(switch_diag_elems), i)))
        print(nu[i].get_shape())
        gamma.append(tf.reshape(self.nn_params.biases[i], [1, -1]))
        gamma_sum = gamma[i]

        print("Propagating")
        for j in range(1, i):
          matrix = tf.matmul(utils.diag(all_diag_elems), tf.transpose(nu[j]))
          nu[j] = tf.transpose(self.nn_params.matrix_forward_pass(matrix, i))
          print(nu[j].get_shape())

        for j in range(0, i):
          gamma[j] = tf.reshape(self.nn_params.forward_pass(tf.multiply(gamma[j], all_diag_elems), i), [1, -1])
          gamma_sum = gamma_sum + gamma[j]


        first_matrix = tf.transpose(self.nn_params.matrix_forward_pass(tf.matmul(utils.diag(all_diag_elems), 
                                                                           tf.transpose(first_matrix)), i))
        print("Computing bounds")
        psi = tf.matmul(tf.reshape(self.test_input, [1, -1]), first_matrix) + gamma_sum
        print(psi.get_shape())

        lower_sum_term = tf.zeros(psi.get_shape())
        for j in range(1, i):
          matrix = tf.matmul(utils.diag(tf.multiply(self.lower[j], self.switch_indices[j])), tf.nn.relu(-nu[j]))
          # matrix = tf.matmul(matrix, utils.diag(self.switch_indices))
          lower_sum_term = lower_sum_term + tf.reduce_sum(matrix, axis=0)

        upper_sum_term = tf.zeros(psi.get_shape())
        for j in range(1, i):
          matrix = tf.matmul(utils.diag(tf.multiply(self.lower[j], self.switch_indices[j])), tf.nn.relu(nu[j]))
          # matrix = tf.matmul(matrix, utils.diag(switch_indices))
          upper_sum_term = upper_sum_term + tf.reduce_sum(matrix, axis=0)

        current_lower = psi - self.epsilon*utils.l1_column(first_matrix) + lower_sum_term
        current_upper = psi + self.epsilon*utils.l1_column(first_matrix) - upper_sum_term 
        self.lower.append(current_lower)
        self.upper.append(current_upper)
        self.switch_indices.append(tf.to_float( tf.multiply(current_lower, current_upper) > 0))

      for i in range(1, self.nn_params.num_hidden_layers+1):
        self.lower[i] = tf.reshape(tf.nn.relu(self.lower[i]), [-1, 1])
        self.upper[i] = tf.reshape(tf.nn.relu(self.upper[i]), [-1, 1])
