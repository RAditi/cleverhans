    # TODO: get rid of the special case for one hidden layer
    # Different projection for 1 hidden layer
    if self.nn_params.num_hidden_layers == 1 and False:
      # Creating equivalent PSD matrix for H by Schur complements
      diag_entries = 0.5*tf.divide(
          tf.square(self.lambda_quad[self.nn_params.num_hidden_layers]),
          (self.lambda_quad[self.nn_params.num_hidden_layers] +
           self.lambda_lu[self.nn_params.num_hidden_layers]))
      # If lambda_quad[i], lambda_lu[i] are 0, entry is NaN currently,
      # but we want to set that to 0
      diag_entries = tf.where(tf.is_nan(diag_entries),
                              tf.zeros_like(diag_entries), diag_entries)
      matrix = (
          tf.matmul(tf.matmul(tf.transpose(
              self.nn_params.weights[self.nn_params.num_hidden_layers-1]),
                              utils.diag(diag_entries)),
                    self.nn_params.weights[self.nn_params.num_hidden_layers-1]))
      new_matrix = utils.diag(
          2*self.lambda_lu[self.nn_params.num_hidden_layers - 1]) - matrix
      # Making symmetric
      new_matrix = 0.5*(new_matrix + tf.transpose(new_matrix))
      eig_vals = tf.self_adjoint_eigvals(new_matrix)
      min_eig = tf.reduce_min(eig_vals)
      # If min_eig is positive, already feasible, so don't add
      # Otherwise add to make PSD [1E-6 is for ensuring strictly PSD (useful
      # while inverting)
      projected_lambda_lu[0] = (projected_lambda_lu[0] +
                                0.5*tf.maximum(-min_eig, 0) + 1E-6)


    ### Binary search stuff 
    # vector_g = self.sess.run(self.pdualconv_object.vector_g, lambda_feed_dict)
    # print(np.linalg.norm(vector_g))
    # print(np.shape(vector_g))
    # print(vector_g)


    # # return self.binary_search(lambda_feed_dict)
    
    # # New version where we change the last entry of g 
    # L = self.dual_object.nn_params.num_hidden_layers
    # const_final_g = np.multiply(projected_lambda_quad[L], self.sess.run(self.dual_object.nn_params.biases[L-1])) + self.sess.run(self.dual_object.final_linear)
    # print(np.shape(const_final_g))

    # positive_indices = self.sess.run(self.dual_object.positive_indices[L]).astype(int)
    # positive_indices = np.ravel(np.where(np.ravel(positive_indices)> 0))

    # negative_indices = self.sess.run(self.dual_object.negative_indices[L]).astype(int)
    # negative_indices = np.ravel(np.where(np.ravel(negative_indices)> 0))

    # switch_indices = self.sess.run(self.dual_object.switch_indices[L]).astype(int)
    # switch_indices = np.ravel(np.where(np.ravel(switch_indices)> 0))

    # print("Switch", switch_indices)
    # print("Positive", positive_indices)
    # print("Negative", negative_indices)
    # print(projected_lambda_neg[L][positive_indices])
    # projected_lambda_pos[L][positive_indices] = -1* const_final_g[positive_indices]
    # lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in zip(self.pdualconv_object.lambda_pos, 
    #                                                                                        projected_lambda_pos)})
    # vector_g = self.sess.run(self.pdualconv_object.vector_g, lambda_feed_dict)
    # print(np.linalg.norm(vector_g))
    # print(np.shape(vector_g))
    # print(vector_g)
