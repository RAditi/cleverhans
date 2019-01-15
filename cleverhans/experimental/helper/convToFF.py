import tensorflow as tf
import numpy as np 


# define input size
input_shape = [8, 8, 1]   # 8x8x1 image
input_num_elements = input_shape[0] * input_shape[1] * input_shape[2]

# some convolutional kernel
kernel = tf.constant([[0.25, 0.5, 0.75], [1.0, 0.75, 0.5], [-10, 0.0, 0.25]], dtype=tf.float32, shape=(3, 3, 1, 1))

# TF graph to compute convolution
flattened_input = tf.placeholder(tf.float32, shape=[1, input_num_elements])  # batch size = 1
input_image = tf.reshape(flattened_input, [1] + input_shape)
output_image = tf.nn.conv2d(input_image, kernel, strides=[1, 2, 2, 1], padding='SAME')
flattaned_output = tf.reshape(output_image, [1, -1])
output_num_elements = int(flattaned_output.shape[1])

# construct the convolutional matrix
conv_matrix = np.zeros([output_num_elements, input_num_elements], dtype=np.float32)
with tf.Session() as sess:
  for i in range(input_num_elements):
    input_vector = np.zeros((1, input_num_elements), dtype=np.float32)
    input_vector[0, i] = 1.0
    output_vector = sess.run(flattaned_output, feed_dict={flattened_input: input_vector})
    conv_matrix[:,i] = output_vector.flatten()
    
# verify that result is the same
random_input = np.random.random((1, input_num_elements))
with tf.Session() as sess:
  conv_output = sess.run(flattaned_output, feed_dict={flattened_input: random_input})
matmul_output = np.reshape(np.dot(conv_matrix, random_input.flatten()), (1, -1))


print('Absolute difference between outputs: ', np.amax(np.abs(conv_output - matmul_output)))
print('Relative difference between outputs: ', np.amax(np.abs(conv_output - matmul_output) / (np.abs(conv_output) + 1e-7)))
