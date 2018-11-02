""" File to create separate directory with MNIST data points and labels """
import os 

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.dataset import MNIST

FLAGS = flags.FLAGS

TEST_START=0
TEST_END=10000

def create_data_folder(test_start, test_end, data_folder_path):
    
    mnist = MNIST(train_start=0, train_end=60000,
                  test_start=test_start, test_end=test_end)
    
    x_test, y_test = mnist.get_set('test')
    
    if not os.path.isdir(data_folder_path):
        os.mkdir(data_folder_path)
    os.chdir(data_folder_path)

    for i in range(0, test_end-test_start):
        x_i = x_test[i, :]
        x_i = x_i.reshape(np.size(x_i), 1)
        np.save('test-' + str(i) + '.npy', x_i)
    print("Created folder with separate %d files" %(test_end-test_start))
    np.savetxt('all_labels', y_test)

def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  create_data_folder(FLAGS.test_start, FLAGS.test_end, 
                     FLAGS.data_folder_path)

if __name__ == '__main__':
  flags.DEFINE_integer('test_start', TEST_START,
                       'first index of test points')
  flags.DEFINE_integer('test_end', TEST_END,
                       'last index of test points')
  flags.DEFINE_string('data_folder_path', None, 
                      'Path of directory to save the test points')
  
  tf.app.run()
  
