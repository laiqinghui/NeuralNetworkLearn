#
#   Chapter 5, example 4a
#

import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

learning_rate = 0.001

no_features = 13
no_labels = 1
no_iters = 5000

seed = 10
tf.set_random_seed(seed)

def ffn(x, hidden1_units, hidden2_units):
  
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, hidden1_units],
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / np.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, no_labels],
                            stddev=1.0 / np.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([no_labels]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases



    
  return logits


def main():

  # Load dataset
  boston = datasets.load_boston()
  x, y = boston.data, boston.target

  # Split dataset into train / test
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=42)

  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train_ = scaler.fit_transform(x_train)
  x_test_ = scaler.transform(x_test)
  y_train_ = y_train.reshape(len(y_train), no_labels)
  y_test_ = y_test.reshape(len(y_test), no_labels)

  print("y_train_", y_train_)

  x = tf.placeholder(tf.float32, [None, no_features])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, no_labels])

  # Build the graph for the deep net
  y = ffn(x, 5, 5)


  error = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis = 1))

  train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)



  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      err = []
      for i in range(no_iters):
        train.run(feed_dict={x: x_train_, y_: y_train_})

        err.append(error.eval(feed_dict={x: x_test_, y_: y_test_}))
        if i % 100 == 0:
          print('step %d, error %g' % (i, err[i]))
            
  # plot learning curves
  plt.figure(1)
  plt.plot(range(no_iters), err)
  plt.xlabel('iterations')
  plt.ylabel('mean square error')
  plt.title('GD learning')
  plt.savefig('5.4a_1.png')

  plt.show()

    

if __name__ == '__main__':
  main()

