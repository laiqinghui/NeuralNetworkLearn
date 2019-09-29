#
#   Chapter 5, example 4a
#

import numpy as np
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf
import pylab as plt
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

learning_rate = 0.01

no_features = 21
no_labels = 3
no_epochs = 500

seed = 10
tf.set_random_seed(seed)

def ffn(x, hidden1_units):
  
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, hidden1_units], #(inputs, no.ofneurons)
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
    
    
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, no_labels],
                            stddev=1.0 / np.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([no_labels]),
                         name='biases')
    logits = tf.matmul(hidden1, weights) + biases
    # logits = tf.exp(u) / tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)

    # logits = tf.argmax(p, axis=1)
    # logits = tf.cast(logits, tf.float32)
    
  return logits


def main():

  # Read the .csv files as pandas dataframe
  train_raw = pd.read_csv('ctg_data_cleaned.csv')
  y = train_raw.NSP.to_numpy()
  x = train_raw.drop(['NSP'], axis=1).to_numpy()

  identity = np.identity(no_labels, dtype=np.uint8)
  y = identity[y-1]
    # print("y.shape: ", y.shape)


  # # Split dataset into train / test
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=0.3, random_state=42)

  print("y_train: ", y_train.shape)
  print("x_train.shape", x_train.shape)
  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train_ = scaler.fit_transform(x_train)
  x_test_ = scaler.transform(x_test)


  # Computational graph starts

  x = tf.placeholder(tf.float32, [None, no_features])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, no_labels])

  # Build the graph for the deep net
  y = ffn(x, 10)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(cross_entropy)

  # Add a scalar summary for the snapshot loss.
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(cross_entropy, global_step=global_step)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  N = len(x_train)
  idx = np.arange(N)
  batch_size = 128

  training_acc = []
  testing_acc = []

  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(no_epochs):
        np.random.shuffle(idx)
        x_train_ = x_train_[idx]
        y_train = y_train[idx]

        kf = KFold(n_splits=5)
        for train_idx, val_idx in kf.split(x_train_, y_train):
          train_x = x_train_[train_idx]
          train_y = y_train[train_idx]
          val_x = x_train_[val_idx]
          val_y = y_train[val_idx]

          for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            train_op.run(feed_dict={x: train_x[start:end], y_: train_y[start:end]})

            if i % 10 == 0:
              train_acc = accuracy.eval(feed_dict={x: val_x, y_: val_y})
              training_acc.append(train_acc)
              test_acc = accuracy.eval(feed_dict={x: x_test_, y_: y_test})
              testing_acc.append(test_acc)
              print('batch %d: iter %d, train accuracy %g' % (batch_size, i, train_acc))
              print('batch %d: iter %d, test accuracy %g' % (batch_size, i, test_acc))



        # t = time.time()




  #plot learning curves
  plt.figure(1)
  plt.plot(range(1, no_epochs, 10), training_acc)
  plt.xlabel('iterations')
  plt.ylabel('Train acc')
  plt.title('SGD learning')
  plt.savefig('q1_train.png')

  plt.figure(1)
  plt.plot(range(1, no_epochs, 10), testing_acc)
  plt.xlabel('iterations')
  plt.ylabel('Test acc')
  plt.title('SGD learning')
  plt.savefig('q1_test.png')

  plt.show()

    

if __name__ == '__main__':
  main()

