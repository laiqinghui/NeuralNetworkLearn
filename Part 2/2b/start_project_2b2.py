import numpy as np
import pandas
import tensorflow as tf
import csv

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 50

no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x):

  # Returns matrix of shape: (5600, 100, 50)
  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  # Return list of 100 matrix of shape: (5600, 50)
  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, word_list, word_vectors

def data_read_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  print("len(list(x_transform_train)): ", len(list(x_transform_train)))
  print("x_train.shape", x_train.shape)
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

def main():
  global n_words

  x_train, y_train, x_test, y_test, n_words = data_read_words()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, word_list, word_vectors = rnn_model(x)


  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # training
  loss = []
  for e in range(no_epochs):
    word_vectors_, word_list_, _, loss_  = sess.run([word_vectors, word_list, train_op, entropy], {x: x_train, y_: y_train})
    loss.append(loss_)
	
    if e%10 == 0:
      print('epoch: %d, entropy: %g'%(e, loss[e]))
      print("word_vectors_.shape: ", word_vectors_.shape)
      print("len(word_list_)", len(word_list_))
      print("word_list_[0].shape", word_list_[0].shape)
      print("word_list_: ")
      print(word_list_)
  
  
if __name__ == '__main__':
  main()
