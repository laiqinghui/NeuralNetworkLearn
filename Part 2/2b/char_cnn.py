#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas
import tensorflow as tf
import csv
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt


# In[37]:


MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]

POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

BATCH_SIZE = 128

no_epochs = 1000
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


#flags
dropout_flag = True


# In[38]:


def char_cnn_model(x):

    input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    keep_prob = tf.placeholder(tf.float32)

    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')

    if dropout_flag:
        pool1 = tf.nn.dropout(pool1, keep_prob)


    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    
    if dropout_flag:
        pool2 = tf.nn.dropout(pool2, keep_prob)

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1]) 

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

    return keep_prob, input_layer, logits


# In[39]:


def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

 


# In[65]:


def buildandrunmodel(x_train, y_train, x_test, y_test, keep_proba):

    x_train, y_train, x_test, y_test = read_data_chars()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    keep_prob, inputs, logits = char_cnn_model(x)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        test_acc = []

        for e in range(no_epochs):

            rand_index = np.random.choice(x_train.shape[0], size=BATCH_SIZE)
            x_batch = x_train[rand_index]
            y_batch = y_train[rand_index]

            if(e==0):
                print(x_train.shape)

            _, loss_  = sess.run([train_op, entropy], {x: x_batch, y_: y_batch, keep_prob:keep_proba})
            loss.append(loss_)


            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob:keep_proba}))


            if e%1 == 0:
                print('iter: %d, entropy: %g, accuracy: %g'%(e, loss[e], test_acc[e]))


    plt.figure(1)
    plt.plot(range(no_epochs), loss, 'r' ,label="Training Loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.savefig("q1figs/loss_keepprob" + str(keep_proba) + ".png")
    plt.close()
    
    plt.figure(2)
    plt.plot(range(no_epochs), test_acc, 'g' ,label="Test Accuracy")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig("q1figs/accuracy_keepprob" + str(keep_proba) + ".png")
    plt.close()


# In[66]:


def main():
    global n_words
    x_train, y_train, x_test, y_test = read_data_chars()
    
    for kp in range(1,10,2):
        print("Keep prob: ", kp/10)
        tf.reset_default_graph()
        buildandrunmodel(x_train, y_train, x_test, y_test, kp/10)
        print("=====================================================================")
        
    print("Keep prob: ", 1)
    tf.reset_default_graph()
    buildandrunmodel(x_train, y_train, x_test, y_test, 1)
    print("=====================================================================")


# In[67]:


main()


# In[ ]:




