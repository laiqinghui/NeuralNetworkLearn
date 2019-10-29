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

MAX_DOCUMENT_LENGTH = 100
MAX_LABEL = 15
HIDDEN_SIZE = 20

no_epochs = 500
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_rnn_model(x):
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256])

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.dynamic_rnn(cell, input_layer, dtype=tf.float32)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features
    keep_prob = tf.placeholder(tf.float32)

    encoding = tf.nn.dropout(encoding, keep_prob)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)


    return keep_prob, input_layer, logits


def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            print(row[1])
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

def buildandrunmodel(x_train, y_train, x_test, y_test, keep_proba):

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    keep_prob, inputs, logits = char_rnn_model(x)

    # Optimizer
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        test_acc = []

        for e in range(no_epochs):
            input_layer_, _, loss_ = sess.run([inputs, train_op, entropy], {x: x_train, y_: y_train, keep_prob: keep_proba})
            loss.append(loss_)
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: keep_proba}))
            # print("input_layer_.shape", input_layer_.shape)

            if e % 1 == 0:
                print('iter: %d, entropy: %g' % (e, loss[e]))
                print('iter: %d, test accuracy: %g' % (e, test_acc[e]))

        # plot learning curves
        plt.figure(1)
        plt.plot(range(no_epochs), loss, 'r', label='Training Loss')
        # plt.legend(loc='upper left')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training Loss for keep prop: ' + str(keep_proba))
        plt.savefig('char_rnn_dropout_figs/loss_' + str(keep_proba) + '.png')

        plt.figure(2)
        plt.plot(range(no_epochs), test_acc, 'g', label='Test Accuracy')
        # plt.legend(loc='upper left')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy for keep prop: ' + str(keep_proba))
        plt.savefig('char_rnn_dropout_figs/accuracy_' + str(keep_proba) + '.png' )



def main():
    x_train, y_train, x_test, y_test = read_data_chars()

    print(len(x_train))
    print(len(x_test))

    for kp in range(1, 10, 2):
        print("Keep prob: ", kp/10)
        buildandrunmodel(x_train, y_train, x_test, y_test, kp/10)
        tf.reset_default_graph()
        print("=====================================================================")






if __name__ == '__main__':
    main()
