import numpy as np
import pandas
import tensorflow as tf
import csv
import os, time
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

MAX_DOCUMENT_LENGTH = 100
MAX_LABEL = 15
HIDDEN_SIZE = 20

no_epochs = 1000
batch_size = 128
lr = 0.001

#tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_rnn_model(x, celltype):

    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256])

    # char_vectors = tf.one_hot(x, 256)
    # char_list = tf.unstack(char_vectors, axis=1)

    if celltype == "Vanilla":
        print("in vanilla")
        cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
        _, encoding = tf.nn.dynamic_rnn(cell, input_layer, dtype=tf.float32)
    elif celltype == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        _, encoding = tf.nn.dynamic_rnn(cell, input_layer, dtype=tf.float32)
    elif celltype == "LSTM":
        cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
        _, encoding = tf.nn.dynamic_rnn(cell, input_layer, dtype=tf.float32)


    if isinstance(encoding, tf.nn.rnn_cell.LSTMStateTuple) or isinstance(encoding, tuple):
        print("true")
        encoding = encoding[-1]  # state tuple is (c, h), we want h as output

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)


    return input_layer, logits


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

def buildandrunmodel(x_train, y_train, x_test, y_test, celltype):


    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    inputs, logits = char_rnn_model(x, celltype)

    # Optimizer
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        N = len(x_train)
        idx = np.arange(N)

        # training
        test_acc = []
        loss_list = []
        loss_temp = []


        for e in range(no_epochs):

            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                input_layer_, _, loss_ = sess.run([inputs, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
                loss_temp.append(loss_)

            loss_list.append(np.mean(loss_temp))
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
            loss_temp = []

            # if e % 10 == 0:
            #     print('epoch', e, 'entropy', loss_list[-1])
            #     print('iter: %d, test accuracy: %g' % (e, test_acc[e]))

        start = time.time()
        accuracy.eval(feed_dict={x: x_test, y_: y_test})
        end = time.time()
        inference_dur = end - start
        print("Inference runtime: ", inference_dur)

        # plot learning curves
        plt.figure(1)
        plt.clf()
        plt.plot(range(no_epochs), loss_list, 'r', label='Training Loss')
        # plt.legend(loc='upper left')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training Loss for cell type: ' + celltype)
        plt.savefig('layer_compare_char_figs/loss_' + celltype + '.png')

        plt.figure(2)
        plt.clf()
        plt.plot(range(no_epochs), test_acc, 'g', label='Test Accuracy')
        # plt.legend(loc='upper left')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy for cell type: ' + celltype)
        plt.savefig('layer_compare_char_figs/accuracy_' + celltype + '.png' )



def main():
    x_train, y_train, x_test, y_test = read_data_chars()

    print(len(x_train))
    print(len(x_test))

    print("Cell type: LSTM")
    buildandrunmodel(x_train, y_train, x_test, y_test, "LSTM")
    tf.reset_default_graph()
    print("=====================================================================")
    print("Cell type: Vanilla")
    buildandrunmodel(x_train, y_train, x_test, y_test, "Vanilla")
    tf.reset_default_graph()
    print("=====================================================================")
    print("Cell type: GRU")
    buildandrunmodel(x_train, y_train, x_test, y_test, "GRU")
    tf.reset_default_graph()
    print("=====================================================================")







if __name__ == '__main__':
    main()
