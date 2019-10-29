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
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

no_epochs = 1000
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

    # Dropout - controls the complexity of the model, prevents co-adaptation of features
    keep_prob = tf.placeholder(tf.float32)

    encoding = tf.nn.dropout(encoding, keep_prob)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return keep_prob, logits, word_list, word_vectors


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

def buildandrunmodel(x_train, y_train, x_test, y_test, keep_proba):

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    keep_prob, logits, word_list, word_vectors = rnn_model(x)

    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        test_acc = []

        for e in range(no_epochs):
            word_vectors_, word_list_, _, loss_ = sess.run([word_vectors, word_list, train_op, entropy],
                                                           {x: x_train, y_: y_train, keep_prob: keep_proba})
            loss.append(loss_)
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: keep_proba}))

            if e % 10 == 0:
                print('epoch: %d, entropy: %g' % (e, loss[e]))
                print('iter: %d, test accuracy: %g' % (e, test_acc[e]))

        # plot learning curves
        plt.figure(1)
        plt.plot(range(no_epochs), loss, 'r', label='Training Loss')
        # plt.legend(loc='upper left')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training Loss for keep prop: ' + str(keep_proba))
        plt.savefig('word_rnn_dropout_figs/loss_' + str(keep_proba) + '.png')

        plt.figure(2)
        plt.plot(range(no_epochs), test_acc, 'g', label='Test Accuracy')
        # plt.legend(loc='upper left')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy for keep prop: ' + str(keep_proba))
        plt.savefig('word_rnn_dropout_figs/accuracy_' + str(keep_proba) + '.png')


def main():
    global n_words

    x_train, y_train, x_test, y_test, n_words = data_read_words()

    for kp in range(1, 10, 2):
        print("Keep prob: ", kp/10)
        buildandrunmodel(x_train, y_train, x_test, y_test, kp/10)
        tf.reset_default_graph()
        print("=====================================================================")




if __name__ == '__main__':
    main()
