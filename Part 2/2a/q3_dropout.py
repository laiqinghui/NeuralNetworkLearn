import math
import tensorflow as tf
import numpy as np
#import pylab as plt
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 500
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1
    print(labels_)

    return data, labels_


def cnn(images, c1_kernel=9, c2_kernel=5):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # Conv 1
    W1 = tf.Variable(tf.truncated_normal([c1_kernel, c1_kernel, NUM_CHANNELS, 50], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
                     name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')

    # Output shape: 50 x 24 x 24
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    # Output shape: 50 x 12 x 12
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')

    # Conv 2
    W2 = tf.Variable(tf.truncated_normal([c2_kernel, c2_kernel, 50, 60], stddev=1.0 / np.sqrt(50 * 9 * 9)),
                     name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')

    # Output shape: 60 x 8 x 8
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    # Output shape: 60 x 4 x 4
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')

    pool_2_shape = str(pool_2.get_shape()[1].value)

    dim2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim2])

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 4x4x60 feature maps -- maps this to 300 features.
    W3 = tf.Variable(tf.truncated_normal([4 * 4 * 60, 300], stddev=1.0),
                     name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Softmax
    W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0),
                     name='weights_4')
    b4 = tf.Variable(tf.zeros([10]), name='biases_4')
    logits = tf.matmul(fc1_drop, W4) + b4

    return logits, keep_prob


def main():
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
    testX = (testX - np.min(testX, axis=0)) / np.max(testX, axis=0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    logits, keep_prob = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_step1 = tf.train.MomentumOptimizer(1e-4, 0.1).minimize(cross_entropy)
    train_step2 = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy)
    train_step3 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    N = len(trainX)
    idx = np.arange(N)

    with tf.Session() as sess:

        print('momentum...')
        sess.run(tf.global_variables_initializer())

        test_acc = []
        loss_list = []
        loss_temp = []

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                _, loss_ = sess.run([train_step1, loss], {x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.5})
                loss_temp.append(loss_)


            loss_list.append(np.mean(loss_temp))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            loss_temp = []
            print('iter %d: test accuracy %g' % (e, test_acc[e]))
            print('epoch', e, 'entropy', loss_list[-1])

        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(epochs), test_acc, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), loss_list, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Adding the momentum term with momentum 𝛾=0.1')
        plt.savefig('q3figs/dropout/momentum.png')

        # ================================================================================================

        print('RMSProp...')
        sess.run(tf.global_variables_initializer())

        test_acc = []
        loss_list = []
        loss_temp = []

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                _, loss_ = sess.run([train_step2, loss], {x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.5})
                loss_temp.append(loss_)

            loss_list.append(np.mean(loss_temp))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            loss_temp = []
            print('iter %d: test accuracy %g' % (e, test_acc[e]))
            print('epoch', e, 'entropy', loss_list[-1])

        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(epochs), test_acc, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), loss_list, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Using RMSProp algorithm for learning')
        plt.savefig('q3figs/dropout/RMSProp.png')

        # ================================================================================================

        print('Adam optimizer...')
        sess.run(tf.global_variables_initializer())

        test_acc = []
        loss_list = []
        loss_temp = []

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                _, loss_ = sess.run([train_step3, loss], {x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.5})
                loss_temp.append(loss_)

            loss_list.append(np.mean(loss_temp))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            loss_temp = []
            print('iter %d: test accuracy %g' % (e, test_acc[e]))
            print('epoch', e, 'entropy', loss_list[-1])

        plt.figure(3)
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(epochs), test_acc, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), loss_list, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Using Adam optimizer for learning')
        plt.savefig('q3figs/dropout/AdamOptimizer.png')






if __name__ == '__main__':
    main()
