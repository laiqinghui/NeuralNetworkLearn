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
epochs = 1000
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


def cnn(images, c1_kernel=50, c2_kernel=60):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # Conv 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, c1_kernel], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
                     name='weights_1')
    b1 = tf.Variable(tf.zeros([c1_kernel]), name='biases_1')

    # Output shape: c1_kernel x 24 x 24
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    # Output shape: c1_kernel x 12 x 12
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')

    # Conv 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, c1_kernel, c2_kernel], stddev=1.0 / np.sqrt(c1_kernel * 9 * 9)),
                     name='weights_2')
    b2 = tf.Variable(tf.zeros([c2_kernel]), name='biases_2')

    # Output shape: c2_kernel x 8 x 8
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    # Output shape: c2_kernel x 4 x 4
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

    pool_2_shape = str(pool_2.get_shape()[1].value)

    dim2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim2])

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 4x4xc2_kernel feature maps -- maps this to 300 features.
    W3 = tf.Variable(tf.truncated_normal([4 * 4 * c2_kernel, 300], stddev=1.0),
                     name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)

    # Softmax
    W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0),
                     name='weights_4')
    b4 = tf.Variable(tf.zeros([10]), name='biases_4')
    logits = tf.matmul(fc1, W4) + b4

    return conv_1, pool_1, logits


def buildandrunmodel(trainX, trainY, testX, testY, c1_kernel, c2_kernel):

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    conv_1, pool_1, logits = cnn(x, c1_kernel, c2_kernel)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    N = len(trainX)
    idx = np.arange(N)

    test_acc = []
    loss_list = []
    loss_temp = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                _conv_1, _pool_1, _, loss_ = sess.run([conv_1, pool_1, train_step, loss],
                                                      {x: trainX[start:end], y_: trainY[start:end]})
                # _, loss_ = sess.run([train_step, loss], {x: trainX, y_: trainY})
                loss_temp.append(loss_)

            loss_list.append(np.mean(loss_temp))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            loss_temp = []
            print('iter %d: test accuracy %g' % (e, test_acc[e]))
            print('epoch', e, 'entropy', loss_list[-1])

    ind = np.random.randint(low=0, high=10000)
    X = trainX[ind, :]

    # plot learning curves

    title = "Conv 1 - " + str(c1_kernel) + " and Conv 2 - " + str(c2_kernel)
    plt.figure(1)
    plt.clf()
    plt.plot(range(epochs), loss_list, 'r', label='Training Loss')
    plt.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training Loss for ' + title)
    plt.savefig('q2figs10000/Training Loss for ' + title + '.png')

    plt.figure(2)
    plt.clf()
    plt.plot(range(epochs), test_acc, 'g', label='Test Accuracy')
    plt.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy for ' + title)
    plt.savefig('q2figs10000/Test Accuracy for ' + title + '.png')


def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
    testX = (testX - np.min(testX, axis=0)) / np.max(testX, axis=0)

    # kernels = [(30,170), (40,40), (40,90), (60,60), (70,40), (70,140),
    #            (70,160), (70,190), (80,30), (80,80), (80,100), (90,130),
    #            (100,40), (100,170), (110,30),( 110,120), (120,30),(120,170),
    #            (130,30), (140,20), (140,150), (150,80), (150,80), (150,150),
    #            (160,60), (160,60), (160,70), (180,60), (190,180),]

    for c1 in range(20, 150, 10):
        for c2 in range(20, 150, 10):
            print("C1 size: ", c1, " and C2 size: ", c2)
            buildandrunmodel(trainX, trainY, testX, testY, c1, c2)
            print("=====================================================================")

    # for kernel in kernels:
    #     buildandrunmodel(trainX, trainY, testX, testY, kernel[0], kernel[1])
    #     print("=====================================================================")





if __name__ == '__main__':
    main()
