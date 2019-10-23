import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 10
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

    return data, labels_


def cnn(images):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # Conv 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
                     name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')

    # Output shape: 50 x 24 x 24
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    # Output shape: 50 x 12 x 12
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')



    # Conv 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0 / np.sqrt(50 * 9 * 9)),
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
    W3 = tf.Variable(tf.truncated_normal([4*4*60, 300], stddev=1.0),
                     name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)

    # Softmax
    W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0),
                     name='weights_4')
    b4 = tf.Variable(tf.zeros([10]), name='biases_4')
    logits = tf.matmul(fc1, W4) + b4

    return logits


def main():
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            _, loss_ = sess.run([train_step, loss], {x: trainX, y_: trainY})

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            print('iter %d: test accuracy %g' % (e, test_acc[e]))
            print('epoch', e, 'entropy', loss_)

    ind = np.random.randint(low=0, high=10000)
    X = trainX[ind, :]

    plt.figure()
    plt.gray()
    X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)
    plt.savefig('./p1b_2.png')


if __name__ == '__main__':
    main()
