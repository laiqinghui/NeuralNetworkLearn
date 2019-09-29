import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    os.makedirs('figures')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

num_iters = 3000
num_features = 2
num_classes = 3
lr = 0.05

SEED = 10
np.random.seed(SEED)

# data
X = np.array([[0.94, 0.18],[-0.58, -0.53],[-0.23, -0.31],[0.42, -0.44],
              [0.5, -1.66],[-1.0, -0.51],[0.78, -0.65],[0.04, -0.20]])
Y = np.array([0, 1, 1, 0, 2, 1, 0, 2])
K = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]]).astype(float)

print(X)
print(Y)
print(lr)

# Model parameters
w = tf.Variable(np.random.rand(num_features, num_classes), dtype=tf.float32)
b = tf.Variable(tf.zeros([num_classes]))

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
k = tf.placeholder(tf.float32, K.shape)

u = tf.matmul(x, w) + b
p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)

y = tf.argmax(p, axis=1)


loss = -tf.reduce_sum(tf.log(p)*k)
err = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(k, 1), y), tf.int32))

grad_u = -(k - p)
grad_w = tf.matmul(tf.transpose(x), grad_u)
grad_b = tf.reduce_sum(grad_u, axis = 0)

w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

loss_, err_ = [], []
for i in range(num_iters):

  if (i == 0):
    u_, p_, y_, l_, e_, grad_u_, grad_w_, grad_b_ = sess.run(
        [u, p, y, loss, err, grad_u, grad_w, grad_b], {x: X, k:K})

    print('iter: {}'.format(i+1))
    print('u: {}'.format(u_))
    print('p: {}'.format(p_))
    print('y: {}'.format(y_))
    print('entropy: {}'.format(l_))
    print('error: {}'.format(e_))
    print('grad_u: {}'.format(grad_u_))
    print('grad_w: {}'.format(grad_w_))
    print('grad_b: {}'.format(grad_b_))
   
  sess.run([w_new, b_new], {x:X, k:K})
  l, e = sess.run([loss, err], {x:X, k:K})
  loss_.append(l)
  err_.append(e)

  if (i == 0):
    w_, b_ = sess.run([w, b])
    print('w: {}, b: {}'.format(w_, b_))

  if not i%100:
    print('epoch:{}, loss:{}, error:{}'.format(i,loss_[i], err_[i]))

# evaluate training accuracy
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x:X, k:K})
print("w: %s b: %s"%(curr_w, curr_b))
print("entropy: %g"%curr_loss)

plt.figure(1)
plot_pred = plt.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class 1')
plot_original = plt.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class 2')
plot_original = plt.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class 3')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('data points')
plt.legend()
plt.savefig('./figures/4.3_1.png')

plt.figure(2)
plt.plot(range(num_iters), loss_)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/4.3_2.png')

plt.figure(3)
plt.plot(range(num_iters), err_)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/4.3_3.png')


plt.show()

