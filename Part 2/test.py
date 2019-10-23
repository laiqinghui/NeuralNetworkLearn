import tensorflow as tf
import numpy as np

n_in = 8
n_hidden = 5
n_out = 3
n_steps = 64
n_seqs = 16

x_train = np.random.rand(n_seqs, n_steps, n_in)

# build the model
X = tf.placeholder(tf.float32,[None, n_steps, n_in])
Y = tf.split(X, n_steps, axis = 1)
Y_shape = tf.shape(Y)

sess = tf.Session()
X_v,Y_v,Y_shape_v = sess.run([X,Y,Y_shape], {X:x_train}) 
# numpy style
print (X_v.shape)
print (len(Y_v))
print (Y_v[0].shape)
# TF style
#print (len(Y))
#print (Y_shape_v)