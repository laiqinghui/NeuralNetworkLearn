import tensorflow as tf

x = 2
y = 3

add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')
useless_op = tf.multiply(x, add_op, name='Unused')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./fypgraphs', sess.graph)
    pow_out, useless_out = sess.run([pow_op, useless_op])