import tensorflow as tf

# a = tf.Variable(2)
# init = tf.global_variables_initializer()

with tf.Session() as sess:
    print(sess.run(tf.zeros([2,3])))
