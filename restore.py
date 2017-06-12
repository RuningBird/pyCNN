import tensorflow as tf
v1 = tf.Variable(5555, name="v1")
v2 = tf.Variable(3, name="v2")
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'd:/tmp/model.ckpt')
    print("model has been restored",sess.run(v1),'-',sess.run(v2))
