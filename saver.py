import tensorflow as tf

v1 = tf.Variable(1, name="v1")
v2 = tf.Variable(2, name="v2")
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, 'd:/tmp/model.ckpt')
    print('Model saved in：', save_path)




