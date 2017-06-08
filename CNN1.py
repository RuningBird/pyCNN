import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')  # 把数据放在/tmp/data文件夹中

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)  # 读取数据集


def layer(inputs, in_size, out_size, active_function):
    W = tf.Variable(tf.random_normal([in_size, out_size], mean=0.5, stddev=0.2))
    b = tf.Variable(tf.zeros([out_size]) + 0.1)
    evidence = tf.matmul(inputs, W) + b
    if active_function is None:
        return evidence
    else:
        return active_function(evidence)
