import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')  # 把数据放在/tmp/data文件夹中

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)  # 读取数据集


def layer(inputs, size_input, size_output, active_function=None):
    W = tf.Variable(tf.random_normal([size_input, size_output], 0.5, 1))
    b = tf.Variable(tf.zeros([1, size_output]) + 0.1)
    evidence = tf.matmul(inputs, W) + b
    if not active_function:
        return evidence
    else:
        return active_function(evidence)


### 定义网络结构
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
prediction = layer(xs, 784, 10, tf.nn.softmax)

### 定义训练方法
# 代价函数：loss或者cross
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 1))
# 学习方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 学习目标
train = optimizer.minimize(cross_entropy)


## 定义测试方法：
def compute_accuracy(vs, vy):
    global prediction
    y_prediction = sess.run(prediction,feed_dict={xs:vs})#sess
    correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(vy,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy)#方法


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
        if i % 20 == 0:
           acc = compute_accuracy(mnist.test.images,mnist.test.labels)
           print(acc)
