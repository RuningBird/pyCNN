# 复习神经网络

# 　导入数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 导入读取/下载数据集函数

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 读取数据集


# 网络结构函数
def layer(inputs, input_size, output_size, active_function=None):
    W = tf.truncated_normal(shape=[input_size, output_size], stddev=0.5)  # 初始化权值
    b = tf.truncated_normal(shape=[1, output_size], stddev=0.1)  # 初始化偏差
    evidence = tf.matmul(inputs, W) + b
    if active_function:
        return tf.nn.softmax(evidence)
    else:
        return evidence


# 定义Session
sess = tf.InteractiveSession()

# 定义网络结构
# 设置迭代变量
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
prediction = layer(xs, 784, 10, tf.nn.softmax)

####### 配置训练方法 #######

# 定义代价函数
J = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 1))

# 设置优化函数
optimizer = tf.train.GradientDescentOptimizer(0.1)
# 设置优化目标
train = optimizer.minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
# ------------------------ 训练---------------------#
for i in range(1000):
    data = mnist.train.next_batch(50)
    sess.run(train, feed_dict={xs: data[0], ys: data[1]})
    if i % 20 == 0:
        acc_mat = tf.equal(tf.argmax(prediction, 1), tf.argmax(data[1]))
        acc = tf.reduce_mean(tf.cast(acc_mat, tf.float32))
        print("训练准确率：", acc)
