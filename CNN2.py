import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


###1.1 定义卷积核|权重 ###
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


### 1.2 定义偏置 ###
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape) + 0.1)


### 1.2 定义卷积层 ###
def conv2d(x, filter):
    return tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')


### 1.3 定义池化层 ###
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


### 2.1 定义网络结构 ###
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

### 2.2 卷积层1 ###

filter_conv1 = weight_variable([5, 5, 1, 32])
filter_b1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, filter_conv1) + filter_b1)
h_pool1 = max_pool_2x2(h_conv1)

### 2.3 卷积层2 ###

filter_conv2 = weight_variable([5, 5, 32, 64])
filter_b2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, filter_conv2) + filter_b2)
h_pool2 = max_pool_2x2(h_conv2)

## 2.4 全连接层 ###

fc_w1 = weight_variable([7 * 7 * 64, 1024])
fc_b1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc_w1) + fc_b1)

### 2.5 dropout(可选)  ###
# 训练时开启，测试时关闭（1）
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

### 2.6 正则化-softmax ###
fc_w2 = weight_variable([1024, 10])
fc_b2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, fc_w2) + fc_b2)

### 3.1 定义训练方法 ###
## 3.1.1 设置loss|cross函数 ##
cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
## 3.1.2 设置优化器 ##
# optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.AdamOptimizer(1e-4)
## 3.1.3 设置优化目标 ##
train_step = optimizer.minimize(cross_entropy)

### 4 准确率 ###
cerrect_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(cerrect_prediction, tf.float32))

### 5 训练 ###
sess.run(tf.global_variables_initializer())

for i in range(101):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        train_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        print('train_step: %s accuracy :%g' % (i, train_acc))

print('fianlly accuracy is %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
