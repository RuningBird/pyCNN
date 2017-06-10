import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### 定义数据流图对象 ###
sess = tf.InteractiveSession()


### 1.1 定义构造函数：卷积核|权值 ###
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


### 1.2 定义构造函数：偏置 ###
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape) + 0.1)


### 1.3 定义卷积函数 ##
def conv2d(inputs, kernel):
    return tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')


### 1.4 定义池化函数
def max_pool_2x2(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


### 1.3 定义placeholder ###
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

### 2 定义网络结构 ###
x_image = tf.reshape(x, [-1, 28, 28, 1])

### 2.1 卷积层1 ###
kernel_conv1 = weight_variable([5, 5, 1, 32])
bias_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, kernel_conv1) + bias_conv1)
h_poo1l = max_pool_2x2(h_conv1)
## 此时图像为：32*【14，14】

### 2.2 卷积层2 ###
kernel_conv2 = weight_variable([5, 5, 32, 64])
bias_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(x_image, kernel_conv1) + bias_conv1)
h_poo12 = max_pool_2x2(h_conv2)
## 此时图像为：64*【7，7】

### 2.3 全连接层 ###
fc1_wight = weight_variable([7 * 7 * 64, 1024])
fc1_bias = bias_variable([1024])

h_poo12_flat = tf.reshape(h_poo12, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_poo12_flat, fc1_wight) + fc1_bias)

### 2.4 dropout ###
drop_prob = tf.placeholder(tf.float32)
h_dropout = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

### 2.5 softmax ###
fc2_weight = weight_variable([1024, 10])
fc2_bias = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_dropout, fc2_weight) + fc2_bias)
