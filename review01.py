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
h_conv2 = tf.nn.relu(conv2d(h_poo1l, kernel_conv2) + bias_conv2)
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

################################## 4, 学习方法设置 ###############################
### 4.1 定义代价函数：corss | loss
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction)))
cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
### 4.2 定义学习（优化）方法 ###
optimizer = tf.train.AdamOptimizer(1e-4)
## 4.3 定义学习（优化）方向 ###
train_step = optimizer.minimize(cross_entropy)

################################## 5, 评估方法设置#######################################
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

################################## 6, 训练 #######################################
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y: batch[1], drop_prob: 0.5})
    if i % 50 == 0:
        print("at step %s ,accuracy is %s" % (i, accuracy.eval(feed_dict={x: batch[0], y: batch[1], drop_prob: 1.0})))
print("finall accuracy is %s", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, drop_prob: 1.0}))
