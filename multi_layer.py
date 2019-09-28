import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

layer_1 = 128
layer_2 = 64
layer_3 = 32
layer_out = 10

weight_1 = tf.Variable(tf.truncated_normal([784, layer_1], stddev=0.1))  # with truncated_normal method we first put random variables to the weight matrix and if we don't put stddev value gaps between them will be too much
bias_1 = tf.Variable(tf.constant(0.1, shape=[layer_1]))
weight_2 = tf.Variable(tf.truncated_normal([layer_1, layer_2], stddev=0.1))
bias_2 = tf.Variable(tf.constant(0.1, shape=[layer_2]))
weight_3 = tf.Variable(tf.truncated_normal([layer_2, layer_3], stddev=0.1))
bias_3 = tf.Variable(tf.constant(0.1, shape=[layer_3]))
weight_out = tf.Variable(tf.truncated_normal([layer_3, layer_out], stddev=0.1))
bias_out = tf.Variable(tf.constant(0.1, shape=[layer_out]))

y1 = tf.nn.relu(tf.matmul(x, weight_1) + bias_1)
y2 = tf.nn.relu(tf.matmul(y1, weight_2) + bias_2)
y3 = tf.nn.relu(tf.matmul(y2, weight_3) + bias_3)
logits = tf.matmul(y3, weight_out) + bias_out
y4 = tf.nn.softmax(logits)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128


def training_steps(iterations):
    for i in range(iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimize, loss], feed_dict=feed_dict_train)

        if i%100 == 0 :
            training_acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print("Iteration:", i, " Loss:", train_loss, " Accuracy:", training_acc)


def accuracy_test():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy is : ", acc)


training_steps(10000)
accuracy_test()
