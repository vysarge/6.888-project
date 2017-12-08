from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Configuration variables
training = True
updateExistingModel = False

os_slash = '\\'
model_path = 'model' + os_slash
model_name = model_path + 'model'
epochs = 10

steps = 200000

device_name = "/cpu:0"
#device_name = "/gpu:0"


# x is expected to have size 32x32x1 (MNIST)
# build model layer by layer
# return output and a variable to control dropout probability
def buildModel(x):
    layers = []
    with tf.variable_scope('reshape') as scope:
        x_square = tf.reshape(x, [-1, 28, 28, 1])
        x_pad = tf.pad(x_square, [[0,0],[2,2],[2,2],[0,0]], "CONSTANT")
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(x_pad, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.zeros([16]), name='biases')
        pre_act = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_act, name=scope.name)
        layers.append((kernel, biases))
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([3,3,16,16], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.zeros([16]), name='biases')
        pre_act = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_act, name=scope.name)
        layers.append((kernel, biases))
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.random_normal([3,3,16,8], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.zeros([8]), name='biases')
        pre_act = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_act, name=scope.name)
        layers.append((kernel, biases))
    with tf.variable_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    with tf.variable_scope('dropout'):
        p_retain = tf.placeholder(tf.float32)
        flat = tf.reshape(pool3, [-1, 8*8*8])
        drop1 = tf.nn.dropout(flat, p_retain)
    with tf.variable_scope('fc1'):
        kernel = tf.Variable(tf.random_normal([8*8*8, 10]))
        conv = tf.matmul(drop1, kernel)
        biases = tf.Variable(tf.zeros([10]))
        fc1 = tf.nn.bias_add(conv, biases)
        layers.append((kernel, biases))
    return layers, fc1, p_retain

def train():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
        # Import data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        layers, y_conv, p_retain = buildModel(x)

        # create saver
        #saver = tf.train.Saver()

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(5e-5).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(steps):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], p_retain: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], p_retain: 0.5})

            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, p_retain: 1.0}))
            for layer in layers:
                kernel, bias = layer
                print("kernel------------------------------------------")
                print(sess.run(kernel))
                print("bias--------------------------------------------")
                print(sess.run(bias))
                print("------------------------------------------------")

if training:
    train()



