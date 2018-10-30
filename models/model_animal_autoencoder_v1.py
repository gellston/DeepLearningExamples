import tensorflow as tf

# 0 cat , 1 dog, 2 elephant, 3 giraffe, 4 horse

class model_animal_autoencoder_v1:

    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name):
            # placeholder 100x100 = 10000

            self.X = tf.placeholder(tf.float32, [None, 100 * 100 * 3], name='input')
            self.keep_layer = tf.placeholder(tf.float32, name='dropout')

            # input
            X_input = tf.reshape(self.X, [-1, 100, 100, 3])
            print(X_input)

            # layer1 encode1
            w1 = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
            bias1 = tf.Variable(tf.constant(0.1, shape=[64]))
            conv1 = tf.nn.relu(tf.nn.conv2d(X_input, w1, strides=[1, 2, 2, 1], padding='SAME') + bias1)
            print(conv1)

            # layer2 encode2
            w2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            bias2 = tf.Variable(tf.constant(0.1, shape=[128]))
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1, 2, 2, 1], padding='SAME') + bias2)
            print(conv2)

            # layer3 encode3
            w3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
            bias3 = tf.Variable(tf.constant(0.1, shape=[256]))
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w3, strides=[1, 2, 2, 1], padding='SAME') + bias3)
            print(conv3)

            # layer4 encode4
            w4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
            bias4 = tf.Variable(tf.constant(0.1, shape=[512]))
            conv4 = tf.nn.relu(tf.nn.conv2d(conv3, w4, strides=[1, 2, 2, 1], padding='SAME') + bias4)
            print(conv4)

            # layer5 decode1
            w5 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
            bias5 = tf.Variable(tf.constant(0.1, shape=[256]))
            conv5 = tf.nn.relu(tf.nn.conv2d_transpose(conv4, w5, output_shape=[tf.shape(X_input)[0], 13, 13, 256], strides=[1, 2, 2, 1], padding='SAME') + bias5)
            print(conv5)

            # layer6 decode2
            w6 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
            bias6 = tf.Variable(tf.constant(0.1, shape=[128]))
            conv6 = tf.nn.relu(tf.nn.conv2d_transpose(conv5, w6, output_shape=[tf.shape(X_input)[0], 25, 25, 128], strides=[1, 2, 2, 1], padding='SAME') + bias6)
            print(conv6)

            # layer7 decode3
            w7 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            bias7 = tf.Variable(tf.constant(0.1, shape=[64]))
            conv7 = tf.nn.relu(tf.nn.conv2d_transpose(conv6, w7, output_shape=[tf.shape(X_input)[0], 50, 50, 64], strides=[1, 2, 2, 1], padding='SAME') + bias7)
            print(conv7)

            # layer8 decode4
            w8= tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
            bias8 = tf.Variable(tf.constant(0.1, shape=[3]))
            conv8 = tf.nn.relu(tf.nn.conv2d_transpose(conv7, w8, output_shape=[tf.shape(X_input)[0], 100, 100, 3], strides=[1, 2, 2, 1], padding='SAME') + bias8)
            print(conv8)

            self.output = conv8 * 255

            self.cost = tf.reduce_mean(tf.pow(X_input - self.output, 2))
            self.accuracy = (255 - tf.sqrt(self.cost))/255
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def reconstruct(self, x_test, keep_prop=1.0):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop})

    def train(self, x_data, keep_prop=0.8):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.keep_layer: keep_prop})