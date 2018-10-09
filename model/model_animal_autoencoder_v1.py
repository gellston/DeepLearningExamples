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
            print(self.X)
            X_input = tf.reshape(self.X, [-1, 100, 100, 3])
            self.keep_layer = tf.placeholder(tf.float32, name='dropout')
            print(self.keep_layer)

            # layer1 encode1
            w1 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01))
            bias1 = tf.Variable(tf.constant(0.1, shape=[1]))
            conv1 = tf.nn.sigmoid(tf.nn.conv2d(X_input, w1, strides=[1, 2, 2, 1], padding='SAME') + bias1)
            drop1 = tf.nn.dropout(conv1, self.keep_layer)
            print(drop1)

            # layer2 encode2
            w2 = tf.Variable(tf.random_normal([5, 5, 1, 16], stddev=0.01))
            bias2 = tf.Variable(tf.constant(0.1, shape=[16]))
            conv2 = tf.nn.sigmoid(tf.nn.conv2d(drop1, w2, strides=[1, 2, 2, 1], padding='SAME') + bias2)
            drop2 = tf.nn.dropout(conv2, self.keep_layer)
            print(drop2)

            # layer3 encode3
            w3 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01))
            bias3 = tf.Variable(tf.constant(0.1, shape=[32]))
            conv3 = tf.nn.sigmoid(tf.nn.conv2d(drop2, w3, strides=[1, 2, 2, 1], padding='SAME') + bias3)
            drop3 = tf.nn.dropout(conv3, self.keep_layer)
            print(drop3)


            # layer4 decode1
            w4 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01))
            bias4 = tf.Variable(tf.constant(0.1, shape=[16]))
            conv4 = tf.nn.sigmoid(tf.nn.conv2d_transpose(drop3, w4, output_shape=[tf.shape(X_input)[0], 25, 25, 16], strides=[1, 2, 2, 1], padding='SAME') + bias4)
            drop4 = tf.nn.dropout(conv4, self.keep_layer)
            print(drop4)

            # layer5 decode2
            w5 = tf.Variable(tf.random_normal([5, 5, 1, 16], stddev=0.01))
            bias5 = tf.Variable(tf.constant(0.1, shape=[1]))
            conv5 = tf.nn.sigmoid(tf.nn.conv2d_transpose(drop4, w5, output_shape=[tf.shape(X_input)[0], 50, 50, 1], strides=[1, 2, 2, 1], padding='SAME') + bias5)
            drop5 = tf.nn.dropout(conv5, self.keep_layer)
            print(drop5)

            # layer6 decode3
            w6 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01))
            bias6 = tf.Variable(tf.constant(0.1, shape=[3]))
            conv6 = tf.nn.sigmoid(tf.nn.conv2d_transpose(drop5, w6, output_shape=[tf.shape(X_input)[0], 100, 100, 3], strides=[1, 2, 2, 1], padding='SAME') + bias6)
            drop6 = tf.nn.dropout(conv6, self.keep_layer)
            self.output = drop6
            print(drop6)

            self.cost = tf.reduce_mean(tf.pow(X_input - drop6, 2))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)


    def reconstruct(self, x_test, keep_prop=1.0):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop})

    def train(self, x_data, keep_prop=0.8):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.keep_layer: keep_prop})