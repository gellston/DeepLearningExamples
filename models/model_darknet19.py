import tensorflow as tf

# 0 cat , 1 dog, 2 elephant, 3 giraffe, 4 horse

class model_animal_v1:

    def __init__(self, sess, name, class_count, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self.class_count = class_count
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name):
            # placeholder 100x100 = 10000
            self.X = tf.placeholder(tf.float32, [None, 224 * 224 * 3], name='input')
            print(self.X)
            X_input = tf.reshape(self.X, [-1, 224, 224, 3])
            self.Y = tf.placeholder(tf.float32, [None, self.class_count])
            print(self.Y)
            self.keep_layer = tf.placeholder(tf.float32, name='dropout')
            print(self.keep_layer)

            # Layer1
            W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
            Bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
            hidden_layer1 = tf.nn.relu(tf.nn.conv2d(X_input, W1, [1, 1, 1, 1], padding='SAME', name='hidden_layer1') + Bias1)
            pool_layer1 = tf.nn.max_pool(hidden_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            drop_layer1 = tf.nn.dropout(pool_layer1, keep_prob=self.keep_layer)

            # Layer2
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            Bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
            hidden_layer2 = tf.nn.relu(tf.nn.conv2d(drop_layer1, W2, [1, 1, 1, 1], padding='SAME', name='hidden_layer2') + Bias2)
            pool_layer2 = tf.nn.max_pool(hidden_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            drop_layer2 = tf.nn.dropout(pool_layer2, keep_prob=self.keep_layer)

            # Layer3
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            Bias3 = tf.Variable(tf.constant(0.1, shape=[128]))
            hidden_layer3 = tf.nn.relu(tf.nn.conv2d(drop_layer2, W3, [1, 1, 1, 1], padding='SAME', name='hidden_layer3') + Bias3)
            drop_layer3 = tf.nn.dropout(hidden_layer3, keep_prob=self.keep_layer)

            # Layer4
            W4 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=0.01))
            Bias4 = tf.Variable(tf.constant(0.1, shape=[64]))
            hidden_layer4 = tf.nn.relu(tf.nn.conv2d(drop_layer3, W4, [1, 1, 1, 1], padding='SAME', name='hidden_layer4') + Bias4)
            drop_layer4 = tf.nn.dropout(hidden_layer4, keep_prob=self.keep_layer)

            # Layer5
            W5 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            Bias5 = tf.Variable(tf.constant(0.1, shape=[128]))
            hidden_layer5 = tf.nn.relu(tf.nn.conv2d(drop_layer4, W5, [1, 1, 1, 1], padding='SAME', name='hidden_layer5') + Bias5)
            pool_layer5 = tf.nn.max_pool(hidden_layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            drop_layer5 = tf.nn.dropout(pool_layer5, keep_prob=self.keep_layer)

            # Layer6
            W6 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
            Bias6 = tf.Variable(tf.constant(0.1, shape=[256]))
            hidden_layer6 = tf.nn.relu(tf.nn.conv2d(drop_layer5, W6, [1, 1, 1, 1], padding='SAME', name='hidden_layer6') + Bias6)
            pool_layer6 = tf.nn.max_pool(hidden_layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            drop_layer6 = tf.nn.dropout(pool_layer6, keep_prob=self.keep_layer)

            print(drop_layer6)

            # Layer11
            flatten = tf.reshape(drop_layer6, [-1, 2 * 2 * 256])
            full_connect_layer1 = tf.layers.dense(flatten, 1000, activation=tf.nn.relu)
            full_connect_dropout1 = tf.nn.dropout(full_connect_layer1, keep_prob=self.keep_layer)
            full_connect_layer2 = tf.layers.dense(full_connect_dropout1, 1000, activation=tf.nn.relu)
            full_connect_dropout2 = tf.nn.dropout(full_connect_layer2, keep_prob=self.keep_layer)
            full_connect_layer3 = tf.layers.dense(full_connect_dropout2, self.class_count)

            self.output = tf.nn.softmax(full_connect_layer3, -1, 'output');
            print(self.output)
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.Y, 1))

            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=full_connect_layer3, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_layer: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.8):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_layer: keep_prop})