import tensorflow as tf
import tensorflow.contrib as contrib
# 0 cat , 1 dog, 2 elephant, 3 giraffe, 4 horse

class model_darknet19:

    def __init__(self, sess, name, class_count, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self.class_count = class_count
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name):
            # placeholder 100x100 = 10000
            self.X = tf.placeholder(tf.float32, [None, 100 * 100 * 3], name='input')
            print(self.X)

            self.keep_layer = tf.placeholder(tf.bool, name='phase')
            print(self.keep_layer)

            self.Y = tf.placeholder(tf.float32, [None, self.class_count])
            print(self.Y)

            X_input = tf.reshape(self.X, [-1, 100, 100, 3])

            # Layer1
            W1 = tf.get_variable("W1", shape=[3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer1 = tf.nn.conv2d(X_input, W1, [1, 1, 1, 1], padding='SAME', name='hidden_layer1')
            hidden_layer1 = contrib.layers.batch_norm(hidden_layer1, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer1 = tf.nn.leaky_relu(hidden_layer1)
            pool_layer1 = tf.nn.max_pool(hidden_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(pool_layer1)


            # Layer2
            W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer2 = tf.nn.conv2d(pool_layer1, W2, [1, 1, 1, 1], padding='SAME', name='hidden_layer2')
            hidden_layer2 = contrib.layers.batch_norm(hidden_layer2, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer2 = tf.nn.leaky_relu(hidden_layer2)
            pool_layer2 = tf.nn.max_pool(hidden_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(pool_layer2)

            # Layer3
            W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer3 = tf.nn.conv2d(pool_layer2, W3, [1, 1, 1, 1], padding='SAME', name='hidden_layer3')
            hidden_layer3 = contrib.layers.batch_norm(hidden_layer3, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer3 = tf.nn.leaky_relu(hidden_layer3)
            print(hidden_layer3)

            # Layer4
            W4 = tf.get_variable("W4", shape=[1, 1, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer4 = tf.nn.conv2d(hidden_layer3, W4, [1, 1, 1, 1], padding='VALID', name='hidden_layer4')
            hidden_layer4 = contrib.layers.batch_norm(hidden_layer4, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer4 = tf.nn.leaky_relu(hidden_layer4)
            print(hidden_layer4)

            # Layer5
            W5 = tf.get_variable("W5", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer5 = tf.nn.conv2d(hidden_layer4, W5, [1, 1, 1, 1], padding='SAME', name='hidden_layer5')
            hidden_layer5 = contrib.layers.batch_norm(hidden_layer5, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer5 = tf.nn.leaky_relu(hidden_layer5)
            pool_layer5 = tf.nn.max_pool(hidden_layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(pool_layer5)

            # Layer6
            W6 = tf.get_variable("W6", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer6 = tf.nn.conv2d(pool_layer5, W6, [1, 1, 1, 1], padding='SAME', name='hidden_layer6')
            hidden_layer6 = contrib.layers.batch_norm(hidden_layer6, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer6 = tf.nn.leaky_relu(hidden_layer6)
            print(hidden_layer6)

            # Layer7
            W7 = tf.get_variable("W7", shape=[1, 1, 256, 128], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer7 = tf.nn.conv2d(hidden_layer6, W7, [1, 1, 1, 1], padding='VALID', name='hidden_layer7')
            hidden_layer7 = contrib.layers.batch_norm(hidden_layer7, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer7 = tf.nn.leaky_relu(hidden_layer7)
            print(hidden_layer7)

            # Layer8
            W8 = tf.get_variable("W8", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer8 = tf.nn.conv2d(hidden_layer7, W8, [1, 1, 1, 1], padding='SAME', name='hidden_layer8')
            hidden_layer8 = contrib.layers.batch_norm(hidden_layer8, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer8 = tf.nn.leaky_relu(hidden_layer8)
            pool_layer8 = tf.nn.max_pool(hidden_layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(pool_layer8)
            
            # Layer9
            W9 = tf.get_variable("W9", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer9 = tf.nn.conv2d(pool_layer8, W9, [1, 1, 1, 1], padding='SAME', name='hidden_layer9')
            hidden_layer9 = contrib.layers.batch_norm(hidden_layer9, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer9 = tf.nn.leaky_relu(hidden_layer9)
            print(hidden_layer9)
            
            # Layer10
            W10 = tf.get_variable("W10", shape=[1, 1, 512, 256], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer10 = tf.nn.conv2d(hidden_layer9, W10, [1, 1, 1, 1], padding='VALID', name='hidden_layer10')
            hidden_layer10 = contrib.layers.batch_norm(hidden_layer10, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer10 = tf.nn.leaky_relu(hidden_layer10)
            print(hidden_layer10)

            # Layer11
            W11 = tf.get_variable("W11", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer11 = tf.nn.conv2d(hidden_layer10, W11, [1, 1, 1, 1], padding='SAME', name='hidden_layer11')
            hidden_layer11 = contrib.layers.batch_norm(hidden_layer11, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer11 = tf.nn.leaky_relu(hidden_layer11)
            print(hidden_layer11)

            # Layer12
            W12 = tf.get_variable("W12", shape=[1, 1, 512, 256], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer12 = tf.nn.conv2d(hidden_layer11, W12, [1, 1, 1, 1], padding='VALID', name='hidden_layer12')
            hidden_layer12 = contrib.layers.batch_norm(hidden_layer12, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer12 = tf.nn.leaky_relu(hidden_layer12)
            print(hidden_layer12)

            # Layer13
            W13 = tf.get_variable("W13", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer13 = tf.nn.conv2d(hidden_layer12, W13, [1, 1, 1, 1], padding='SAME', name='hidden_layer13')
            hidden_layer13 = contrib.layers.batch_norm(hidden_layer13, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer13 = tf.nn.leaky_relu(hidden_layer13)
            pool_layer13 = tf.nn.max_pool(hidden_layer13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(pool_layer13)
            
            # Layer14
            W14 = tf.get_variable("W14", shape=[3, 3, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer14 = tf.nn.conv2d(pool_layer13, W14, [1, 1, 1, 1], padding='SAME', name='hidden_layer14')
            hidden_layer14 = contrib.layers.batch_norm(hidden_layer14, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer14 = tf.nn.leaky_relu(hidden_layer14)
            print(hidden_layer14)
            
            # Layer15
            W15 = tf.get_variable("W15", shape=[1, 1, 1024, 512], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer15 = tf.nn.conv2d(hidden_layer14, W15, [1, 1, 1, 1], padding='VALID', name='hidden_layer15')
            hidden_layer15 = contrib.layers.batch_norm(hidden_layer15, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer15 = tf.nn.leaky_relu(hidden_layer15)
            print(hidden_layer15)

            # Layer16
            W16 = tf.get_variable("W16", shape=[3, 3, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer16 = tf.nn.conv2d(hidden_layer15, W16, [1, 1, 1, 1], padding='SAME', name='hidden_layer16')
            hidden_layer16 = contrib.layers.batch_norm(hidden_layer16, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer16 = tf.nn.leaky_relu(hidden_layer16)
            print(hidden_layer16)

            # Layer17
            W17= tf.get_variable("W17", shape=[1, 1, 1024, 512], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer17 = tf.nn.conv2d(hidden_layer16, W17, [1, 1, 1, 1], padding='VALID', name='hidden_layer17')
            hidden_layer17 = contrib.layers.batch_norm(hidden_layer17, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer17 = tf.nn.leaky_relu(hidden_layer17)
            print(hidden_layer17)

            # Layer18
            W18 = tf.get_variable("W18", shape=[3, 3, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer18 = tf.nn.conv2d(hidden_layer17, W18, [1, 1, 1, 1], padding='SAME', name='hidden_layer18')
            hidden_layer18 = contrib.layers.batch_norm(hidden_layer18, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer18 = tf.nn.leaky_relu(hidden_layer18)
            print(hidden_layer18)

            # Layer19
            W19 = tf.get_variable("W19", shape=[1, 1, 1024, self.class_count], initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer19 = tf.nn.conv2d(hidden_layer18, W19, [1, 1, 1, 1], padding='VALID', name='hidden_layer19')
            hidden_layer19 = contrib.layers.batch_norm(hidden_layer19, center=True, scale=True, is_training=self.keep_layer)
            hidden_layer19 = tf.nn.leaky_relu(hidden_layer19)
            print(hidden_layer19)


            # Global Average Pooling
            #self.average_polling = tf.reduce_mean(hidden_layer19, axis=(1, 2), name="GAP")
            #print(self.average_polling)
            self.average_polling = tf.layers.average_pooling2d(hidden_layer19, pool_size=[4, 4], strides=1)
            self.average_polling = tf.layers.flatten(self.average_polling)
            print(self.average_polling)

            self.output = tf.nn.softmax(self.average_polling, -1, 'output');
            print(self.output)
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # define cost/loss & optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.average_polling, labels=self.Y))
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


    def predict(self, x_test, keep_prop=False):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=False):
        #print(self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop}))
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_layer: [keep_prop]})

    def train(self, x_data, y_data, keep_prop=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_layer: keep_prop})

