import tensorflow as tf


from util.custom_util import residual_block
from util.custom_util import transition_down3x3
from util.custom_util import transition_up



class model_custom_mobile_segmentation_v3:

    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            # placeholder 100x100 = 10000
            self.X = tf.placeholder(tf.float32, [None, 256 * 256 * 3], name='input')
            print(self.X)
            self.Y = tf.placeholder(tf.float32, [None, 256 * 256 * 1], name='output')

            self.keep_layer = tf.placeholder(tf.bool, name='phase')
            print(self.keep_layer)

            self.X_input = tf.reshape(self.X, [-1, 256, 256, 3])
            self.Y_input = tf.reshape(self.Y, [-1, 256, 256, 1])

            print('=== network structure ===')
            print(self.X_input)



            encode1 = tf.layers.separable_conv2d(self.X_input, filters=8, kernel_size=[3, 3], strides=[1, 1],
                                                 use_bias=False, padding='SAME', dilation_rate=[1, 1], activation=None,
                                                 pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                                 depthwise_initializer=tf.contrib.layers.xavier_initializer(), name='encode1_1')
            encode1 = residual_block('encode1_2', encode1, 8, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            transition_down1 = transition_down3x3('transition_down1', encode1, 32, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            print(transition_down1)


            encode2 = residual_block('encode2', transition_down1, 32, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            transition_down2 = transition_down3x3('transition_down2', encode2, 64, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            print(transition_down2)


            encode3 = residual_block('encode3', transition_down2, 64, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            transition_down3 = transition_down3x3('transition_down3', encode3, 128, self.keep_layer, self.keep_layer,  tf.contrib.layers.xavier_initializer())
            print(transition_down3)


            encode4 = residual_block('encode4_1', transition_down3, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            encode4 = residual_block('encode4_2', encode4, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            transition_down4 = transition_down3x3('transition_down4', encode4, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            print(transition_down4)


            encode5 = residual_block('encode5_1', transition_down3, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            encode5 = residual_block('encode5_2', encode5, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            encode5 = residual_block('encode5_3', encode5, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            encode5 = residual_block('encode5_4', encode5, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            encode5 = residual_block('encode5_5', encode5, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            encode5 = residual_block('encode5_6', encode5, 128, self.keep_layer, self.keep_layer, tf.contrib.layers.xavier_initializer())
            transition_up1 = transition_up('transition_up1', encode5, 128, tf.contrib.layers.xavier_initializer())
            print(transition_up1)


            print(transition_up4)

            self.output = tf.nn.sigmoid(decode5_2)
            print('=== network structure ===')

            pre = tf.cast(self.output > 0.5, dtype=tf.float32)
            truth = tf.cast(self.Y_input > 0.5, dtype=tf.float32)
            inse = tf.reduce_sum(tf.multiply(pre, truth), axis=(1, 2, 3))  # AND
            union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=(1, 2, 3))  # OR
            batch_iou = (inse + 1e-5) / (union + 1e-5)
            self.accuracy = tf.reduce_mean(batch_iou, name='iou_coe1')


            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_input, logits=decode5_2))
            #self.cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.output, self.Y_input), 1), name='mse')
            #self.cost = tf.sqrt(tf.reduce_mean(tf.pow(self.Y_input - self.output, 2)))
            #self.cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_input, logits=decode5_2)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def reconstruct(self, x_test, keep_prop=False):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop, self.keep_layer:keep_prop})

    def train(self, x_data, y_data, keep_prop=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_layer: keep_prop, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_layer: keep_prop, self.keep_layer:keep_prop})
