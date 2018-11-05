import tensorflow as tf


from util.custom_util import dense_block
from util.custom_util import transition_down
from util.custom_util import transition_up


class model_custom_mobile_segmentation_v2:

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

            encode1_1 = tf.layers.conv2d(self.X_input, filters=48, kernel_size=[3, 3], strides=[1, 1], padding='SAME', dilation_rate=[1, 1], activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='first_conv3x3')
            encode1_2 = dense_block('encode1_1', encode1_1, 4, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            concat1 = tf.concat([encode1_1, encode1_2], axis=3)
            transition_down1 = transition_down('transition_down1', concat1, concat1.get_shape()[-1], self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            print(transition_down1)

            encode2_1 = dense_block('encode2_1', transition_down1, 5, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            concat2 = tf.concat([encode2_1, transition_down1], axis=3)
            transition_down2 = transition_down('transition_down2', concat2, concat2.get_shape()[-1], self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            print(transition_down2)

            encode3_1 = dense_block('encode3_1', transition_down2, 7, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            concat3 = tf.concat([encode3_1, transition_down2], axis=3)
            transition_down3 = transition_down('transition_down3', concat3, concat3.get_shape()[-1], self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            print(transition_down3)

            encode4_1 = dense_block('encode4_1', transition_down3, 10, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            concat4 = tf.concat([encode4_1, transition_down3], axis=3)
            transition_down4 = transition_down('transition_down4', concat4, concat4.get_shape()[-1], self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            print(transition_down4)

            encode5_1 = dense_block('encode5_1', transition_down4, 12, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            concat5 = tf.concat([encode5_1, transition_down4], axis=3)
            transition_down5 = transition_down('transition_down5', concat5, concat5.get_shape()[-1], self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            print(transition_down5)

            middle = dense_block('middle', transition_down5, 15, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            transition_up1 = transition_up('transition_up1', middle, middle.get_shape()[-1], initializer=tf.initializers.he_uniform())
            concat6 = tf.concat([concat5, transition_up1], axis=3)
            print('middle', concat6)

            decode1_1 = dense_block('decode1_1', concat6, 12, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            transition_up2 = transition_up('transitin_up2', decode1_1, decode1_1.get_shape()[-1], initializer=tf.initializers.he_uniform())
            concat7 = tf.concat([concat4, transition_up2], axis=3)
            print(concat7)

            decode2_1 = dense_block('decode2_1', concat7, 10, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            transition_up3 = transition_up('transitin_up3', decode2_1, decode2_1.get_shape()[-1], initializer=tf.initializers.he_uniform())
            concat8 = tf.concat([concat3, transition_up3], axis=3)
            print(concat8)

            decode3_1 = dense_block('decode3_1', concat8, 7, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            transition_up4 = transition_up('transitin_up4', decode3_1, decode3_1.get_shape()[-1], initializer=tf.initializers.he_uniform())
            concat9 = tf.concat([concat2, transition_up4], axis=3)
            print(concat9)

            decode4_1 = dense_block('decode4_1', concat9, 5, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            transition_up5 = transition_up('transitin_up5', decode4_1, decode4_1.get_shape()[-1], initializer=tf.initializers.he_uniform())
            concat10 = tf.concat([concat1, transition_up5], axis=3)
            print(concat10)

            decode5_1 = dense_block('decoded2_1', concat10, 4, self.keep_layer, self.keep_layer, initializer=tf.initializers.he_uniform())
            decode5_2 = tf.layers.conv2d(decode5_1, filters=1, kernel_size=[1, 1], strides=[1, 1], padding='SAME', dilation_rate=[1, 1], activation=None,  kernel_initializer=tf.initializers.he_uniform(), name='last_conv1x1')
            print(decode5_2)

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
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def reconstruct(self, x_test, keep_prop=False):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop, self.keep_layer:keep_prop})

    def train(self, x_data, y_data, keep_prop=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_layer: keep_prop, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_layer: keep_prop, self.keep_layer:keep_prop})
