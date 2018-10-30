import tensorflow as tf
import numpy as np

import tensorflow.contrib as contrib
from util.custom_util import conv_batch_leaky
from util.custom_util import deconv_batch_leaky
from util.custom_util import conv_batch


# 0 cat , 1 dog, 2 elephant, 3 giraffe, 4 horse

class model_ternausnet:

    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            # placeholder 100x100 = 10000
            self.X = tf.placeholder(tf.float32, [1, 256 * 256 * 3], name='input')
            print(self.X)
            self.Y = tf.placeholder(tf.float32, [1, 256 * 256 * 1], name='output')

            self.keep_layer = tf.placeholder(tf.bool, name='phase')
            print(self.keep_layer)

            self.X_input = tf.reshape(self.X, [-1, 256 , 256 , 3])
            self.Y_input = tf.reshape(self.Y, [-1, 256, 256, 1])
            print(self.X_input)
            print(self.Y_input)

            left_layer1 = conv_batch_leaky("encode_layer1", self.X_input, shape=[3, 3, 3, 64], is_pooling=False, is_batch_norm=self.keep_layer)
            polling = tf.nn.max_pool(left_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(polling)

            left_layer2 = conv_batch_leaky("encode_layer2", polling, shape=[3, 3, 64, 64], is_pooling=False,  is_batch_norm=self.keep_layer)
            left_layer3 = conv_batch_leaky("encode_layer3", left_layer2, shape=[3, 3, 64, 128], is_pooling=False, is_batch_norm=self.keep_layer)
            polling = tf.nn.max_pool(left_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(polling)

            left_layer4 = conv_batch_leaky("encode_layer4", polling, shape=[3, 3, 128, 128], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer5 = conv_batch_leaky("encode_layer5", left_layer4, shape=[3, 3, 128, 256], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer6 = conv_batch_leaky("encode_layer6", left_layer5, shape=[3, 3, 256, 256], is_pooling=False,  is_batch_norm=self.keep_layer)
            polling = tf.nn.max_pool(left_layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(polling)

            left_layer7 = conv_batch_leaky("encode_layer7", polling, shape=[3, 3, 256, 256], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer8 = conv_batch_leaky("encode_layer8", left_layer7, shape=[3, 3, 256, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer9 = conv_batch_leaky("encode_layer9", left_layer8, shape=[3, 3, 512, 512], is_pooling=False,  is_batch_norm=self.keep_layer)
            polling = tf.nn.max_pool(left_layer9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(polling)

            left_layer10 = conv_batch_leaky("encode_layer10", polling, shape=[3, 3, 512, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer11 = conv_batch_leaky("encode_layer11", left_layer10, shape=[3, 3, 512, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer12 = conv_batch_leaky("encode_layer12", left_layer11, shape=[3, 3, 512, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            polling = tf.nn.max_pool(left_layer12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(polling)

            left_layer13 = conv_batch_leaky("encode_layer13", polling, shape=[3, 3, 512, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            left_layer14 = conv_batch_leaky("encode_layer14", left_layer13, shape=[3, 3, 512, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            print(left_layer14)

            right_layer1 = deconv_batch_leaky("decode_layer1", left_layer14, in_shape=[3, 3, 256, 512], out_shape=[tf.shape(self.X_input)[0], 16, 16, 256], is_batch_norm=self.keep_layer)
            right_concat1 = tf.concat([right_layer1, left_layer12], 3)
            right_layer2 = conv_batch_leaky("decode_layer2", right_concat1, shape=[3, 3, 768, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            print(right_layer2)

            right_layer3 = deconv_batch_leaky("decode_layer3", right_layer2, in_shape=[3, 3, 256, 512], out_shape=[tf.shape(self.X_input)[0], 32, 32, 256], is_batch_norm=self.keep_layer)
            right_concat2 = tf.concat([right_layer3, left_layer9], 3)
            right_layer4 = conv_batch_leaky("decode_layer4", right_concat2, shape=[3, 3, 768, 512], is_pooling=False, is_batch_norm=self.keep_layer)
            print(right_layer4)

            right_layer5 = deconv_batch_leaky("decode_layer5", right_layer4, in_shape=[3, 3, 128, 512], out_shape=[tf.shape(self.X_input)[0], 64, 64, 128], is_batch_norm=self.keep_layer)
            right_concat3 = tf.concat([right_layer5, left_layer6], 3)
            right_layer6 = conv_batch_leaky("decode_layer6", right_concat3, shape=[3, 3, 384, 256], is_pooling=False, is_batch_norm=self.keep_layer)
            print(right_layer6)

            right_layer7 = deconv_batch_leaky("decode_layer7", right_layer6, in_shape=[3, 3, 64, 256], out_shape=[tf.shape(self.X_input)[0], 128, 128, 64], is_batch_norm=self.keep_layer)
            right_concat4 = tf.concat([right_layer7, left_layer3], 3)
            right_layer9 = conv_batch_leaky("decode_layer8", right_concat4, shape=[3, 3, 192, 128], is_pooling=False, is_batch_norm=self.keep_layer)
            print(right_layer9)

            right_layer10 = deconv_batch_leaky("decode_layer9", right_layer9, in_shape=[3, 3, 32, 128], out_shape=[tf.shape(self.X_input)[0], 256, 256, 32], is_batch_norm=self.keep_layer)
            right_concat5 = tf.concat([right_layer10, left_layer1], 3)
            right_layer11 = conv_batch("decode_layer10", right_concat5, shape=[3, 3, 96, 1], is_pooling=False, is_batch_norm=self.keep_layer)
            self.output = tf.nn.softmax(right_layer11)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.output - self.Y_input)))

                argmax_probs = tf.round(self.output)  # 0x1
                correct_pred = tf.cast(tf.equal(argmax_probs, self.Y_input), tf.float32)
                self.accuracy = tf.reduce_mean(correct_pred)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def reconstruct(self, x_test, keep_prop=False):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop})

    def train(self, x_data, y_data, keep_prop=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y:y_data, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y:y_test, self.keep_layer: keep_prop})
