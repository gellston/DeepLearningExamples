import tensorflow as tf
import tensorflow.contrib as contrib

class model_yolo_v2_detection:

    def __init__(self, sess, name, num_anchor, num_classes, learning_rate):
        self.scale = 32
        self.grid_w, self.grid_h = 13, 13
        self.image_height, self.image_width, self.image_depth = self.grid_h * self.scale, self.grid_w * self.scale, 3

        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_anchor = num_anchor
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # placeholder 100x100 = 10000
            self.X = tf.placeholder(tf.float32, [None, 416 * 416 * 3], name='input')
            self.keep_layer = tf.placeholder(tf.bool, name='phase')
            self.X_input = tf.reshape(self.X, [-1, 416, 416, 3])
            self.Y_input = tf.placeholder(shape=[None, self.grid_w, self.grid_h, self.num_anchor, self.num_classes + 5], dtype=tf.float32)


            print('=== network structure ===')
            print(self.X_input)

            with tf.variable_scope('layer1'):
                conv = tf.layers.conv2d(self.X_input, filters=32, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                max_pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
                print(max_pool)

            with tf.variable_scope('layer2'):
                conv = tf.layers.conv2d(max_pool, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                max_pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
                print(max_pool)

            with tf.variable_scope('layer3'):
                conv = tf.layers.conv2d(max_pool, filters=128, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer4'):
                conv = tf.layers.conv2d(relu, filters=64, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer5'):
                conv = tf.layers.conv2d(relu, filters=128, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                max_pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
                print(max_pool)

            with tf.variable_scope('layer6'):
                conv = tf.layers.conv2d(max_pool, filters=256, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer7'):
                conv = tf.layers.conv2d(relu, filters=128, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer8'):
                conv = tf.layers.conv2d(relu, filters=256, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                max_pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
                print(max_pool)

            with tf.variable_scope('layer9'):
                conv = tf.layers.conv2d(max_pool, filters=512, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer10'):
                conv = tf.layers.conv2d(relu, filters=256, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer11'):
                conv = tf.layers.conv2d(relu, filters=512, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer12'):
                conv = tf.layers.conv2d(relu, filters=256, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)


            with tf.variable_scope('layer13'):
                conv = tf.layers.conv2d(relu, filters=512, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                layer13_relu = relu
                max_pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
                print(max_pool)

            with tf.variable_scope('layer14'):
                conv = tf.layers.conv2d(max_pool, filters=1024, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer15'):
                conv = tf.layers.conv2d(relu, filters=512, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)


            with tf.variable_scope('layer16'):
                conv = tf.layers.conv2d(relu, filters=1024, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer17'):
                conv = tf.layers.conv2d(relu, filters=512, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer18'):
                conv = tf.layers.conv2d(relu, filters=1024, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer19'):
                conv = tf.layers.conv2d(relu, filters=1024, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)

            with tf.variable_scope('layer20'):
                conv = tf.layers.conv2d(relu, filters=1024, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                layer20_relu = relu
                print(relu)

            with tf.variable_scope('layer21'):
                skip_connection = tf.layers.conv2d(layer13_relu, filters=64, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(skip_connection, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                skip_space_to_depth_x2 = tf.space_to_depth(relu, block_size=2)
                layer21_concatnation = tf.concat([skip_space_to_depth_x2, layer20_relu], axis=-1)
                print(layer21_concatnation)


            with tf.variable_scope('layer22'):
                conv = tf.layers.conv2d(layer21_concatnation, filters=1024, kernel_size=[3, 3], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
                batch = contrib.layers.batch_norm(conv, center=True, scale=True, is_training=self.keep_layer)
                relu = tf.nn.leaky_relu(batch)
                print(relu)


            output_channel = self.num_anchor * (5 + self.num_classes)
            self.logit = tf.layers.conv2d(relu, filters=output_channel, kernel_size=[1, 1], strides=[1, 1],
                                        use_bias=False, padding='SAME', activation=None,
                                        kernel_initializer=contrib.layers.xavier_initializer(), name='conv')
            print(self.logit)
            self.predict = tf.reshape(self.logit, shape=(-1, self.grid_w, self.grid_h, self.num_anchor, self.num_classes + 5), name='output')
            print(self.predict)
            print('=== network structure ===')

            # yolo loss function design
            mask = self.Y_input[..., 5:6]
            print(mask)
            label = self.Y_input[..., 0:5]
            print(label)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def slice_tensor(x, start, end=None):
        if end < 0:
            y = x[..., start:]
        else:
            if end is None:
                end = start
            y = x[..., 5:5 + 1]
        return y

    def reconstruct(self, x_test, keep_prop=False):
        return self.sess.run(self.output, feed_dict={self.X: x_test, self.keep_layer: keep_prop, self.keep_layer:keep_prop})

    def train(self, x_data, y_data, keep_prop=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_layer: keep_prop, self.keep_layer: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_layer: keep_prop, self.keep_layer:keep_prop})

