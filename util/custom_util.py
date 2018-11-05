import tensorflow as tf
import tensorflow.contrib as contrib

def conv(name, input, shape, initializer):
    W = tf.get_variable(name + "_W", shape=shape, initializer=initializer)
    encode = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME', name=name + '_conv3x3')
    return encode

def dense_layer(name, input, filters, is_dropout, is_batch_norm, initializer):
    encode = contrib.layers.batch_norm(input, center=True, scale=True, is_training=is_batch_norm)
    encode = tf.nn.relu(encode)
    encode = tf.layers.conv2d(encode, filters=filters, kernel_size=[3, 3], strides=[1, 1], padding='SAME', dilation_rate=[1, 1], activation=None, kernel_initializer=initializer, name=name + '_conv3x3')
    #encode = tf.nn.dropout(encode, keep_prob=is_dropout)
    encode = tf.layers.dropout(encode, rate=0.2, training=is_dropout, name=name + '_dropout')
    return encode

def dense_block(name, x, block_depth, is_dropout, is_batch_norm, initializer):
    dense_out = []
    for i in range(block_depth):
        conv = dense_layer(name + "_layer_" + str(i), x, filters=16, is_dropout=is_dropout, is_batch_norm=is_batch_norm, initializer=initializer)
        x = tf.concat([conv, x], axis=3)
        dense_out.append(conv)
    x = tf.concat(dense_out, axis=3)
    return x

def transition_down(name, x, filters, is_dropout, is_batch_norm, initializer):
    x = contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_batch_norm)
    x = tf.nn.relu(x, name=name+'relu')
    x = tf.layers.conv2d(x, filters=filters, kernel_size=[1, 1], strides=[1, 1], padding='SAME', dilation_rate=[1, 1], activation=None, kernel_initializer=initializer,  name=name+'_conv1x1')
    #x = tf.nn.dropout(x, keep_prob=is_dropout)
    x = tf.layers.dropout(x, rate=0.2, training=is_dropout, name=name + '_dropout')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name+'_maxpool2x2')

    return x

def transition_up(name, x, filters, initializer):
    x = tf.layers.conv2d_transpose(x, filters=filters,  kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation=None, kernel_initializer=initializer, name=name+'_trans_conv3x3')
    return x