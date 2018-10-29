import tensorflow as tf
import tensorflow.contrib as contrib



def conv_batch_leaky(name, input, shape, is_pooling, is_batch_norm):
    # Layer1
    W = tf.get_variable(name + "_W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    encode = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME', name=name + '_conv')
    encode = contrib.layers.batch_norm(encode, center=True, scale=True, is_training=is_batch_norm)
    encode = tf.nn.leaky_relu(encode)
    if(is_pooling == True):
        pool_layer = tf.nn.max_pool(encode, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool_layer
    else:
        return encode

def deconv_batch_leaky(name,input, in_shape, out_shape, is_batch_norm):
    W = tf.get_variable(name + "_W", shape=in_shape, initializer=tf.contrib.layers.xavier_initializer())
    decode = tf.nn.conv2d_transpose(input, W, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME')
    decode = contrib.layers.batch_norm(decode, center=True, scale=True, is_training=is_batch_norm)
    decode = tf.nn.leaky_relu(decode)
    return decode

