import tensorflow as tf
import tensorflow.contrib as contrib


### from here for residual layer (model_custom_densenet_segmentation_v1)
def dense_layer(name, input, filters, is_dropout, is_batch_norm, initializer):
    encode = contrib.layers.batch_norm(input, center=True, scale=True, is_training=is_batch_norm)
    encode = tf.nn.relu(encode)
    encode = tf.layers.conv2d(encode, filters=filters, kernel_size=[3, 3], strides=[1, 1], padding='SAME', dilation_rate=[1, 1], activation=None, kernel_initializer=initializer, name=name + '_conv3x3')
    #encode = tf.layers.separable_conv2d(encode, filters=filters, kernel_size=[3, 3], strides=[1, 1], use_bias=False, padding='SAME', dilation_rate=[1, 1], activation=None, pointwise_initializer=initializer, depthwise_initializer=initializer, name=name + '_conv3x3')
    encode = tf.layers.dropout(encode, rate=0.1, training=is_dropout, name=name + '_dropout')
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
    x = tf.layers.conv2d(x, filters=filters, kernel_size=[1, 1], strides=[1, 1], use_bias=False, padding='SAME', dilation_rate=[1, 1], activation=None, kernel_initializer=initializer,  name=name+'_conv1x1')
    x = tf.layers.dropout(x, rate=0.1, training=is_dropout, name=name + '_dropout')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name+'_maxpool2x2')
    return x

def transition_up(name, x, filters, initializer):
    x = tf.layers.conv2d_transpose(x, filters=filters,  kernel_size=[3, 3], strides=[2, 2], use_bias=False, padding='SAME', activation=None, kernel_initializer=initializer, name=name+'_trans_conv3x3')
    return x



### from here for residual layer (model_custom_mobile_segmentation)
def residual_layer(name, input, filters, is_dropout, is_batch_norm, initializer):
    encode = contrib.layers.batch_norm(input, center=True, scale=True, is_training=is_batch_norm)
    encode = tf.nn.relu(encode)
    encode = tf.layers.separable_conv2d(encode, filters=filters, kernel_size=[3, 3], strides=[1, 1], use_bias=False, padding='SAME', activation=None, pointwise_initializer=initializer, depthwise_initializer=initializer, name=name + '_conv3x3')
    return encode

def residual_block(name, x, block_depth, is_dropout, is_batch_norm, initializer):
    identity = x
    x = residual_layer(name + 'layer1', x, block_depth, is_dropout, is_batch_norm, initializer)
    x = residual_layer(name + 'layer2', x, block_depth, is_dropout, is_batch_norm, initializer)
    output = x + identity
    return output


def transition_down3x3(name, x, filters, is_dropout, is_batch_norm, initializer):
    x = contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_batch_norm)
    x = tf.nn.relu(x, name=name + 'relu')
    x = tf.layers.separable_conv2d(x, filters=filters, kernel_size=[1, 1], strides=[1, 1], use_bias=False, padding='SAME', activation=None, pointwise_initializer=initializer, depthwise_initializer=initializer, name=name + '_conv1x1')
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name=name + '_maxpool3x3')
    return x


