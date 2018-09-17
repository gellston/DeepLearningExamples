import tensorflow as tf
import datasetloader as loader


loader = loader.datasetloader('/animal', loader.pathtype.relative)
classCount = loader.label_count()


## cat 0 , dog 1, horse 2

# placeholder 100x100 = 10000
X = tf.placeholder(tf.float32, [None, 30000], name='Input')
print(X)
# input shape should be 2 dimension
X_input = tf.reshape(X, [-1, 100, 100, 3])
# Output should be same as class count
Y = tf.placeholder(tf.float32, [None, classCount])
print(Y)

# Layer1
print('\n\nlayer1')
W1 = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
print(X_input)
layer1 = tf.nn.conv2d(X_input, W1, [1, 1, 1, 1], padding='SAME', name='layer1')
print(layer1)
layer1 = tf.nn.relu(layer1)
print(layer1)
layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer1)

print('\n\nlayer2')
W2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
layer2 = tf.nn.conv2d(layer1, W2, [1, 1, 1, 1], padding='SAME', name='layer2')
print(layer2)
layer2 = tf.nn.relu(layer2)
print(layer2)
layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer2)
layer2 = tf.reshape(layer2, [-1, 25*25*128])
print(layer2)


# fully connected layer
W3 = tf.get_variable('W3', shape=[25*25*128, classCount], initializer=tf.contrib.layers.xavier_initializer())
bias = tf.Variable(tf.random_normal([classCount]))
hypothesis = tf.matmul(layer2, W3) + bias
Output = tf.nn.softmax(hypothesis, name='output')
print(Output)


# define cost function & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


print('learning started')

train_epoch = 22
batch_size = 10
sample_size = loader.sample_count()
total_batch = int(sample_size / batch_size)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(train_epoch):
    avg_cost = 0
    avg_sum = 0
    loader.clear()
    while True:
        inputs, outputs = loader.load([30000], 255, train_epoch)
        if inputs is None or outputs is None:
            loader.clear()
            break
        feed_dict = {X: inputs, Y: outputs}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_sum += c

    avg_cost = avg_sum / total_batch
    print('Epoch : ', '%04d' %(epoch + 1), 'cost =',  '{:.9f}'.format(avg_cost))


saver = tf.train.Saver()
saver.save(sess, './pre-trained-model/animal.model')

print('Learning finished. ')

