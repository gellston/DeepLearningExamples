import tensorflow as tf
import matplotlib.pyplot as plt
from util.datasetloader import datasetloader
from util.datasetloader import pathtype


loader_train = datasetloader('/digits_train', pathtype.relative)
loader_validation = datasetloader('/digits_validation', pathtype.relative)

classCount = loader_train.label_count()
validationCount =  loader_validation.sample_count()

## one 0 ~ nine 9

# placeholder 100x100 = 10000
X = tf.placeholder(tf.float32, [None, 28*28*3], name='input')
print(X)
# input shape should be 2 dimension
X_input = tf.reshape(X, [-1, 28, 28, 3])
# Output should be same as class count
Y = tf.placeholder(tf.float32, [None, classCount])
print(Y)

# Layer1
print('\n\nlayer1')
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
print(X_input)
layer1 = tf.nn.conv2d(X_input, W1, [1, 1, 1, 1], padding='SAME', name='layer1')
print(layer1)
layer1 = tf.nn.relu(layer1)
print(layer1)
layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer1)

print('\n\nlayer2')
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
layer2 = tf.nn.conv2d(layer1, W2, [1, 1, 1, 1], padding='SAME', name='layer2')
print(layer2)
layer2 = tf.nn.relu(layer2)
print(layer2)
layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer2)
layer2 = tf.reshape(layer2, [-1, 7*7*64])
print(layer2)


# fully connected layer
W3 = tf.get_variable('W3', shape=[7*7*64, classCount], initializer=tf.contrib.layers.xavier_initializer())
bias = tf.Variable(tf.random_normal([classCount]))
hypothesis = tf.matmul(layer2, W3) + bias
output = tf.nn.softmax(hypothesis, -1, 'output')
print(output)


# define cost function & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


print('learning started')

train_epoch = 10
batch_size = 100
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.98
accuracy_count = 0

cost_graph = []
accuracy_graph = []


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(train_epoch):
    avg_cost = 0
    accuracy_cost = 0

    loader_train.clear()
    for i in range(total_batch):
        inputs_train, outputs_train = loader_train.load([28*28*3], 1, batch_size)
        if inputs_train is None or outputs_train is None:
            loader_train.clear()
            break
        tf.argmax(outputs_train)
        feed_dict = {X: inputs_train, Y: outputs_train}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

    inputs_validation, output_validation = loader_validation.load([28*28*3], 1, validationCount)
    loader_validation.clear()
    accuracy_cost = sess.run(correct_prediction, feed_dict={X: inputs_validation, Y: output_validation})

    accuracy_graph.append(accuracy_cost)
    cost_graph.append(avg_cost)

    print('Epoch : ', '%04d' %(epoch + 1), 'cost =',  '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(accuracy_cost))

    if accuracy_cost >= target_accuracy:
        accuracy_count = accuracy_count + 1
    if accuracy_count == 6:
        break;


saver = tf.train.Saver()
saver.save(sess, './digits-trained-model/digits_model')

plt.plot(cost_graph)
plt.plot(accuracy_graph)
plt.ylabel('cost, accuracy')
plt.legend(['cost', 'accuracy'], loc='upper left')
plt.savefig('./digits-trained-model/pre-trained-digits-graph.png')
plt.show()

print('Learning finished. ')

