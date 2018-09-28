import tensorflow as tf
import datasetloader as loader
import matplotlib.pyplot as plt

loader_train = loader.datasetloader('/animal_train', loader.pathtype.relative)
loader_validation = loader.datasetloader('/animal_validation', loader.pathtype.relative)

classCount = loader_validation.label_count()
validationCount =  loader_validation.sample_count()

train_epoch = 500
batch_size = 200
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.90



## cat 0, dog 1, elephant 2, giraffe 3, hourse 4

# placeholder 100x100 = 10000
X = tf.placeholder(tf.float32, [None, 100*100*3], name='input')
print(X)
# input shape should be 2 dimension
X_input = tf.reshape(X, [-1, 100, 100, 3])
# Output should be same as class count
Y = tf.placeholder(tf.float32, [None, classCount])
print(Y)
# Drop out
keep_layer = tf.placeholder(tf.float32, name='dropout')
print(keep_layer)



# Layer1
print('\n\nlayer1')
W1 = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=0.01))
Bias1 = tf.Variable(tf.constant(0.1, shape=[64]))
hidden_layer1 = tf.nn.relu(tf.nn.conv2d(X_input, W1, [1, 1, 1, 1], padding='SAME', name='hidden_layer1') + Bias1)
pool_layer1 = tf.nn.max_pool(hidden_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
drop_layer1 = tf.nn.dropout(pool_layer1, keep_prob=keep_layer)
print(drop_layer1)

W2 = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=0.01))
Bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
hidden_layer2 = tf.nn.relu(tf.nn.conv2d(drop_layer1, W2, [1, 1, 1, 1], padding='SAME', name='hidden_layer2') + Bias2)
pool_layer2 = tf.nn.max_pool(hidden_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
drop_layer2 = tf.nn.dropout(pool_layer2, keep_prob=keep_layer)
print(drop_layer2)


# Layer2
print('\n\nlayer2')
W3 = tf.Variable(tf.random_normal([4, 4, 64, 128], stddev=0.01))
Bias3 = tf.Variable(tf.constant(0.1, shape=[128]))
hidden_layer3 = tf.nn.relu(tf.nn.conv2d(drop_layer2, W3, [1, 1, 1, 1], padding='SAME', name='hidden_layer3') + Bias3)
pool_layer3 = tf.nn.max_pool(hidden_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
drop_layer3 = tf.nn.dropout(pool_layer3, keep_prob=keep_layer)
print(drop_layer3)

W4 = tf.Variable(tf.random_normal([4, 4, 128, 128], stddev=0.01))
Bias4 = tf.Variable(tf.constant(0.1, shape=[128]))
hidden_layer4 = tf.nn.relu(tf.nn.conv2d(drop_layer3, W4, [1, 1, 1, 1], padding='SAME', name='hidden_layer4') + Bias4)
pool_layer4 = tf.nn.max_pool(hidden_layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
drop_layer4 = tf.nn.dropout(hidden_layer4, keep_prob=keep_layer)
print(drop_layer4)



# Layer3
print('\n\nLayer3')
W5 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
Bias5 = tf.Variable(tf.constant(0.1, shape=[256]))
hidden_layer5 = tf.nn.relu(tf.nn.conv2d(drop_layer4, W5, [1, 1, 1, 1], padding='SAME', name='hidden_layer5') + Bias5)
pool_layer5 = tf.nn.max_pool(hidden_layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
drop_layer5 = tf.nn.dropout(pool_layer5, keep_prob=keep_layer)
print(drop_layer5)

W6 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
Bias6 = tf.Variable(tf.constant(0.1, shape=[256]))
hidden_layer6 = tf.nn.relu(tf.nn.conv2d(drop_layer5, W6, [1, 1, 1, 1], padding='SAME', name='hidden_layer6') + Bias6)
pool_layer6 = tf.nn.max_pool(hidden_layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
drop_layer6 = tf.nn.dropout(hidden_layer6, keep_prob=keep_layer)
print(drop_layer6)



print('\n\nfcn')
flatten = tf.reshape(drop_layer6, [-1, 7*7*256])
print(flatten)
fcn1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
fcn1_dropout = tf.nn.dropout(fcn1, keep_prob=keep_layer)
print(fcn1_dropout)
fcn2 = tf.layers.dense(fcn1_dropout, classCount)
print(fcn2)
output = tf.nn.softmax(fcn2,  -1, 'output');
print(output)




# define cost function & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fcn2, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(cost)


correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('learning started')


cost_graph = []
accuracy_graph = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(train_epoch):
    avg_cost = 0
    accuracy_cost = 0

    loader_train.clear()
    for i in range(total_batch):
        inputs_train, outputs_train = loader_train.load([100*100*3], 1, batch_size)
        if inputs_train is None or outputs_train is None:
            loader_train.clear()
            break

        feed_dict = {X: inputs_train, Y: outputs_train, keep_layer: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    inputs_validation, output_validation = loader_validation.load([100*100*3], 1, validationCount)

    loader_validation.clear()
    accuracy_value = sess.run(accuracy, feed_dict={X: inputs_validation, Y: output_validation, keep_layer: 1.0})
    #accuracy_cost = correct_prediction.eval(feed_dict={X: inputs_validation, Y: output_validation, keep_layer: 1.0})

    accuracy_graph.append(accuracy_value)
    cost_graph.append(avg_cost)

    print('Epoch : ', '%04d' %(epoch + 1), 'cost =',  '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(accuracy_value))

    if accuracy_value >= target_accuracy:
        break

saver = tf.train.Saver()
saver.save(sess, './animal-trained-model/animal_model')

plt.plot(cost_graph)
plt.plot(accuracy_graph)
plt.ylabel('cost, accuracy')
plt.legend(['cost', 'accuracy'], loc='upper left')
plt.savefig('./animal-trained-model/pre-trained-animal-graph.png')
plt.show()

print('Learning finished. ')

