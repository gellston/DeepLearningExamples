import tensorflow as tf
import datasetloader as loader
import matplotlib.pyplot as plt

loader_train = loader.datasetloader('/animal_train', loader.pathtype.relative)
loader_validation = loader.datasetloader('/animal_validation', loader.pathtype.relative)

classCount = loader_train.label_count()
validationCount =  loader_validation.sample_count()

train_epoch = 500
batch_size =  10
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.90
accuracy_count = 0


## cat 0, dog 1, elephant 2, giraffe 3, hourse 4

# placeholder 100x100 = 10000
X = tf.placeholder(tf.float32, [None, 224*224*3], name='input')
print(X)
# input shape should be 2 dimension
X_input = tf.reshape(X, [-1, 224, 224, 3])
# Output should be same as class count
Y = tf.placeholder(tf.float32, [None, classCount])
print(Y)
# Drop out
keep_layer = tf.placeholder(tf.float32, name='dropout')
print(keep_layer)



# Layer1
print('\n\nlayer1')
W1 = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
print(X_input)
layer1 = tf.nn.conv2d(X_input, W1, [1, 1, 1, 1], padding='SAME', name='layer1')
print(layer1)
layer1 = tf.nn.relu(layer1)
print(layer1)


W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
print(layer1)
layer1 = tf.nn.conv2d(layer1, W2, [1, 1, 1, 1], padding='SAME', name='layer1')
print(layer1)
layer1 = tf.nn.relu(layer1)
print(layer1)
layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer1)



# Layer2
print('\n\nlayer2')
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
print(layer1)
layer2 = tf.nn.conv2d(layer1, W3, [1, 1, 1, 1], padding='SAME', name='layer2')
print(layer2)
layer2 = tf.nn.relu(layer2)
print(layer2)

W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
print(layer2)
layer2 = tf.nn.conv2d(layer2, W4, [1, 1, 1, 1], padding='SAME', name='layer2')
print(layer2)
layer2 = tf.nn.relu(layer2)
print(layer2)
layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer2)


# Layer3
print('\n\nlayer3')
W5 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
print(layer2)
layer3 = tf.nn.conv2d(layer2, W5, [1, 1, 1, 1], padding='SAME', name='layer3')
print(layer3)
layer3 = tf.nn.relu(layer3)
print(layer3)


W6 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
print(layer3)
layer3 = tf.nn.conv2d(layer3, W6, [1, 1, 1, 1], padding='SAME', name='layer3')
print(layer3)
layer3 = tf.nn.relu(layer3)
print(layer3)
layer3 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer3)


# Layer4
print('\n\nlayer4')
W7 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
print(layer3)
layer4 = tf.nn.conv2d(layer3, W7, [1, 1, 1, 1], padding='SAME', name='layer4')
print(layer4)
layer4 = tf.nn.relu(layer4)
print(layer4)


W8 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
print(layer4)
layer4 = tf.nn.conv2d(layer4, W8, [1, 1, 1, 1], padding='SAME', name='layer4')
print(layer4)
layer4 = tf.nn.relu(layer4)
print(layer4)
layer4 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer4)



# Layer5
print('\n\nlayer5')
W9 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
print(layer4)
layer5 = tf.nn.conv2d(layer4, W9, [1, 1, 1, 1], padding='SAME', name='layer5')
print(layer5)
layer5 = tf.nn.relu(layer5)
print(layer5)


W10 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
print(layer5)
layer5 = tf.nn.conv2d(layer5, W10, [1, 1, 1, 1], padding='SAME', name='layer5')
print(layer5)
layer5 = tf.nn.relu(layer5)
print(layer5)
layer5 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(layer5)



# fully connected layer
flatten = tf.contrib.layers.flatten(layer5)
print('\n\nfcn')
fcn1 = tf.layers.dense(flatten, 4096, activation=tf.nn.relu)
print(fcn1)
fcn1 = tf.nn.dropout(fcn1, keep_prob=keep_layer)
print(fcn1)
fcn2 = tf.layers.dense(fcn1, 2, activation=tf.nn.relu)
print(fcn2)
output = tf.nn.softmax(fcn2,  -1, 'output');
print(output)




# define cost function & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fcn2, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00146).minimize(cost)


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
        inputs_train, outputs_train = loader_train.load([224*224*3], 1, batch_size)
        if inputs_train is None or outputs_train is None:
            loader_train.clear()
            break

        tf.argmax(outputs_train)
        feed_dict = {X: inputs_train, Y: outputs_train, keep_layer: 0.8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

    inputs_validation, output_validation = loader_validation.load([224*224*3], 1, validationCount)

    loader_validation.clear()
    accuracy_cost = sess.run(correct_prediction, feed_dict={X: inputs_validation, Y: output_validation, keep_layer: 1.0})

    accuracy_graph.append(accuracy_cost)
    cost_graph.append(avg_cost)

    print('Epoch : ', '%04d' %(epoch + 1), 'cost =',  '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(accuracy_cost))

    if accuracy_cost >= target_accuracy:
        accuracy_count = accuracy_count + 1
    if accuracy_count == 6:
        break;


saver = tf.train.Saver()
saver.save(sess, './animal-trained-model/animal_model')

plt.plot(cost_graph)
plt.plot(accuracy_graph)
plt.ylabel('cost, accuracy')
plt.legend(['cost', 'accuracy'], loc='upper left')
plt.savefig('./animal-trained-model/pre-trained-animal-graph.png')
plt.show()

print('Learning finished. ')

