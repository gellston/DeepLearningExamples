import tensorflow as tf
import numpy as np
import os
import cv2 as cv2


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/animal"


classList = []
classCount = 0
for className in os.listdir(dir_path):
    if className == '.DS_Store': continue
    fullPath = dir_path + '/' + className;
    if className == 'mixed':
        continue
    if os.path.isdir(fullPath):
        classList.append([className, classCount])
        classCount = classCount + 1

print('class count : %d \n' % len(classList))
print('class =', classList, '\n')


samples = []
lables = []
mixedPath = dir_path + '/mixed'
for fileName in os.listdir(mixedPath):
    if fileName == '.DS_Store': continue
    image = cv2.imread(mixedPath + '/' + fileName)
    npImage = np.array(image)
    npImage = npImage / 255
    npImage = npImage.flatten().reshape(30000)
    print (npImage.shape)

    samples.append(npImage)
    for classInfo in classList:
        if classInfo[0] in fileName:
            label = [0] * classCount
            label[classInfo[1]] = 1
            lables.append(label)


# placeholder 100x100 = 10000
X = tf.placeholder(tf.float32, [None, 30000])
# input shape should be 2 dimension
X_input = tf.reshape(X, [-1, 100, 100, 3])
# Output should be same as class count
Y = tf.placeholder(tf.float32, [None, classCount])


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
print(hypothesis)

# define cost function & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

print('learning started')

train_epoch = 10
batch_size = 10

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(train_epoch):
    avg_cost = 0
    total_batch = int(len(samples) / batch_size)
    train_samples = []
    train_labeles= []
    for index in range(total_batch):
        for one_batch in range(batch_size):
            if index * batch_size + one_batch >= len(samples):
                break
            train_samples.append(samples[index * batch_size + one_batch])
            train_labeles.append(lables[index * batch_size  + one_batch])

        feed_dict = {X: train_samples, Y: train_labeles}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        print('Epoch : ','%04d' %(epoch + 1), 'cost =',  '{:.9f}'.format(avg_cost))


print('Learning finished. ')








print('learning started. it will takes sometimes')
