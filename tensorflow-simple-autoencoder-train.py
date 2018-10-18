import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

from util.datasetloader import datasetloader
from util.datasetloader import pathtype
from models.model_animal_autoencoder_v1 import model_animal_autoencoder_v1

loader_train = datasetloader('D://DeepLearning//dataset//animal_train-v1', pathtype.absolute)
loader_validation = datasetloader('D://DeepLearning//dataset//animal_train-v1', pathtype.absolute)

train_epoch = 500
batch_size = 100
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.95;

sess = tf.Session()
model1 = model_animal_autoencoder_v1(sess=sess, name="AnimalAutoEncoderClassifier", learning_rate=0.00005)
sess.run(tf.global_variables_initializer())

print('learning started')


cost_graph = []
accuracy_graph = []
for step in range(train_epoch):

    loader_train.clear()
    avg_cost = 0

    for batch in range(total_batch):
        inputs_train, _ = loader_train.load([100*100*3], 1, batch_size)

        if inputs_train is None:
            loader_train.clear()
            break

        cost, _ = model1.train(inputs_train, keep_prop=0.7)
        avg_cost += cost / total_batch

    accuracy = model1.get_accuracy(inputs_train, keep_prop=1.0)
    cost_graph.append(avg_cost)
    accuracy_graph.append(accuracy)

    print('Epoch : ', '%04d' % (step + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(accuracy))
    if accuracy > target_accuracy:
        break;

    loader_validation.clear()
    inputs_validation, _ = loader_validation.load([100,100,3], 1, 1)

    cv2.imshow('validation_input', inputs_validation[0])

    loader_validation.clear()
    inputs_validation, _ = loader_validation.load([100*100*3], 1, 1)
    results = model1.reconstruct(inputs_validation,keep_prop=1.0)
    output = results[0];
    output = np.array(output, dtype=np.uint8)

    cv2.imshow('validation_output', output)
    cv2.imwrite('D://DeepLearning//pretrained-models//autoencoder_trained-model(v1)//reconstruct_image' + str(step) + '.jpg', output)
    cv2.waitKey(10)

saver = tf.train.Saver()
saver.save(sess, './pretrained-models/autoencoder_trained-model(v1)/autoencoder_model')

plt.plot(cost_graph)
plt.plot(accuracy_graph)
plt.ylabel('cost, accuracy')
plt.legend(['cost', 'accuracy'], loc='upper left')
plt.savefig('./pretrained-models/autoencoder_trained-model(v1)/pre-trained-animal-graph.png')
plt.show()

print('Learning finished.')



