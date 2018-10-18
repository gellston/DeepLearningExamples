import tensorflow as tf
import matplotlib.pyplot as plt
from util.datasetloader import datasetloader
from util.datasetloader import pathtype
from models.model_darknet19 import model_darknet19

loader_train = datasetloader('/Development/DeepLearningExamples/dataset/animal-train-v1', pathtype.absolute)
loader_validation = datasetloader('/Development/DeepLearningExamples/dataset/animal-validation-v1', pathtype.absolute)

classCount = loader_validation.label_count()
validationCount = loader_validation.sample_count()

train_epoch = 500
batch_size = 100
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.95

## cat 0, dog 1, elephant 2, giraffe 3, hourse 4
sess = tf.Session()
model1 = model_darknet19(sess=sess, name="animal-darknet19", class_count=classCount, learning_rate=0.00005)
sess.run(tf.global_variables_initializer())

print('learning started')

cost_graph = []
accuracy_graph = []

for epoch in range(train_epoch):
    avg_cost = 0
    accuracy_cost = 0
    loader_train.clear()

    for i in range(total_batch):
        inputs_train, outputs_train = loader_train.load([100*100*3], 1, batch_size)
        if inputs_train is None or outputs_train is None:
            loader_train.clear()
            break
        c, _ = model1.train(inputs_train, outputs_train, True)
        avg_cost += (c / total_batch)

    inputs_validation, output_validation = loader_validation.load([100*100*3], 1, validationCount)
    loader_validation.clear()
    accuracy_value = model1.get_accuracy(inputs_validation, output_validation, False)
    accuracy_graph.append(accuracy_value)
    cost_graph.append(avg_cost)
    print('Epoch : ', '%04d' %(epoch + 1), 'cost =',  '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(accuracy_value))

    if accuracy_value >= target_accuracy:
        break

saver = tf.train.Saver()
saver.save(sess, './pretrained-models/animal-darknet19/animal-darknet19')

plt.plot(cost_graph)
plt.plot(accuracy_graph)
plt.ylabel('cost, accuracy')
plt.legend(['cost', 'accuracy'], loc='upper left')
plt.savefig('./pretrained-models/animal-darknet19/animal-darknet19-graph.png')
plt.show()

print('Learning finished.')