import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

from models.model_custom_mobile_segmentation_v3 import model_custom_mobile_segmentation_v3
from util.segmentation_dataloader_v1 import segmentation_dataloader_v1

train_loader = segmentation_dataloader_v1('D://portrait-dataset//train_input256x256//', 'D://portrait-dataset//train_label256x256//')
validation_loader = segmentation_dataloader_v1('D://portrait-dataset//train_input256x256//', 'D://portrait-dataset//train_label256x256//')

train_epoch = 10000
batch_size = 1
sample_size = train_loader.size()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.90

sess = tf.Session()
model = model_custom_mobile_segmentation_v3(sess=sess, name="model_custom_mobile_segmentation_v3", learning_rate=1e-3)
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

cost_graph = []
accuracy_graph = []

print('Learning start.')

for step in range(train_epoch):

    train_loader.clear()
    avg_cost = 0
    avg_accuracy = 0
    accuracy = 0
    for batch in range(total_batch):
        #train_loader.clear()
        input_images, input_labels = train_loader.load([256*256*3], [256*256*1], 1, 255, batch_size)

        if input_images is None:
            train_loader.clear()
            break

        cost, _ = model.train(input_images, input_labels, keep_prop=True)
        avg_cost += cost / total_batch

    validation_loader.clear()
    for validation_step in range(100):
        validation_images, validation_labels = validation_loader.load([256*256*3], [256*256*1], 1, 255, 1)
        accuracy = model.get_accuracy(validation_images, validation_labels, keep_prop=False)
        avg_accuracy += (accuracy / 100)
        print('Validation Step: ', '%04d' % (validation_step + 1), ' step accuracy = ', '{:.9f}'.format(accuracy),)

    output_images = model.reconstruct(validation_images, keep_prop=False)
    output_reshape = np.array(output_images[0] * 255, dtype=np.uint8)
    input_reshape = np.array(np.reshape(validation_images[0], [256, 256, 3]), dtype=np.uint8)
    input_label = np.array(np.reshape(validation_labels[0], [256, 256, 1]) * 255, dtype=np.uint8)


    cv2.imshow('reconstruced image', output_reshape)
    cv2.imshow('input image', input_reshape)
    cv2.imshow('input label', input_label)
    cv2.waitKey(10)
    validation_loader.clear()

    accuracy_graph.append(accuracy)
    cost_graph.append(avg_cost)

    print('Epoch : ', '%04d' % (step + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(avg_accuracy))
    if avg_accuracy > target_accuracy:
        break;

tf.train.write_graph(sess.graph.as_graph_def(),"./pretrained-models/densenet_segmentation/", "densenet_segmentation.pb")
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './pretrained-models/densenet_segmentation/densenet_segmentation.ckpt')

plt.plot(cost_graph)
plt.plot(accuracy_graph)
plt.ylabel('cost, accuracy')
plt.legend(['cost', 'accuracy'], loc='upper left')
plt.savefig('./pretrained-models/densenet_segmentation/densenet_segmentation.png')
plt.show()

print('Learning finished.')