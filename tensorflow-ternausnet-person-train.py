import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

from models.model_ternausnet import model_ternausnet
from util.segmentation_dataloader_v1 import segmentation_dataloader_v1

sample_loader = segmentation_dataloader_v1('D://coco-dataset//coco-person//validation-input//', 'D://coco-dataset//coco-person//validation-label//')

train_epoch = 500
batch_size = 1
sample_size = sample_loader.size()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.95;

sess = tf.Session()
model = model_ternausnet(sess=sess, name="person_ternausnet", learning_rate=0.1)
sess.run(tf.global_variables_initializer())


for step in range(train_epoch):

    sample_loader.clear()
    avg_cost = 0
    accuracy = 0
    for batch in range(total_batch):
        input_images, input_labels = sample_loader.load([256*256*3], [256*256*1], 1, 255, batch_size)
        if input_images is None:
            sample_loader.clear()
            break


        cost, _ = model.train(input_images, input_labels, keep_prop=True)
        avg_cost += cost / total_batch

    accuracy = model.get_accuracy(input_images, input_labels, keep_prop=False)

    print('Epoch : ', '%04d' % (step + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(accuracy))
    if accuracy > target_accuracy:
        break;
