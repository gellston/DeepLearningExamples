import tensorflow as tf
import cv2 as cv2
import numpy as np


sess = tf.Session()
saver = tf.train.import_meta_graph('./animal_trained-model(v1)/animal_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./animal_trained-model(v1)'))

graph = tf.get_default_graph()
output = graph.get_tensor_by_name("AnimalClassifier/output:0")
input = graph.get_tensor_by_name("AnimalClassifier/input:0")
dropout = graph.get_tensor_by_name("AnimalClassifier/dropout:0")

image = []
npImage = []
image.append(cv2.imread('./animal_validation/0_cat/1_1_0_cat.jpg'))
image.append(cv2.imread('./animal_validation/1_dog/10_1_1_dog.jpg'))
image.append(cv2.imread('./animal_validation/0_cat/15_1_0_cat.jpg'))

for index in range(3):
    temp = np.array(image[index])
    temp = temp.flatten().reshape([100*100*3])
    npImage.append(temp)

feed_dict = {input: npImage, dropout: 1}

prediction = sess.run(output, feed_dict)

print ('=== prediction values ===')
print(prediction[0], '\n')
print(prediction[1], '\n')
print(prediction[2], '\n')


cv2.imshow('image' + str(0), image[0])
cv2.imshow('image' + str(3), image[1])
cv2.imshow('image' + str(7), image[2])
cv2.waitKey(0)
